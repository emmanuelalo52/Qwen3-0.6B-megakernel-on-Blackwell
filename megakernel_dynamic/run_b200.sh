#!/bin/bash
# =============================================================================
# run_b200.sh — a whole B200 session in one command, as used for results/*_v2:
# setup, correctness tests (CUDA + CuTeDSL), per-tile traces, and the serving
# benchmark against vLLM (eager and CUDA graphs) on the 100 prompts.
#
#   git clone https://github.com/emmanuelalo52/Qwen3-0.6B-megakernel-RTX-5090.git
#   cd Qwen3-0.6B-megakernel-RTX-5090
#   nohup bash megakernel_dynamic/run_b200.sh > run_b200.log 2>&1 &
#   tail -f run_b200.log
#
# Runs in the background, so a dropped SSH connection does not stop it. Needs nvcc
# 12.8+, uv, and ~35 GB of disk (two PyTorch builds: the megakernel's and vLLM's).
#
# Environment:
#   PY=...         python for the megakernel (default /venv/main/bin/python, else python3)
#   VLLM_VENV=...  where vLLM gets its own venv (default /venv/vllm); vLLM pins its own torch
#   RUN=...        name for this run's result files (default: date and time)
#   SKIP_SERVING=1 stop after the traces
# =============================================================================

set -o pipefail
cd "$(dirname "$0")/.."
: "${PY:=$([ -x /venv/main/bin/python ] && echo /venv/main/bin/python || command -v python3)}"
: "${VLLM_VENV:=/venv/vllm}"
: "${RUN:=$(date +%Y%m%d_%H%M)}"
export PATH="$(dirname "$PY"):$PATH"
R=megakernel_dynamic/results
mkdir -p "$R"
step() { echo; echo "######## $(date +%H:%M:%S) $*"; }

step "run $RUN on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(git log --oneline -1)"

VLLM_PID=""
if [ -z "$SKIP_SERVING" ] && [ ! -x "$VLLM_VENV/bin/vllm" ]; then
    step "vLLM install in the background ($VLLM_VENV)"
    ( uv venv -q "$VLLM_VENV" --python 3.12 && uv pip install -q --python "$VLLM_VENV/bin/python" vllm \
        && echo VLLM_INSTALL_DONE ) > "$R/vllm_install_$RUN.log" 2>&1 &
    VLLM_PID=$!
fi

step "megakernel environment"
CUDA_MAJOR=$(nvcc --version | grep -o -E "release [0-9]+" | grep -o -E "[0-9]+")
TORCH_INDEX=https://download.pytorch.org/whl/cu130
[ "$CUDA_MAJOR" = "12" ] && TORCH_INDEX=https://download.pytorch.org/whl/cu128   # torch must match nvcc's major
uv pip install -q --python "$PY" torch --index-url "$TORCH_INDEX" || exit 1
uv pip install -q --python "$PY" numpy transformers nvidia-cutlass-dsl==4.4.0 fastapi uvicorn openai orjson \
    setuptools ninja || exit 1
"$PY" -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"

step "build the normal and trace modules"
( cd megakernel_dynamic/cuda && "$PY" setup.py build_ext --inplace > "../../$R/build_$RUN.log" 2>&1 \
    && DPS_TRACE=1 "$PY" setup.py build_ext --inplace >> "../../$R/build_$RUN.log" 2>&1 ) \
    || { echo BUILD_FAILED; grep -i error "$R/build_$RUN.log" | head -30; exit 1; }

step "test_dps, CUDA backend (every format x atomic / static / clc)"
"$PY" megakernel_dynamic/test_dps.py 2>&1 | grep -v "Loading weights" | tee "$R/b200_${RUN}_cuda.txt"

step "test_dps, CuTeDSL backend"
if ! "$PY" megakernel_dynamic/test_dps.py --backend cutedsl 2>&1 | grep -v "Loading weights" | tee "$R/b200_${RUN}_cutedsl.txt"; then
    step "CuTeDSL failed or timed out: hang probe (with per-CTA markers)"
    timeout 300 "$PY" megakernel_dynamic/hang_probe.py 2>&1 | grep -v "Loading weights" | tee "$R/b200_${RUN}_hang_probe.txt"
    DPS_CRUMBS=0 timeout 300 "$PY" megakernel_dynamic/hang_probe.py 2>&1 | grep -v "Loading weights" \
        | tee "$R/b200_${RUN}_hang_probe_nomarkers.txt"
fi

step "per-tile traces"
"$PY" megakernel_dynamic/trace_dps.py --formats fp16,fp8,fp4 --out-dir "$R/traces_$RUN" 2>&1 \
    | grep -v "Loading weights" | tee "$R/b200_${RUN}_traces.txt"

if [ -n "$SKIP_SERVING" ]; then step "done (SKIP_SERVING)"; exit 0; fi

step "wait for vLLM"
[ -n "$VLLM_PID" ] && wait "$VLLM_PID"
"$VLLM_VENV/bin/vllm" --version || { echo VLLM_MISSING; tail -20 "$R/vllm_install_$RUN.log"; exit 1; }

step "serving benchmark: megakernel configs, then vLLM eager and CUDA graphs"
S="$R/serving_$RUN"
DPS_CONFIGS="fp16:static fp8:static fp4:static fp16:auto" VLLM_MODES="eager graphs" \
    VLLM_BIN="$VLLM_VENV/bin/vllm" PY="$PY" OUT_DIR="$S" bash bench_serving.sh

step "comparison against vLLM with CUDA graphs"
others=()
for f in "$S"/dps_*.json "$S/vllm_eager.json"; do [ -f "$f" ] && others+=("$f"); done
"$PY" compare_benchmarks.py --baseline "$S/vllm_graphs.json" "${others[@]}" | tee "$S/comparison_vs_vllm_graphs.md"

step "ALL DONE: results in $R (b200_${RUN}_*, traces_$RUN, serving_$RUN)"
