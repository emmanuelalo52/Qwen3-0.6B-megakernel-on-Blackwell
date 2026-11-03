| Metric | Megakernel (fp16, static) | Megakernel (fp8, static) | Megakernel (fp4, static) | Megakernel (fp16, auto) | vLLM (eager) | vLLM (graphs) |
|---|---|---|---|---|---|---|
| Avg latency | 0.024 s (**1.84x**) | 0.025 s (**1.79x**) | 0.025 s (**1.79x**) | 0.036 s (**1.25x**) | 0.324 s (**0.14x**) | 0.045 s |
| Median latency | 0.028 s (**2.00x**) | 0.029 s (**1.93x**) | 0.029 s (**1.93x**) | 0.040 s (**1.37x**) | 0.416 s (**0.13x**) | 0.055 s |
| P90 latency | 0.030 s (**1.85x**) | 0.031 s (**1.80x**) | 0.031 s (**1.79x**) | 0.044 s (**1.26x**) | 0.435 s (**0.13x**) | 0.056 s |
| Tokens/s (per request) | 930.8 (**1.83x**) | 908.7 (**1.78x**) | 888.5 (**1.74x**) | 635.8 (**1.25x**) | 72.7 (**0.14x**) | 509.4 |
| End-to-end throughput | 983.4 tok/s (**1.84x**) | 960.5 tok/s (**1.79x**) | 949.5 tok/s (**1.77x**) | 671.7 tok/s (**1.25x**) | 74.1 tok/s (**0.14x**) | 535.4 tok/s |
| Req/s | 40.92 (**1.84x**) | 39.77 (**1.79x**) | 39.95 (**1.79x**) | 27.95 (**1.25x**) | 3.08 (**0.14x**) | 22.28 |
| Latency min-max | 0.015-0.031 s | 0.015-0.032 s | 0.015-0.032 s | 0.022-0.046 s | 0.120-0.459 s | 0.023-0.062 s |
| Mean output tokens | 24.0 | 24.1 | 23.8 | 24.0 | 24.0 | 24.0 |
| Successful / failed | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 |

Comparability checks against the baseline:
  Megakernel (fp16, static): 97/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp8, static): 43/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp4, static): 13/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp16, auto): 97/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  vLLM (eager): 98/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
