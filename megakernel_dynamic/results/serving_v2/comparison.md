| Metric | Megakernel (fp16, static) | Megakernel (fp8, static) | Megakernel (fp4, static) | Megakernel (fp16, auto) | Megakernel (fp8, auto) | Megakernel (fp4, auto) | vLLM (graphs) | vLLM (eager) |
|---|---|---|---|---|---|---|---|---|
| Avg latency | 0.024 s (**13.27x**) | 0.025 s (**12.90x**) | 0.025 s (**12.96x**) | 0.036 s (**9.07x**) | 0.041 s (**7.93x**) | 0.040 s (**8.08x**) | 0.045 s (**7.23x**) | 0.324 s |
| Median latency | 0.028 s (**15.09x**) | 0.029 s (**14.56x**) | 0.029 s (**14.56x**) | 0.040 s (**10.32x**) | 0.046 s (**9.00x**) | 0.046 s (**9.06x**) | 0.055 s (**7.54x**) | 0.416 s |
| P90 latency | 0.030 s (**14.41x**) | 0.031 s (**14.04x**) | 0.031 s (**13.95x**) | 0.044 s (**9.82x**) | 0.050 s (**8.63x**) | 0.050 s (**8.69x**) | 0.056 s (**7.80x**) | 0.435 s |
| Tokens/s (per request) | 930.8 (**12.81x**) | 908.7 (**12.51x**) | 888.5 (**12.23x**) | 635.8 (**8.75x**) | 558.8 (**7.69x**) | 553.9 (**7.62x**) | 509.4 (**7.01x**) | 72.7 |
| End-to-end throughput | 983.4 tok/s (**13.27x**) | 960.5 tok/s (**12.96x**) | 949.5 tok/s (**12.82x**) | 671.7 tok/s (**9.07x**) | 590.6 tok/s (**7.97x**) | 591.8 tok/s (**7.99x**) | 535.4 tok/s (**7.23x**) | 74.1 tok/s |
| Req/s | 40.92 (**13.27x**) | 39.77 (**12.90x**) | 39.95 (**12.96x**) | 27.95 (**9.07x**) | 24.46 (**7.93x**) | 24.90 (**8.08x**) | 22.28 (**7.23x**) | 3.08 |
| Latency min-max | 0.015-0.031 s | 0.015-0.032 s | 0.015-0.032 s | 0.022-0.046 s | 0.025-0.052 s | 0.025-0.052 s | 0.023-0.062 s | 0.120-0.459 s |
| Mean output tokens | 24.0 | 24.1 | 23.8 | 24.0 | 24.1 | 23.8 | 24.0 | 24.0 |
| Successful / failed | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 |

Comparability checks against the baseline:
  Megakernel (fp16, static): 99/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp8, static): 43/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp4, static): 13/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp16, auto): 99/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp8, auto): 43/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  Megakernel (fp4, auto): 13/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
  vLLM (graphs): 98/100 answers identical to the baseline; on those, output tokens differ by +0 (median)
