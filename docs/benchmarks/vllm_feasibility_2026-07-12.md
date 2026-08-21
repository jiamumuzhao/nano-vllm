# vLLM Feasibility - 2026-07-12

## Status

vLLM was measured under the strict controlled offline benchmark protocol using a project-local venv runner created with `--system-site-packages`, reusing the preinstalled vLLM package from the Conda environment.

## Isolation

```bash
/opt/anaconda3/bin/python -m venv --system-site-packages .venv-vllm-bench
.venv-vllm-bench/bin/python -m pip install vllm==0.24.0 --no-deps
.venv-vllm-bench/bin/python -c "import sys, vllm; print(sys.executable); print(vllm.__version__)"
```

The install command reported `Requirement already satisfied` for `vllm==0.24.0` from `/opt/anaconda3/lib/python3.13/site-packages`. The project dependency file was not modified. This runner avoids adding vLLM to project dependencies, but it is not a fully independent dependency environment. The benchmark command was run with `.venv-vllm-bench/bin/python`.

## Evidence

- Feasibility JSON: [vllm_feasibility_2026-07-12.json](vllm_feasibility_2026-07-12.json)
- Strict smoke JSONL: [smoke_strict_2026-07-12.jsonl](smoke_strict_2026-07-12.jsonl)
- Strict formal JSONL: [offline_comparison_strict_2026-07-12.jsonl](offline_comparison_strict_2026-07-12.jsonl)
- Strict comparison: [offline_comparison_strict_2026-07-12.md](offline_comparison_strict_2026-07-12.md)

## Notes

The environment uses Python 3.13 and RTX 2080 Ti GPUs with compute capability 7.5. The existing installed vLLM package imported successfully and completed both smoke and formal offline generation runs.
