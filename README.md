# ishan-agents

Coding agent eval & experimentation repo.

## Dev

```bash
uv run ruff check .   # lint
uv run ruff format .  # format
uv run pytest         # test
```

## Running Experiments


```
uv run harbor run -c experiments/terminal_bench_smoke.yaml --env-file .env
```

```
uv run harbor run -c
  experiments/swebench_fireworks_qwen.yaml --env-file .env
```