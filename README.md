# ishan-agents

Coding agent eval & experimentation repo.
Harbor-based runner/orchestration and trajectory logging; I use their yaml format (see experiments/).

Example experiment run -
```
uv run harbor run -c
  experiments/swebench_fireworks_qwen.yaml --env-file .env
```


Harness is designed to be fully interoperable (Anthropic & OpenAI compat clients (OpenAI, Tinker, Fireworks) already supported), and different combos of tools supported through tools param.

Rollout management for RL & context management not impl'ed yet but I plan on it!!

## Dev

```bash
uv run ruff check .   # lint
uv run ruff format .  # format
uv run pytest         # test
uv run harbor run -c experiments/terminal_bench_smoke.yaml --env-file .env # Run experiment
```
