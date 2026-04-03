# TODO

## Up next

- [x] Implement `claude_code` tools (Read, Write, Edit, Glob, Grep; Bash)
- [x] Implement the agent turn loop in `agent.py` (Anthropic client, tool dispatch, max_turns)
- [ ] Harbor integration + run SWE-bench
      - `run_agent_loop(sandbox, tools, config)` — core loop, creates tools, calls API, dispatches execution
      - Harbor integration = `HarborSandbox` + pass to `run_agent_loop` with experiment config
      - Harbor requires subclassing its `Agent` class — `HarborAgent` wraps `run_agent_loop` and satisfies that interface
      - NOTE: Smoke test the setup against a small SWEbench run here
- [ ] Experiment config/handler — meaty eng job, covers:
      - Provider abstraction (wraps client + message format + tool format) to swap Anthropic ↔ OpenAI-compatible (Tinker)
      - `ExperimentConfig` dataclass: model, provider, tools, sandbox, system prompt, max_turns, etc.
      - `run_agent_loop` takes a config instead of individual params
      - Config serialization (JSON/YAML) for reproducibility + logging
- [ ] Integrate with an RL provider (Tinker API or TRL; probably start with Tinker)
- [ ] Actual experiments!!!!

## Backlog

- [ ] Docker and/or a real sandbox (gvisor or firecracker)
- [ ] OpenAI tool namespace


