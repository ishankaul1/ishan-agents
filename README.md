# ishan-agents

Building a minimal coding agent harness to experiment across tool call strats, models, and build my own RL suite.

## Phase 1 Goals

- Minimal agent loop with Anthropic api and configurable model name and max_turns. Most likely will just return the entire trace
- Basic evaluation & observability suite
    - JSONify the entire trace and persist to local disk at a path akin to a session/turn ID
    - Should be able to run against swe bench lite


    