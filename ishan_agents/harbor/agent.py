import json
import uuid
from datetime import datetime, timezone

import aiofiles
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.agent import Agent as ATIFAgent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.metrics import Metrics
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory

from ishan_agents.agent_loop import LoopResult, run_agent_loop
from ishan_agents.harbor.sandbox import HarborSandbox
from ishan_agents.tools.claude_code import claude_code_tools

DEFAULT_MODEL = "claude-sonnet-4-6"
AGENT_VERSION = "0.1.0"
TRAJECTORY_FILENAME = "trajectory.json"

AGENT_NAME = "ishan-agents"


def _build_trajectory(
    model: str,
    tools: list,
    result: LoopResult,
    session_id: str,
) -> Trajectory:
    """Build an ATIF Trajectory from a LoopResult."""
    messages = result.messages
    turn_usages = result.turn_usages

    atif_agent = ATIFAgent(
        name=AGENT_NAME,
        version=AGENT_VERSION,
        model_name=model,
        tool_definitions=[t.to_openai()["function"] for t in tools],
    )

    steps: list[Step] = []
    step_id = 1
    agent_turn_idx = 0

    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            # Tool results are paired with the preceding agent step — skip them here.
            content = msg["content"]
            if isinstance(content, str):
                steps.append(Step(
                    step_id=step_id,
                    source="user",
                    message=content,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                step_id += 1

        elif msg["role"] == "assistant":
            content = msg["content"]

            # Extract text
            text_parts = [b.text for b in content if hasattr(b, "type") and b.type == "text"]
            message_text = "\n".join(text_parts) if text_parts else ""

            # Extract tool calls
            tool_calls = [
                ToolCall(
                    tool_call_id=b.id,
                    function_name=b.name,
                    arguments=dict(b.input),
                )
                for b in content if hasattr(b, "type") and b.type == "tool_use"
            ]

            # Observation: tool results from the immediately following user message.
            # NOTE: assumes sequential tool execution (one tool_result batch per assistant turn).
            # Parallel tool calls within the same turn are all captured in that single following
            # user message, so this still works — but if the loop ever interleaves turns differently
            # this pairing logic will need revisiting.
            observation = None
            if tool_calls and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg["role"] == "user" and isinstance(next_msg["content"], list):
                    obs_results = [
                        ObservationResult(
                            source_call_id=r["tool_use_id"],
                            content=r.get("content", ""),
                        )
                        for r in next_msg["content"]
                        if r.get("type") == "tool_result"
                    ]
                    if obs_results:
                        observation = Observation(results=obs_results)

            # Metrics from this agent turn
            usage = turn_usages[agent_turn_idx] if agent_turn_idx < len(turn_usages) else None
            metrics = None
            if usage:
                metrics = Metrics(
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.output_tokens,
                    cached_tokens=getattr(usage, "cache_read_input_tokens", None),
                )

            steps.append(Step(
                step_id=step_id,
                source="agent",
                message=message_text,
                tool_calls=tool_calls or None,
                observation=observation,
                metrics=metrics,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            step_id += 1
            agent_turn_idx += 1

    total_prompt = sum(u.input_tokens for u in turn_usages)
    total_completion = sum(u.output_tokens for u in turn_usages)
    total_cached = sum(getattr(u, "cache_read_input_tokens", 0) or 0 for u in turn_usages)

    return Trajectory(
        schema_version="ATIF-v1.6",
        session_id=session_id,
        agent=atif_agent,
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
            total_steps=len(steps),
        ),
    )


class HarborAgent(BaseAgent):
    SUPPORTS_ATIF = True

    def __init__(
        self,
        tools_factory=None,
        system_prompt: str | None = None,
        max_turns: int = 50,
        work_dir: str = "/testbed",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._tools_factory = tools_factory or claude_code_tools
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._work_dir = work_dir

    @staticmethod
    def name() -> str:
        return AGENT_NAME

    def version(self) -> str | None:
        return AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        pass  # No CLI to install — we run directly in Python.

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        session_id = str(uuid.uuid4())
        model = self.model_name or DEFAULT_MODEL

        sandbox = HarborSandbox(environment, self._work_dir)
        tools = self._tools_factory(sandbox)

        result = await run_agent_loop(
            model=model,
            sandbox=sandbox,
            tools=tools,
            user_message=instruction,
            system_prompt=self._system_prompt,
            max_turns=self._max_turns,
        )

        # Write ATIF trajectory
        trajectory = _build_trajectory(model, tools, result, session_id)
        traj_path = self.logs_dir / TRAJECTORY_FILENAME
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(traj_path, "w") as f:
            await f.write(json.dumps(trajectory.to_json_dict(), indent=2))

        # Populate Harbor context with token usage
        context.n_input_tokens = sum(u.input_tokens for u in result.turn_usages)
        context.n_output_tokens = sum(u.output_tokens for u in result.turn_usages)
        context.n_cache_tokens = sum(
            getattr(u, "cache_read_input_tokens", 0) or 0 for u in result.turn_usages
        )
