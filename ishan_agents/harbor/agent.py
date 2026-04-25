import json
import os
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
from harbor.models.trajectories.tool_call import ToolCall as ATIFToolCall
from harbor.models.trajectories.trajectory import Trajectory
from loguru import logger

from ishan_agents.agent_loop import LoopResult, default_system_prompt, run_agent_loop
from ishan_agents.harbor.sandbox import HarborSandbox
from ishan_agents.llms.factory import make_client
from ishan_agents.log import add_file_sink, configure_stdout_logging

DEFAULT_MODEL = "claude-sonnet-4-6"
AGENT_VERSION = "0.1.0"
TRAJECTORY_FILENAME = "trajectory.json"
AGENT_NAME = "ishan-agents"
DEFAULT_TOOLS = ["claude_code"]


def _build_trajectory(model: str, tools: list, result: LoopResult, session_id: str) -> Trajectory:
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

    for i, msg in enumerate(messages):
        if msg.role == "user" and msg.tool_results is None:
            steps.append(Step(
                step_id=step_id,
                source="user",
                message=msg.content or "",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            step_id += 1

        elif msg.role == "assistant":
            atif_tool_calls = [
                ATIFToolCall(tool_call_id=tc.id, function_name=tc.name, arguments=tc.input)
                for tc in (msg.tool_calls or [])
            ]

            observation = None
            if atif_tool_calls and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.tool_results:
                    obs_results = [
                        ObservationResult(source_call_id=r.tool_call_id, content=r.content)
                        for r in next_msg.tool_results
                    ]
                    if obs_results:
                        observation = Observation(results=obs_results)

            metrics = None
            if msg.usage:
                metrics = Metrics(
                    prompt_tokens=msg.usage.input_tokens,
                    completion_tokens=msg.usage.output_tokens,
                    cached_tokens=msg.usage.cache_read_tokens or None,
                )

            steps.append(Step(
                step_id=step_id,
                source="agent",
                message=msg.content or "",
                tool_calls=atif_tool_calls or None,
                observation=observation,
                metrics=metrics,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            step_id += 1

    total_prompt = sum(u.input_tokens for u in turn_usages)
    total_completion = sum(u.output_tokens for u in turn_usages)
    total_cached = sum(u.cache_read_tokens for u in turn_usages)

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
        provider: str = "anthropic",
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        max_turns: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._provider = provider
        self._tool_specs = tools or DEFAULT_TOOLS
        self._system_prompt = system_prompt
        self._max_turns = max_turns

    @staticmethod
    def name() -> str:
        return AGENT_NAME

    def version(self) -> str | None:
        return AGENT_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        session_id = str(uuid.uuid4())
        model = self.model_name or DEFAULT_MODEL

        configure_stdout_logging()

        log_file = self.logs_dir / "agent.log"
        file_sink_id = None
        if os.getenv("AGENT_LOG"):
            file_sink_id = add_file_sink(log_file)
            logger.info(f"Logging to: {log_file}")

        try:
            pwd = await environment.exec("pwd")
            work_dir = (pwd.stdout or "").strip() or "/"

            sandbox = HarborSandbox(environment, work_dir)
            client = make_client(self._provider, model)

            result = await run_agent_loop(
                client=client,
                sandbox=sandbox,
                tools=self._tool_specs,
                user_message=instruction,
                system_prompt=self._system_prompt or default_system_prompt(sandbox.work_dir),
                max_turns=self._max_turns,
            )

            trajectory = _build_trajectory(model, result.tools, result, session_id)
            traj_path = self.logs_dir / TRAJECTORY_FILENAME
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(traj_path, "w") as f:
                await f.write(json.dumps(trajectory.to_json_dict(), indent=2))

            context.n_input_tokens = sum(u.input_tokens for u in result.turn_usages)
            context.n_output_tokens = sum(u.output_tokens for u in result.turn_usages)
            context.n_cache_tokens = sum(u.cache_read_tokens for u in result.turn_usages)
        finally:
            if file_sink_id is not None:
                logger.remove(file_sink_id)
