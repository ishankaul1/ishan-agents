import os

from dotenv import load_dotenv

load_dotenv()
# Core agent runner
# V0 Impl will just keep running Anthropic responses client in a loop until no tools generated/


# Observability/eval will be run in a separate script; likely just harbor


def _base_agent_system_prompt(work_dir: str) -> str:
    return (
        f"You are a coding agent. You must try your best to solve the task at hand autonomously; "
        f"the user will never respond to you. All file paths and bash commands run from {work_dir}. "
        f"Use relative paths from there or absolute paths."
    )


def run_agent_loop(self, work_dir: str, user_message: str, model_name: int, max_turns: int):
    assert os.path.isdir(work_dir), f"work_dir does not exist: {work_dir}"
    assert max_turns >= 0, f"max_turns <= 0: {max_turns}"

    # system_prompt = _base_agent_system_prompt(work_dir=work_dir)

    # TODO(ishankaul1), 03/23/2026): Finish Impl
