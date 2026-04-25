import os
from abc import abstractmethod

from pydantic import BaseModel

from ishan_agents.llms.anthropic_client import AnthropicClient
from ishan_agents.llms.base import LLMClient
from ishan_agents.llms.openai_client import OpenAICompatClient


class Provider(BaseModel, frozen=True):
    api_key_env: str

    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if key is None:
            raise ValueError(f"Missing env var '{self.api_key_env}'")
        return key

    @abstractmethod
    def make_client(self, model: str) -> LLMClient: ...


class AnthropicProvider(Provider):
    def make_client(self, model: str) -> LLMClient:
        return AnthropicClient(model)


class OpenAICompatProvider(Provider):
    base_url: str | None = None

    def make_client(self, model: str) -> LLMClient:
        return OpenAICompatClient(model, base_url=self.base_url, api_key=self.api_key())


_PROVIDERS: dict[str, Provider] = {
    "anthropic": AnthropicProvider(api_key_env="ANTHROPIC_API_KEY"),
    "openai": OpenAICompatProvider(api_key_env="OPENAI_API_KEY"),
    "fireworks": OpenAICompatProvider(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
    ),
    "tinker": OpenAICompatProvider(
        base_url="https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
        api_key_env="TINKER_API_KEY",
    ),
    "together": OpenAICompatProvider(
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    ),
}


def make_client(provider: str, model: str) -> LLMClient:
    p = _PROVIDERS.get(provider)
    if p is None:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list(_PROVIDERS)}")
    return p.make_client(model)
