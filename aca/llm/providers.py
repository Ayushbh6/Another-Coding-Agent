"""
Provider configuration for ACA's LLM layer.

Each Provider instance carries everything needed to initialise an OpenAI-compatible
client:  base URL, the env var that holds the API key, and a display name.
Both OpenRouter and OpenAI are accessed through the OpenAI SDK pointed at the
appropriate base_url.
"""

import os
from dataclasses import dataclass
from enum import Enum


class ProviderName(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"


@dataclass(frozen=True)
class Provider:
    name: ProviderName
    base_url: str
    api_key_env_var: str
    display_name: str

    def api_key(self) -> str:
        """Resolve the API key from the environment at call time."""
        key = os.environ.get(self.api_key_env_var)
        if not key:
            raise EnvironmentError(
                f"Missing API key for provider '{self.name}'. "
                f"Set the '{self.api_key_env_var}' environment variable."
            )
        return key


_REGISTRY: dict[ProviderName, Provider] = {
    ProviderName.OPENROUTER: Provider(
        name=ProviderName.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        api_key_env_var="OPENROUTER_API_KEY",
        display_name="OpenRouter",
    ),
    ProviderName.OPENAI: Provider(
        name=ProviderName.OPENAI,
        base_url="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
        display_name="OpenAI",
    ),
}


def get_provider(name: str | ProviderName = ProviderName.OPENROUTER) -> Provider:
    """Return the Provider config for the given name.

    Accepts either a ProviderName enum value or its string equivalent.
    Defaults to OpenRouter.
    """
    key = ProviderName(name) if isinstance(name, str) else name
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Valid options: {[p.value for p in ProviderName]}"
        )
    return _REGISTRY[key]
