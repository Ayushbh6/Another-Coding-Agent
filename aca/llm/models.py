"""
Model name constants for ACA.

OpenRouter model strings follow the format  <provider>/<model-name>.
Add new models here — never hardcode strings elsewhere in the codebase.
"""

# ── OpenRouter ────────────────────────────────────────────────────────────────


class OpenRouterModels:
    GLM_5_1 = "z-ai/glm-5.1:nitro"
    GLM_5V = "z-ai/glm-5v-turbo"
    KIMI_K_2_5 = "moonshotai/kimi-k2.5:nitro"
    minimax_2_7 = "minimax/minimax-m2.7:nitro"
    gemma_4 = "google/gemma-4-31b-it:nitro"


# ── Native OpenAI (provider == openai) ───────────────────────────────────────


class OpenAIModels:
    GPT_5 = "gpt-5-2025-08-07"
    GPT_5_4_MINI = "gpt-5.4-mini"


# ── Project default ───────────────────────────────────────────────────────────

DEFAULT_MODEL: str = OpenRouterModels.minimax_2_7
