from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_SQLITE_URL = "sqlite:///./data/aca.db"
DEFAULT_CHROMA_PATH = "./data/chroma"
DEFAULT_CHROMA_COLLECTION = "aca-memory"

OPENROUTER_MODELS: dict[str, str] = {
    "kimi_k2_5": "moonshotai/kimi-k2.5:nitro",
    "minimax_m2_7": "minimax/minimax-m2.7:nitro",
    "google_gem_4": "google/gemma-4-31b-it:nitro",
    "glm_5v": "z-ai/glm-5v-turbo",
    "glm_5": "z-ai/glm-5:nitro",
}

OPENROUTER_MODEL_ORDER: list[str] = [
    "kimi_k2_5",
    "minimax_m2_7",
    "google_gem_4",
    "glm_5v",
    "glm_5",
]

DEFAULT_OPENROUTER_MODEL_KEY = "kimi_k2_5"


@dataclass(frozen=True, slots=True)
class Settings:
    sqlite_url: str = DEFAULT_SQLITE_URL
    chroma_path: str = DEFAULT_CHROMA_PATH
    chroma_collection: str = DEFAULT_CHROMA_COLLECTION
    default_openrouter_model: str = OPENROUTER_MODELS[DEFAULT_OPENROUTER_MODEL_KEY]


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        sqlite_url=os.getenv("ACA_SQLITE_URL", DEFAULT_SQLITE_URL),
        chroma_path=os.getenv("ACA_CHROMA_PATH", DEFAULT_CHROMA_PATH),
        chroma_collection=os.getenv("ACA_CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION),
        default_openrouter_model=resolve_openrouter_model(
            os.getenv("ACA_OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL_KEY)
        ),
    )


def ensure_parent_dir(path_like: str) -> None:
    if path_like.startswith("sqlite:///"):
        db_path = path_like.removeprefix("sqlite:///")
        Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return

    Path(path_like).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def resolve_openrouter_model(model_key_or_id: str) -> str:
    return OPENROUTER_MODELS.get(model_key_or_id, model_key_or_id)


def get_allowed_openrouter_models() -> list[tuple[str, str]]:
    return [(key, OPENROUTER_MODELS[key]) for key in OPENROUTER_MODEL_ORDER]
