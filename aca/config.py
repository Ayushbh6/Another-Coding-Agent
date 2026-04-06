from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_SQLITE_URL = "sqlite:///./data/aca.db"
DEFAULT_CHROMA_PATH = "./data/chroma"
DEFAULT_CHROMA_COLLECTION = "aca-memory"


@dataclass(frozen=True, slots=True)
class Settings:
    sqlite_url: str = DEFAULT_SQLITE_URL
    chroma_path: str = DEFAULT_CHROMA_PATH
    chroma_collection: str = DEFAULT_CHROMA_COLLECTION


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        sqlite_url=os.getenv("ACA_SQLITE_URL", DEFAULT_SQLITE_URL),
        chroma_path=os.getenv("ACA_CHROMA_PATH", DEFAULT_CHROMA_PATH),
        chroma_collection=os.getenv("ACA_CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION),
    )


def ensure_parent_dir(path_like: str) -> None:
    if path_like.startswith("sqlite:///"):
        db_path = path_like.removeprefix("sqlite:///")
        Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return

    Path(path_like).expanduser().resolve().mkdir(parents=True, exist_ok=True)

