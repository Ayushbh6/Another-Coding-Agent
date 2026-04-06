from __future__ import annotations

from dataclasses import dataclass

import chromadb
from sqlalchemy import Engine, event
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session, sessionmaker

from aca.config import Settings, ensure_parent_dir, get_settings

from .models import Agent, Base


@dataclass(slots=True)
class StorageBootstrap:
    engine: Engine
    session_factory: sessionmaker[Session]
    chroma_client: chromadb.PersistentClient
    chroma_collection: chromadb.Collection


def initialize_storage(settings: Settings | None = None) -> StorageBootstrap:
    resolved = settings or get_settings()
    ensure_parent_dir(resolved.sqlite_url)
    ensure_parent_dir(resolved.chroma_path)

    engine = create_engine(resolved.sqlite_url, future=True)
    _enable_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    with session_factory.begin() as session:
        _seed_bootstrap_agents(session)

    chroma_client = chromadb.PersistentClient(path=resolved.chroma_path)
    chroma_collection = chroma_client.get_or_create_collection(name=resolved.chroma_collection)

    return StorageBootstrap(
        engine=engine,
        session_factory=session_factory,
        chroma_client=chroma_client,
        chroma_collection=chroma_collection,
    )


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _seed_bootstrap_agents(session: Session) -> None:
    seeds = [
        {
            "id": "master",
            "name": "Master",
            "role": "master",
            "owns_json": [],
            "reports_to_agent_id": None,
        },
        {
            "id": "challenger",
            "name": "Challenger",
            "role": "challenger",
            "owns_json": [],
            "reports_to_agent_id": "master",
        },
        {
            "id": "codebase_mapper",
            "name": "CodebaseMapper",
            "role": "mapper",
            "owns_json": [],
            "reports_to_agent_id": "master",
        },
    ]

    for seed in seeds:
        if session.get(Agent, seed["id"]) is not None:
            continue
        session.add(Agent(**seed, status="active", metadata_json={}))

