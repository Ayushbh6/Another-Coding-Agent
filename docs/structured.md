# Repo Structure

## Current Assessment

The repository is now **partially well-structured**:

- the real application code is organized under `aca/`
- persistence, services, and provider code are split cleanly
- automated tests already live under `tests/`

What was still messy before this cleanup was the repo root:

- demo files were mixed with application code
- scratch experiment files were mixed with real entrypoints
- there was no single document explaining the intended layout

This file defines the cleaner structure going forward.

## Intended Layout

```text
Another(ACA)/
├── aca/
│   ├── llm/
│   │   ├── providers/
│   │   ├── history.py
│   │   ├── openrouter_client.py
│   │   └── types.py
│   ├── services/
│   │   └── conversation.py
│   ├── storage/
│   │   ├── bootstrap.py
│   │   └── models.py
│   ├── config.py
│   ├── core_types.py
│   ├── master.py
│   ├── runtime.py
│   └── token_utils.py
├── docs/
│   └── structured.md
├── examples/
│   ├── demo_conversation_service.py
│   └── demo_runtime.py
├── experiments/
│   ├── test_agent.py
│   └── test_openai.py
├── tests/
│   └── test_conversation_service.py
├── data/
│   ├── aca.db
│   └── chroma/
├── MEMORY.md
├── idea.md
├── still-missing.md
└── requirements.txt
```

## Folder Responsibilities

### `aca/`

Main application package. Real product code should go here.

Substructure:

- `aca/llm/`: provider abstractions, history handling, request/response normalization
- `aca/llm/providers/`: concrete model-provider implementations
- `aca/services/`: orchestration layer that coordinates runtime + persistence
- `aca/storage/`: schema, DB bootstrap, persistence primitives

Rule:

- all reusable application logic belongs here

### `docs/`

Repository documentation that explains structure, conventions, or architecture.

Rule:

- new organizational or implementation notes should prefer `docs/` unless they are intended to remain as top-level canonical files

### `examples/`

Runnable example entrypoints and smoke-test scripts.

Rule:

- keep examples here
- do not mix them with package code

### `experiments/`

Scratch files, prototypes, provider tests, or one-off exploratory scripts.

Rule:

- code here is useful, but not part of the main runtime package
- these files can later be deleted, rewritten, or promoted into `aca/` if they become real features

### `tests/`

Automated tests for the actual implementation.

Rule:

- keep unit and integration tests here

### `data/`

Local runtime state:

- SQLite database
- Chroma local storage

Rule:

- this is runtime-generated state, not application source

## Top-Level Files

These currently stay at the root because they are already being used as canonical project notes:

- `MEMORY.md`
- `idea.md`
- `still-missing.md`
- `requirements.txt`

If the repo grows, these can later move into `docs/`, but for now keeping them stable avoids breaking the current workflow and references.

## Practical Rules Going Forward

1. New product code goes into `aca/`.
2. New demos go into `examples/`.
3. New scratch experiments go into `experiments/`.
4. New automated tests go into `tests/`.
5. New structure or architecture notes go into `docs/`.
6. The repo root should stay minimal and mostly contain only canonical project files.
