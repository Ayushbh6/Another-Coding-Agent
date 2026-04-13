"""
ChallengerAgent — bounded plan critic for ACA.

Challenger responsibilities:
  - Receive James's current plan (task.md + plan.md + todo.md)
  - Challenge it: find flaws, blind spots, missed edge cases, scope creep
  - Suggest simpler or safer alternatives
  - Return critique — stop

Challenger differences from James and Worker:
  - READ permission mode only: cannot write files, only read and reason
  - Very tight tool_call_limit (8): bounded critic, not a second planner
  - No routing steering (routing_budget=0)
  - Single-purpose: one call to run_turn(), receives plan context, returns critique

Usage:
    challenger = ChallengerAgent(registry=registry, ...)
    critique, _ = challenger.run_turn(
        user_message="Critique this plan.",
        extra_context=[
            {"role": "user", "content": task_md_content},
            {"role": "user", "content": plan_md_content},
        ]
    )
    # James reads `critique` and updates plan.md if needed

Tool budget:
  - tool_call_limit = 8   (tight — read a few files, then give critique)
  - routing_budget  = 0   (no routing phase)
"""

from __future__ import annotations

from typing import Any

from aca.agents.base_agent import BaseAgent
from aca.llm.models import DEFAULT_MODEL
from aca.llm.providers import ProviderName
from aca.tools.registry import PermissionMode, ToolRegistry


_CHALLENGER_SYSTEM_PROMPT = """\
You are the Challenger, a bounded plan critic for ACA.

## Your role
James has drafted a plan. Your job is to challenge it before execution begins.

## What you must do
1. Read the task description and plan provided in context.
2. Identify as many of these as you can find:
   - Logical flaws or incorrect assumptions
   - Missing edge cases or failure modes
   - Unnecessary complexity (suggest simpler alternatives)
   - Scope creep (steps that go beyond what the user asked)
   - Missing safety steps (no verification, no rollback plan)
   - Ambiguous steps that could be interpreted multiple ways
3. Produce a structured critique.

## Output format
Your output must be structured markdown:

### Flaws
- <list any logical flaws>

### Missing edge cases
- <list any edge cases not covered>

### Simplifications
- <suggest simpler or safer approaches if applicable>

### Scope concerns
- <note any out-of-scope steps>

### Overall verdict
APPROVE — plan is solid, proceed.
REVISE  — specific changes needed before proceeding (list them).

## Constraints
- You are a critic, not a second planner. Do not rewrite the plan.
- Be specific. Vague concerns are not useful.
- If the plan is good, say so clearly (APPROVE).
- Keep your critique focused and actionable.
"""


class ChallengerAgent(BaseAgent):
    """
    Bounded plan critic.

    READ-only permission mode. Tight tool call limit.
    Invoked by James before delegating a large task.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_mode: PermissionMode = PermissionMode.READ,
        tool_call_limit: int = 30,
        provider: ProviderName = ProviderName.OPENROUTER,
        model: str = DEFAULT_MODEL,
        thinking: bool = False,
        stream: bool = True,
        db: Any = None,
        session_id: str | None = None,
        console: Any = None,
        context_token_threshold: int = 120_000,
    ) -> None:
        super().__init__(
            registry=registry,
            permission_mode=permission_mode,
            tool_call_limit=tool_call_limit,
            provider=provider,
            model=model,
            thinking=thinking,
            stream=stream,
            db=db,
            session_id=session_id,
            console=console,
            context_token_threshold=context_token_threshold,
            routing_budget=0,   # no routing phase
        )

    def agent_name(self) -> str:
        return "challenger"

    def system_prompt(self) -> str:
        return _CHALLENGER_SYSTEM_PROMPT
