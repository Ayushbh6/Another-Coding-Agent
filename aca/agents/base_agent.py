"""
BaseAgent — shared agentic loop for all ACA agents.

Steering injection strings live in aca.agents.steering; this module wires them
into the loop and calls hooks for agent-specific extensions (James).
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from aca.agents.steering import SteeringJunction, junction_compact_context, junction_force_reply, junction_route
from aca.llm.client import LLMResponse, call_llm
from aca.llm.models import DEFAULT_MODEL
from aca.llm.providers import ProviderName
from aca.tools.registry import PermissionMode, ToolRegistry, ToolResult


class BaseAgent(ABC):
    """
    Abstract base class for all ACA agents.

    Subclasses may override:
      - _extra_pre_llm_steering(...) — additional junctions (James)
      - _on_tool_completed(...) — track tool side-effects (James artifacts)
      - _reset_agent_turn_state() — clear per-turn flags (James)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_mode: PermissionMode = PermissionMode.READ,
        tool_call_limit: int = 20,
        provider: ProviderName = ProviderName.OPENROUTER,
        model: str = DEFAULT_MODEL,
        thinking: bool = False,
        stream: bool = True,
        db: Any = None,
        session_id: str | None = None,
        console: Any = None,
        context_token_threshold: int = 80_000,
        routing_budget: int = 0,
    ) -> None:
        self._registry = registry
        self._permission_mode = permission_mode
        self._tool_call_limit = tool_call_limit
        self._provider = provider
        self._model = model
        self._thinking = thinking
        self._stream = stream
        self._db = db
        self._session_id = session_id
        self._console = console
        self._context_token_threshold = context_token_threshold
        self._routing_budget = routing_budget

        self._history: list[dict] = []
        # Parallel to self._history: one entry per Q&A pair recording turn_id.
        # len(_history_meta) == len(_history) // 2 at all times.
        self._history_meta: list[dict] = []
        self._turn_index: int = 0
        self._steering_fired: set[str] = set()
        # After routing / task creation, tool dispatch uses this (James: EDIT; orient uses READ)
        self._full_permission_mode: PermissionMode = permission_mode

    @abstractmethod
    def agent_name(self) -> str:
        """Return 'james' | 'worker' | 'challenger'."""

    @abstractmethod
    def system_prompt(self) -> str:
        """Full system prompt."""

    # ── Hooks (override in JamesAgent) ─────────────────────────────────────────

    def _reset_agent_turn_state(self) -> None:
        """Clear per-turn steering extensions. Base: no-op."""
        self._steering_fired.clear()

    def _on_tool_completed(
        self,
        tool_name: str,
        raw_args: str,
        result: ToolResult,
        *,
        routed: bool,
    ) -> None:
        """Track tool outcomes for steering. Base: no-op."""
        del tool_name, raw_args, result, routed

    def _extra_pre_llm_steering(
        self,
        messages: list[dict],
        tools: list | None,
        tool_choice: str | None,
        current_mode: PermissionMode,
        *,
        routing_tools_used: int,
        routed: bool,
        tool_calls_this_turn: int,
    ) -> tuple[list | None, str | None, PermissionMode]:
        """
        Apply agent-specific steering after base ROUTE / FORCE_REPLY handling.

        Returns (tools, tool_choice, current_mode) — may replace tools list.
        """
        del messages, routing_tools_used, routed, tool_calls_this_turn
        return tools, tool_choice, current_mode

    # ── Public API ────────────────────────────────────────────────────────────

    def run_turn(
        self,
        user_message: str,
        extra_context: list[dict] | None = None,
        stream_callback: Any = None,
        *,
        _continuation: bool = False,
    ) -> tuple[str, list[dict]]:
        if not _continuation:
            self._reset_agent_turn_state()

        turn_id = self._log_turn_start(user_message)
        turn_start_ms = int(time.time() * 1000)

        tool_calls_this_turn = 0
        routing_tools_used = 0
        routed = False
        cumulative_input_tokens = 0
        cumulative_output_tokens = 0
        call_index = 0

        messages = self._build_messages(user_message, extra_context)
        # Marks where current-turn content starts being appended.
        # Used to reconstruct messages after compaction without losing work.
        current_turn_start_idx = len(messages)
        in_compaction_phase = False
        final_response = ""
        current_mode = self._permission_mode

        if self._console and not _continuation:
            self._console.step(f"{self.agent_name().upper()} turn #{self._turn_index + 1}")

        while True:
            # ── Choose tools ──────────────────────────────────────────────────
            if in_compaction_phase:
                from aca.tools.memory import get_compact_context_agent_schema
                tools: list | None = [get_compact_context_agent_schema()]
                tool_choice: str | None = "required"
            else:
                tools = self._registry.get_schemas(current_mode)
                tools, tool_choice, current_mode = self._apply_pre_llm_steering(
                    messages,
                    tools,
                    current_mode,
                    routing_tools_used=routing_tools_used,
                    routed=routed,
                    tool_calls_this_turn=tool_calls_this_turn,
                )

            if self._console:
                n_tools = len(tools) if tools else 0
                self._console.llm_call(self._model, call_index, n_tools)

            resp: LLMResponse = call_llm(
                messages=messages,
                system_prompt=self.system_prompt(),
                model=self._model,
                provider=self._provider,
                tools=tools if tools and tool_choice != "none" else None,
                tool_choice=tool_choice,
                thinking=self._thinking,
                stream=self._stream,
                stream_callback=stream_callback or (
                    self._console.make_stream_callback(thinking=self._thinking) if self._console else None
                ),
                db=self._db,
                turn_id=turn_id,
                session_id=self._session_id,
                agent=self.agent_name(),
                call_index=call_index,
            )

            call_index += 1
            cumulative_input_tokens += resp.input_tokens
            cumulative_output_tokens += resp.output_tokens

            # ── Threshold check: enter compaction phase ───────────────────────
            if (
                not in_compaction_phase
                and resp.input_tokens >= self._context_token_threshold
                and self._history_meta  # need at least one past pair to evict
            ):
                in_compaction_phase = True
                inventory = self._build_compaction_inventory()
                j = junction_compact_context(
                    pairs_inventory=inventory,
                    current_tokens=resp.input_tokens,
                    threshold=self._context_token_threshold,
                    current_turn_id=turn_id,
                )
                self._inject_steering(messages, j)

            # ── Stop check ─────────────────────────────────────────────────────
            if resp.stop_reason in ("end_turn", "stop") or not resp.tool_calls:
                if in_compaction_phase:
                    # Agent produced text instead of calling compact_context.
                    # Preserve the text in messages so it has full context, then
                    # loop back — tool_choice="required" will force the call.
                    if resp.content:
                        messages.append({"role": "assistant", "content": resp.content})
                    continue
                final_response = resp.content or ""
                # content_panel is a debug display for non-CLI contexts only.
                # In streaming mode the live tokens are the display; in non-streaming
                # (CLI) mode _render_response handles the single final render.
                if self._console and final_response and not self._stream and getattr(self._console, "_show_content_panel", True):
                    self._console.content_panel(
                        f"{self.agent_name().capitalize()} response", final_response
                    )
                break

            assistant_msg: dict = {"role": "assistant", "content": resp.content or ""}
            if resp.tool_calls:
                assistant_msg["tool_calls"] = resp.tool_calls
            if resp.thinking_blocks:
                assistant_msg["reasoning_details"] = resp.thinking_blocks
            messages.append(assistant_msg)

            for tc in resp.tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                raw_args = tc.get("function", {}).get("arguments", "{}")
                tool_call_id = tc.get("id", str(uuid.uuid4()))

                if self._console:
                    try:
                        args_preview = json.loads(raw_args)
                    except Exception:
                        args_preview = raw_args
                    self._console.tool_call(tool_name, args_preview)

                t_start = int(time.time() * 1000)

                if tool_name == "compact_context":
                    # ── Intercept: handle compaction directly ─────────────────
                    result = self._execute_compact_context(
                        tc=tc,
                        current_turn_id=turn_id,
                        current_tokens=resp.input_tokens,
                    )
                    t_end = int(time.time() * 1000)

                    if result.success:
                        in_compaction_phase = False
                        # Rebuild messages from trimmed history + current-turn content
                        new_msgs = self._build_messages(user_message, extra_context)
                        new_msgs.extend(messages[current_turn_start_idx:])
                        messages = new_msgs
                else:
                    result = self._registry.dispatch(
                        tool_call=tc,
                        mode=current_mode,
                        db=self._db,
                        turn_id=turn_id,
                        session_id=self._session_id,
                        agent=self.agent_name(),
                        llm_call_id=None,
                    )
                    t_end = int(time.time() * 1000)

                if self._console:
                    self._console.tool_result(
                        name=tool_name,
                        success=result.success,
                        latency_ms=t_end - t_start,
                        output=result.output if result.success else None,
                        error=result.error if not result.success else None,
                    )

                messages.append(self._registry.to_tool_message(result))

                tool_calls_this_turn += 1

                if tool_name != "compact_context":
                    self._on_tool_completed(tool_name, raw_args, result, routed=routed)

                    if self._routing_budget > 0 and not routed:
                        if tool_name != "create_task_workspace":
                            # Only count actual orient reads, not the routing action itself
                            routing_tools_used += 1
                        if tool_name == "create_task_workspace" and result.success:
                            routed = True
                            current_mode = self._full_permission_mode

            if tool_choice == "none":
                final_response = resp.content or ""
                break

        self._turn_index += 1
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": final_response})
        self._history_meta.append({"turn_id": turn_id})

        total_latency = int(time.time() * 1000) - turn_start_ms
        self._log_turn_end(
            turn_id=turn_id,
            final_response=final_response,
            input_tokens=cumulative_input_tokens,
            output_tokens=cumulative_output_tokens,
            latency_ms=total_latency,
        )

        return final_response, list(self._history)

    def reset_history(self) -> None:
        self._history = []
        self._history_meta = []
        self._turn_index = 0

    def _apply_pre_llm_steering(
        self,
        messages: list[dict],
        tools: list | None,
        current_mode: PermissionMode,
        *,
        routing_tools_used: int,
        routed: bool,
        tool_calls_this_turn: int,
    ) -> tuple[list | None, str | None, PermissionMode]:
        tool_choice: str | None = None

        if (
            tool_calls_this_turn >= self._tool_call_limit
            and "force_reply" not in self._steering_fired
        ):
            j = junction_force_reply(tool_calls_this_turn)
            self._inject_steering(messages, j)
            self._steering_fired.add("force_reply")
            tool_choice = j.tool_choice
            if self._console:
                self._console.tool_limit_warning(self._tool_call_limit)
            return None, tool_choice, current_mode

        if (
            self._routing_budget > 0
            and not routed
            and routing_tools_used >= self._routing_budget
            and "route" not in self._steering_fired
        ):
            j = junction_route(routing_tools_used)
            self._inject_steering(messages, j)  # also calls console.steering_junction()
            self._steering_fired.add("route")
            tools = self._registry.get_schemas(PermissionMode.EDIT)
            current_mode = PermissionMode.EDIT
            return tools, tool_choice, current_mode

        tools, tool_choice, current_mode = self._extra_pre_llm_steering(
            messages,
            tools,
            tool_choice,
            current_mode,
            routing_tools_used=routing_tools_used,
            routed=routed,
            tool_calls_this_turn=tool_calls_this_turn,
        )
        return tools, tool_choice, current_mode

    def _inject_steering(self, messages: list[dict], junction: SteeringJunction) -> None:
        messages.append({"role": "user", "content": junction.agent_msg})
        if self._console:
            self._console.steering_junction(junction.key, junction.user_terminal_msg)

    def _build_messages(
        self,
        user_message: str,
        extra_context: list[dict] | None,
    ) -> list[dict]:
        msgs: list[dict] = []
        if extra_context:
            msgs.extend(extra_context)
        msgs.extend(self._history)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    # ── Agent-driven compaction helpers ───────────────────────────────────────

    def _build_compaction_inventory(self) -> str:
        """
        Build a human-readable inventory of all Category B Q&A pairs currently
        in self._history, formatted for injection into the compaction steering
        message.  Each entry shows the pair index, turn_id, and brief previews.
        """
        if not self._history_meta:
            return "  (no past Q&A pairs available)"
        lines: list[str] = []
        for i, meta in enumerate(self._history_meta):
            user_raw = self._history[i * 2].get("content", "")
            asst_raw = self._history[i * 2 + 1].get("content", "")
            # Handle content that may be a list (multimodal)
            user_preview = (
                str(user_raw)[:120] if not isinstance(user_raw, list)
                else str(user_raw[0])[:120]
            )
            asst_preview = (
                str(asst_raw)[:100] if not isinstance(asst_raw, list)
                else str(asst_raw[0])[:100]
            )
            lines.append(
                f"  [{i}] turn_id={meta['turn_id']!r}\n"
                f"      User:      {user_preview!r}\n"
                f"      Assistant: {asst_preview!r}"
            )
        return "\n".join(lines)

    def _execute_compact_context(
        self,
        tc: dict,
        current_turn_id: str,
        current_tokens: int,
    ) -> ToolResult:
        """
        Intercept the agent's compact_context call.

        1. Parses compacted_turn_ids from the tool-call arguments.
        2. Validates that the current turn is not being evicted.
        3. Removes the selected pairs from self._history and self._history_meta.
        4. Logs the compaction event to the DB via compact_context().
        5. Returns a synthetic ToolResult so the conversation can continue.
        """
        from aca.tools.memory import compact_context as _compact_context_fn

        tool_call_id = tc.get("id", str(uuid.uuid4()))
        started_at = int(time.time() * 1000)

        try:
            args: dict = json.loads(tc.get("function", {}).get("arguments", "{}") or "{}")
        except json.JSONDecodeError:
            args = {}

        compacted_turn_ids: list[str] = args.get("compacted_turn_ids", [])
        if not isinstance(compacted_turn_ids, list):
            compacted_turn_ids = []

        # Safety: never allow the current turn to be evicted, and validate that
        # every requested id actually exists in history (ignore hallucinated ids)
        valid_ids = {m["turn_id"] for m in self._history_meta}
        compacted_turn_ids = [
            t for t in compacted_turn_ids
            if t != current_turn_id and t in valid_ids
        ]

        # Apply removal — build keep lists in one pass
        remove_set = set(compacted_turn_ids)
        pairs_to_keep: list[dict] = []
        meta_to_keep: list[dict] = []
        actually_evicted_ids: list[str] = []
        for i, meta in enumerate(self._history_meta):
            if meta["turn_id"] in remove_set:
                actually_evicted_ids.append(meta["turn_id"])
            else:
                pairs_to_keep.append(self._history[i * 2])
                pairs_to_keep.append(self._history[i * 2 + 1])
                meta_to_keep.append(meta)

        pairs_removed = len(actually_evicted_ids)
        self._history = pairs_to_keep
        self._history_meta = meta_to_keep

        # Estimate tokens after (proportional to pairs remaining)
        total_original = pairs_removed + len(meta_to_keep)
        tokens_after_est = (
            int(current_tokens * len(meta_to_keep) / total_original)
            if total_original > 0
            else current_tokens
        )
        tokens_freed = max(0, current_tokens - tokens_after_est)
        target_tokens = int(self._context_token_threshold * 0.40)

        # DB log (non-fatal)
        try:
            _compact_context_fn(
                session_id=self._session_id or "",
                turn_id=current_turn_id,
                agent=self.agent_name(),
                context_tokens_before=current_tokens,
                compacted_turn_ids=actually_evicted_ids,
                context_tokens_after=tokens_after_est,
                db=self._db,
            )
        except Exception:  # noqa: BLE001
            pass

        if self._console and pairs_removed > 0:
            self._console.info(
                f"Compaction complete: evicted {pairs_removed} Q&A pair(s), "
                f"~{tokens_freed:,} tokens freed"
            )

        output = {
            "evicted_turn_ids": actually_evicted_ids,
            "pairs_removed": pairs_removed,
            "tokens_before": current_tokens,
            "tokens_freed_est": tokens_freed,
            "tokens_after_est": tokens_after_est,
            "target_tokens": target_tokens,
            "needs_more_compaction": tokens_after_est > target_tokens,
        }
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="compact_context",
            output=output,
            output_json=json.dumps(output),
            success=True,
            error=None,
            latency_ms=int(time.time() * 1000) - started_at,
            started_at=started_at,
        )

    def _log_turn_start(self, user_message: str) -> str:
        turn_id = str(uuid.uuid4())
        if self._db is None:
            return turn_id
        try:
            self._db.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, turn_index, user_message,
                    started_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    self._session_id or "",
                    self._turn_index,
                    user_message,
                    int(time.time() * 1000),
                ),
            )
            self._db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][base_agent] Failed to log turn start: {exc}")
        return turn_id

    def _log_turn_end(
        self,
        turn_id: str,
        final_response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
    ) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                """
                UPDATE turns SET
                    final_response      = ?,
                    ended_at            = ?,
                    total_input_tokens  = ?,
                    total_output_tokens = ?,
                    total_latency_ms    = ?
                WHERE turn_id = ?
                """,
                (
                    final_response,
                    int(time.time() * 1000),
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    turn_id,
                ),
            )
            self._db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][base_agent] Failed to log turn end: {exc}")

    @staticmethod
    def init_session(
        db: Any,
        repo_path: str,
        model: str,
        permission_mode: PermissionMode,
    ) -> str:
        session_id = str(uuid.uuid4())
        if db is None:
            return session_id
        try:
            db.execute(
                """
                INSERT INTO sessions (
                    session_id, repo_path, started_at, model, permission_mode
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    repo_path,
                    int(time.time() * 1000),
                    model,
                    permission_mode.value,
                ),
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][base_agent] Failed to init session: {exc}")
        return session_id

    @staticmethod
    def close_session(db: Any, session_id: str) -> None:
        if db is None:
            return
        try:
            db.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (int(time.time() * 1000), session_id),
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][base_agent] Failed to close session: {exc}")
