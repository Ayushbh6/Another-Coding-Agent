from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from aca.approval import ApprovalPolicy, ApprovalRequest
from aca.config import get_allowed_openrouter_models, get_settings, resolve_openrouter_model
from aca.master import OpenRouterMasterClassifier
from aca.llm.providers import OpenRouterProvider
from aca.prompts import COMMAND_HELP, WELCOME_BANNER, WELCOME_NAME_PROMPT, WELCOME_SUBTITLE, WELCOME_TITLE
from aca.runtime import ToolLoopRuntime
from aca.services import TriageOrchestrator
from aca.services.chat import ChatService
from aca.storage import initialize_storage


@dataclass(slots=True)
class CliSessionState:
    active_user_id: str
    active_conversation_id: str
    active_model: str
    thinking_enabled: bool = False


class TerminalApprovalPolicy(ApprovalPolicy):
    def __init__(self, console: Console) -> None:
        self._console = console

    def request(self, request: ApprovalRequest) -> bool:
        self._console.print()
        self._console.print("[bold yellow]approval[/bold yellow]> "
                            f"{request.agent_id} wants to call [bold]{request.tool_name}[/bold]")
        self._console.print(Text(request.preview, style="yellow"))
        return Confirm.ask("[bold cyan]Allow this action?[/bold cyan]", console=self._console, default=False)


def parse_command(raw_text: str) -> tuple[str, str]:
    cleaned = raw_text.strip()
    if not cleaned.startswith("/"):
        return "", cleaned

    parts = cleaned[1:].split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    argument = parts[1].strip() if len(parts) > 1 else ""
    return command, argument


class ChatCLIApp:
    def __init__(self) -> None:
        self.console = Console()
        self.settings = get_settings()
        self.storage = initialize_storage(self.settings)
        self.provider = OpenRouterProvider(title="ACA-CLI")
        self.runtime = ToolLoopRuntime(provider=self.provider, tool_registry={})
        self.classifier = OpenRouterMasterClassifier(title="ACA-CLI-Master-Classifier")
        self.chat_service = ChatService(
            session_factory=self.storage.session_factory,
            runtime=self.runtime,
        )
        self.triage = TriageOrchestrator(
            session_factory=self.storage.session_factory,
            provider=self.provider,
            classifier=self.classifier,
            approval_policy=TerminalApprovalPolicy(self.console),
        )
        self.state: CliSessionState | None = None

    def run(self) -> None:
        user = self.chat_service.get_single_user()
        if user is None:
            user = self._run_first_launch()
            conversation = self.chat_service.create_conversation(user.id)
        else:
            self._print_welcome_banner(user.name)
            conversation = self.chat_service.get_or_create_active_conversation(user.id)

        self.state = CliSessionState(
            active_user_id=user.id,
            active_conversation_id=conversation.id,
            active_model=conversation.active_model or self.settings.default_openrouter_model,
            thinking_enabled=conversation.thinking_enabled,
        )

        self.console.print(
            Panel.fit(
                COMMAND_HELP,
                title="[bold magenta]Commands[/bold magenta]",
                border_style="bright_cyan",
            )
        )
        self.console.print(self._conversation_header(conversation.title))

        while True:
            try:
                raw_text = Prompt.ask(f"[bold cyan]{user.name}[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break

            if not raw_text.strip():
                continue

            command, argument = parse_command(raw_text)
            if command:
                if not self._handle_command(command, argument):
                    break
                continue

            self._send_chat_message(raw_text)

    def _run_first_launch(self):
        self._print_welcome_banner()

        name = ""
        while not name:
            name = Prompt.ask(f"[bold cyan]{WELCOME_NAME_PROMPT}[/bold cyan]").strip()

        user = self.chat_service.initialize_user(name)
        self.console.print(f"[bright_cyan]Identity locked:[/] [bold magenta]{user.name}[/bold magenta]")
        return user

    def _print_welcome_banner(self, name: str | None = None) -> None:
        subtitle = WELCOME_SUBTITLE
        if name:
            subtitle = f"Welcome back, [bold]{name}[/bold]! {subtitle}"

        self.console.print(
            Panel.fit(
                f"[bold magenta]{WELCOME_BANNER}[/bold magenta]\n\n[bright_cyan]{subtitle}[/bright_cyan]",
                title=f"[bold magenta]{WELCOME_TITLE}[/bold magenta]",
                border_style="bright_magenta",
            )
        )

    def _handle_command(self, command: str, argument: str) -> bool:
        if self.state is None:
            raise RuntimeError("CLI session state is not initialized.")

        if command == "exit":
            return False
        if command == "show":
            self.console.print(
                Panel.fit(
                    COMMAND_HELP,
                    title="[bold magenta]Commands[/bold magenta]",
                    border_style="bright_cyan",
                )
            )
            return True
        if command == "new":
            conversation = self.chat_service.create_conversation_with_settings(
                user_id=self.state.active_user_id,
                active_model=self.state.active_model,
                thinking_enabled=self.state.thinking_enabled,
            )
            self.state.active_conversation_id = conversation.id
            self.console.print(self._conversation_header(conversation.title))
            return True
        if command == "clear":
            self.chat_service.soft_clear_conversation(self.state.active_conversation_id)
            summary = self.chat_service.get_conversation_summary(self.state.active_conversation_id)
            self.console.print("[bold magenta]Current chat cleared.[/bold magenta]")
            self.console.print(self._token_status(summary.total_input_tokens, summary.total_output_tokens, summary.total_tokens))
            return True
        if command == "delete":
            self._handle_delete_command()
            return True
        if command == "list":
            self._handle_list_command()
            return True
        if command == "model":
            self._handle_model_command(argument)
            return True
        if command == "thinking":
            self._handle_thinking_command(argument)
            return True

        self.console.print("[bold red]Unknown command.[/bold red]")
        self.console.print(COMMAND_HELP)
        return True

    def _handle_model_command(self, argument: str) -> None:
        if self.state is None:
            return

        options = get_allowed_openrouter_models()
        if argument:
            resolved = self._resolve_model_choice(argument, options)
            if resolved is None:
                self.console.print("[bold red]Invalid model choice.[/bold red]")
                return
            summary = self.chat_service.update_conversation_model(self.state.active_conversation_id, resolved)
            self.state.active_model = resolved
            self.state.thinking_enabled = summary.thinking_enabled
            self.console.print(f"[bright_cyan]Model set to:[/] [bold magenta]{resolved}[/bold magenta]")
            return

        table = Table(title="Allowed Models", border_style="bright_cyan")
        table.add_column("#", style="bright_magenta")
        table.add_column("Key", style="bold cyan")
        table.add_column("Model ID", style="white")
        table.add_column("Current", style="green")
        for index, (key, model_id) in enumerate(options, start=1):
            table.add_row(str(index), key, model_id, "yes" if model_id == self.state.active_model else "")
        self.console.print(table)
        choice = Prompt.ask("[bold cyan]Select model number or key[/bold cyan]", default="")
        if not choice:
            return
        resolved = self._resolve_model_choice(choice, options)
        if resolved is None:
            self.console.print("[bold red]Invalid model choice.[/bold red]")
            return
        summary = self.chat_service.update_conversation_model(self.state.active_conversation_id, resolved)
        self.state.active_model = resolved
        self.state.thinking_enabled = summary.thinking_enabled
        self.console.print(f"[bright_cyan]Model set to:[/] [bold magenta]{resolved}[/bold magenta]")

    def _handle_thinking_command(self, argument: str) -> None:
        if self.state is None:
            return
        if not argument:
            status = "ON" if self.state.thinking_enabled else "OFF"
            self.console.print(f"[bright_cyan]Thinking is currently:[/] [bold magenta]{status}[/bold magenta]")
            return

        value = argument.strip().lower()
        if value not in {"on", "off"}:
            self.console.print("[bold red]Use /thinking ON or /thinking OFF[/bold red]")
            return

        summary = self.chat_service.update_conversation_thinking(
            self.state.active_conversation_id,
            value == "on",
        )
        self.state.thinking_enabled = summary.thinking_enabled
        if summary.active_model:
            self.state.active_model = summary.active_model
        self.console.print(
            f"[bright_cyan]Thinking set to:[/] [bold magenta]{'ON' if self.state.thinking_enabled else 'OFF'}[/bold magenta]"
        )

    def _handle_list_command(self) -> None:
        if self.state is None:
            return

        conversations = self.chat_service.list_conversations(self.state.active_user_id)
        table = Table(title="Chats", border_style="bright_cyan")
        table.add_column("#", style="bright_magenta")
        table.add_column("Title", style="bold cyan")
        table.add_column("Messages", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Current", style="green")

        for index, conversation in enumerate(conversations, start=1):
            table.add_row(
                str(index),
                self._truncate_title(conversation.title),
                str(conversation.visible_message_count),
                str(conversation.total_tokens),
                "yes" if conversation.id == self.state.active_conversation_id else "",
            )

        self.console.print(table)
        choice = Prompt.ask("[bold cyan]Select chat number to switch, or press Enter to cancel[/bold cyan]", default="")
        if not choice:
            return
        try:
            selected_index = int(choice)
        except ValueError:
            self.console.print("[bold red]Please enter a valid chat number.[/bold red]")
            return
        if not 1 <= selected_index <= len(conversations):
            self.console.print("[bold red]Chat number out of range.[/bold red]")
            return

        selected = conversations[selected_index - 1]
        self.state.active_conversation_id = selected.id
        self.state.active_model = selected.active_model or self.settings.default_openrouter_model
        self.state.thinking_enabled = selected.thinking_enabled
        self.console.print(self._conversation_header(selected.title))

    def _handle_delete_command(self) -> None:
        if self.state is None:
            return

        confirmed = Confirm.ask(
            "[bold red]Delete the current chat permanently?[/bold red]",
            console=self.console,
            default=False,
        )
        if not confirmed:
            self.console.print("[bright_cyan]Delete cancelled.[/bright_cyan]")
            return

        self.chat_service.delete_conversation(self.state.active_conversation_id)
        replacement = self.chat_service.create_conversation_with_settings(
            user_id=self.state.active_user_id,
            active_model=self.state.active_model,
            thinking_enabled=self.state.thinking_enabled,
        )
        self.state.active_conversation_id = replacement.id
        self.state.active_model = replacement.active_model or self.settings.default_openrouter_model
        self.state.thinking_enabled = replacement.thinking_enabled
        self.console.print("[bold magenta]Current chat deleted.[/bold magenta]")
        self.console.print(self._conversation_header(replacement.title))

    def _send_chat_message(self, raw_text: str) -> None:
        if self.state is None:
            return

        spinner = self.console.status("[bold magenta]Neon relay warming up...[/bold magenta]", spinner="dots")
        spinner.start()
        active_streams: set[tuple[str, str]] = set()
        last_stream_type: str | None = None
        stream_source = getattr(self, "triage", None)
        use_legacy_stream = stream_source is None

        try:
            event_iterator = (
                self.chat_service.stream_chat_turn(
                    conversation_id=self.state.active_conversation_id,
                    user_input=raw_text,
                    model=self.state.active_model,
                    thinking_enabled=self.state.thinking_enabled,
                )
                if use_legacy_stream
                else self.triage.stream_turn(
                    conversation_id=self.state.active_conversation_id,
                    user_input=raw_text,
                    model=self.state.active_model,
                    thinking_enabled=self.state.thinking_enabled,
                )
            )

            for event in event_iterator:
                if use_legacy_stream:
                    if event.type == "reasoning.delta":
                        spinner.stop()
                        if ("assistant", "reasoning") not in active_streams:
                            self.console.print("[italic bright_black]thinking[/italic bright_black]> ", end="")
                            active_streams.add(("assistant", "reasoning"))
                        self.console.print(
                            Text(event.thinking_text or "", style="italic bright_black"),
                            end="",
                            soft_wrap=True,
                            highlight=False,
                        )
                        last_stream_type = "reasoning"
                        continue

                    if event.type == "text.delta":
                        spinner.stop()
                        if ("assistant", "text") not in active_streams:
                            if last_stream_type == "reasoning":
                                self.console.print()
                            self.console.print("[bold magenta]assistant[/bold magenta]> ", end="")
                            active_streams.add(("assistant", "text"))
                        self.console.print(event.text or "", end="", soft_wrap=True, highlight=False)
                        last_stream_type = "text"
                        continue

                    if event.type == "completed" and event.summary is not None:
                        spinner.stop()
                        if last_stream_type == "reasoning":
                            self.console.print()
                        if ("assistant", "text") not in active_streams:
                            self.console.print("[bold magenta]assistant[/bold magenta]> ", end="")
                            self.console.print(event.final_answer or "", end="", soft_wrap=True, highlight=False)
                        self.console.print()
                        self.console.print(
                            self._token_status(
                                event.summary.total_input_tokens,
                                event.summary.total_output_tokens,
                                event.summary.total_tokens,
                            )
                        )
                        break
                    continue

                if event.type == "phase.started":
                    spinner.stop()
                    label = event.agent or "agent"
                    phase = event.phase or "phase"
                    phase_text = f"[{phase}]"
                    if event.message:
                        self._print_agent_line(label, f"{phase_text} {event.message}")
                    else:
                        self._print_agent_line(label, phase_text)
                    active_streams.discard((label, "reasoning"))
                    active_streams.discard((label, "text"))
                    last_stream_type = None
                    continue

                if event.type == "reasoning.delta":
                    spinner.stop()
                    label = event.agent or "agent"
                    if (label, "reasoning") not in active_streams:
                        if last_stream_type == "text":
                            self.console.print()
                        self.console.print(f"[italic bright_black]{label} thinking[/italic bright_black]> ", end="")
                        active_streams.add((label, "reasoning"))
                    self.console.print(Text(event.thinking_text or "", style="italic bright_black"), end="", soft_wrap=True, highlight=False)
                    last_stream_type = "reasoning"
                    continue

                if event.type == "text.delta":
                    spinner.stop()
                    label = event.agent or "agent"
                    if (label, "text") not in active_streams:
                        if last_stream_type == "reasoning":
                            self.console.print()
                        self.console.print(f"[bold magenta]{label}[/bold magenta]> ", end="")
                        active_streams.add((label, "text"))
                    self.console.print(event.text or "", end="", soft_wrap=True, highlight=False)
                    last_stream_type = "text"
                    continue

                if event.type == "worker.status":
                    spinner.stop()
                    if last_stream_type in {"reasoning", "text"}:
                        self.console.print()
                    self._print_agent_line(event.agent or "worker", event.message or "")
                    last_stream_type = None
                    continue

                if event.type == "phase.completed":
                    spinner.stop()
                    if last_stream_type in {"reasoning", "text"}:
                        self.console.print()
                    if event.message:
                        self._print_agent_line(event.agent or "agent", event.message)
                    last_stream_type = None
                    continue

                if event.type == "completed" and event.summary is not None:
                    spinner.stop()
                    if last_stream_type == "reasoning":
                        self.console.print()
                    if last_stream_type != "text":
                        spinner.stop()
                        self.console.print(f"[bold magenta]{event.agent or 'assistant'}[/bold magenta]> ", end="")
                        self.console.print(event.final_answer or "", end="", soft_wrap=True, highlight=False)
                    self.console.print()
                    self.console.print(
                        self._token_status(
                            event.summary.total_input_tokens,
                            event.summary.total_output_tokens,
                            event.summary.total_tokens,
                        )
                    )
                    break
        finally:
            spinner.stop()

    def _print_agent_line(self, label: str, message: str) -> None:
        line = Text()
        line.append(label, style="bold magenta")
        line.append("> ")
        if message:
            line.append(message)
        self.console.print(line)

    def _resolve_model_choice(self, choice: str, options: list[tuple[str, str]]) -> str | None:
        cleaned = choice.strip()
        if not cleaned:
            return None

        if cleaned.isdigit():
            index = int(cleaned)
            if 1 <= index <= len(options):
                return options[index - 1][1]
            return None

        if cleaned in dict(options):
            return dict(options)[cleaned]

        resolved = resolve_openrouter_model(cleaned)
        allowed_models = {model_id for _, model_id in options}
        if resolved in allowed_models:
            return resolved

        return None

    def _conversation_header(self, title: str) -> Panel:
        return Panel.fit(
            f"[bright_cyan]Current chat:[/] [bold magenta]{self._truncate_title(title)}[/bold magenta]",
            border_style="bright_magenta",
        )

    def _token_status(self, input_tokens: int, output_tokens: int, total_tokens: int) -> str:
        return (
            f"[dim]chat tokens[/dim] "
            f"[bright_cyan]in[/bright_cyan]={input_tokens} "
            f"[bright_magenta]out[/bright_magenta]={output_tokens} "
            f"[bold white]total={total_tokens}[/bold white]"
        )

    def _truncate_title(self, title: str, max_words: int = 5) -> str:
        words = title.split()
        if len(words) <= max_words:
            return title
        return " ".join(words[:max_words]) + " ..."


def main() -> None:
    ChatCLIApp().run()
