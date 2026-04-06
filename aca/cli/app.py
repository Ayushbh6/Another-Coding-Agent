from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from aca.config import get_allowed_openrouter_models, get_settings, resolve_openrouter_model
from aca.llm.providers import OpenRouterProvider
from aca.prompts import COMMAND_HELP, WELCOME_BANNER, WELCOME_NAME_PROMPT, WELCOME_SUBTITLE, WELCOME_TITLE
from aca.runtime import ToolLoopRuntime
from aca.services.chat import ChatService, ChatStreamEvent
from aca.storage import initialize_storage


@dataclass(slots=True)
class CliSessionState:
    active_user_id: str
    active_conversation_id: str
    active_model: str
    thinking_enabled: bool = False


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
        self.chat_service = ChatService(
            session_factory=self.storage.session_factory,
            runtime=self.runtime,
        )
        self.state: CliSessionState | None = None

    def run(self) -> None:
        user = self.chat_service.get_single_user()
        if user is None:
            user = self._run_first_launch()
            conversation = self.chat_service.create_conversation(user.id)
        else:
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
        self.console.print(
            Panel.fit(
                f"[bold magenta]{WELCOME_BANNER}[/bold magenta]\n\n[bright_cyan]{WELCOME_SUBTITLE}[/bright_cyan]",
                title=f"[bold magenta]{WELCOME_TITLE}[/bold magenta]",
                border_style="bright_magenta",
            )
        )

        name = ""
        while not name:
            name = Prompt.ask(f"[bold cyan]{WELCOME_NAME_PROMPT}[/bold cyan]").strip()

        user = self.chat_service.initialize_user(name)
        self.console.print(f"[bright_cyan]Identity locked:[/] [bold magenta]{user.name}[/bold magenta]")
        return user

    def _handle_command(self, command: str, argument: str) -> bool:
        if self.state is None:
            raise RuntimeError("CLI session state is not initialized.")

        if command == "exit":
            return False
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

    def _send_chat_message(self, raw_text: str) -> None:
        if self.state is None:
            return

        spinner = self.console.status("[bold magenta]Neon relay warming up...[/bold magenta]", spinner="dots")
        spinner.start()
        started_stream = False

        try:
            for event in self.chat_service.stream_chat_turn(
                conversation_id=self.state.active_conversation_id,
                user_input=raw_text,
                model=self.state.active_model,
                thinking_enabled=self.state.thinking_enabled,
            ):
                if event.type == "text.delta":
                    if not started_stream:
                        spinner.stop()
                        self.console.print("[bold magenta]assistant[/bold magenta]> ", end="")
                        started_stream = True
                    self.console.print(event.text or "", end="", soft_wrap=True, highlight=False)
                    continue

                if event.type == "completed" and event.summary is not None:
                    if not started_stream:
                        spinner.stop()
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
        finally:
            spinner.stop()

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
