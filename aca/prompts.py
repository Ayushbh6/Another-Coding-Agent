from __future__ import annotations


HELPFUL_ASSISTANT_PROMPT = """
You are a helpful assistant.

Be clear, direct, and useful. Keep answers concise unless the user asks for depth.
If you are unsure, say so plainly instead of bluffing.
""".strip()


MASTER_CLASSIFICATION_PROMPT = """
You are the Master classifier for a coding agent.

Classify the current user turn into exactly one intent:
- chat: casual or conversational, no meaningful repo work required
- analyze: inspect or explain code/repo behavior, but do not change code
- implement: make or plan to make code or file changes

Return a strict JSON object with these keys:
- intent
- task_title
- task_description
- reasoning_summary

Rules:
- intent must be one of: chat, analyze, implement
- task_title and task_description must be concise but useful
- for chat, task_title and task_description may be empty strings
- reasoning_summary must be brief and explain the classification
- do not include markdown or any text outside the JSON object
""".strip()


WELCOME_BANNER = r"""
 █████╗  ██████╗  █████╗
██╔══██╗██╔════╝ ██╔══██╗
███████║██║      ███████║
██╔══██║██║      ██╔══██║
██║  ██║╚██████╗ ██║  ██║
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝
""".strip("\n")


WELCOME_TITLE = "Another Coding Agent ;)"
WELCOME_SUBTITLE = "A local-first coding assisatnt that drives developmemt with you."
WELCOME_NAME_PROMPT = "Type your name"


COMMAND_HELP = """
/model       Choose one of the allowed models
/thinking    Toggle reasoning ON or OFF
/new         Start a new chat
/clear       Soft-clear the current chat
/list        List saved chats and switch to one
/exit        Exit the app
""".strip()
