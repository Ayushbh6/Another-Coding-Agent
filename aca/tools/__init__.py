from aca.tools import execution, memory, read, workspace, write
from aca.tools.registry import PermissionMode, ToolCategory, ToolDefinition, ToolRegistry


def register_builtin_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register ACA's standard tool set into an existing registry."""
    read.register(registry)
    write.register(registry)
    workspace.register(registry)
    memory.register(registry)
    execution.register(registry)
    return registry


def build_registry() -> ToolRegistry:
    """Create a ToolRegistry preloaded with ACA's standard tools."""
    return register_builtin_tools(ToolRegistry())


__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "PermissionMode",
    "ToolCategory",
    "register_builtin_tools",
    "build_registry",
]
