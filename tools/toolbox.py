"""Toolbox for managing agent tools"""
from typing import List, Dict, Any, Optional, Callable
import asyncio

from langchain_core.tools import BaseTool

from tools.base import BaseAgentTool


class ToolBox:
    """Manages available tools for the agent

    Claude Code 使用 8 个核心工具:
    - Bash, Read, Edit, Write, Grep, Glob, Task (子代理), TodoWrite
    """

    def __init__(self, tools: Optional[List[Any]] = None):
        self._tools: Dict[str, Any] = {}
        self._tool_schemas: List[Dict[str, Any]] = []

        if tools:
            for tool in tools:
                self.add_tool(tool)

    def add_tool(self, tool: Any):
        """Add a tool to the toolbox"""
        if isinstance(tool, BaseAgentTool):
            name = tool.name
            self._tools[name] = tool
            self._tool_schemas.append(tool.get_schema())
        elif isinstance(tool, BaseTool):
            name = tool.name
            self._tools[name] = tool
            self._tool_schemas.append({
                "name": name,
                "description": tool.description,
                "parameters": tool.args_schema.schema() if tool.args_schema else {}
            })
        elif callable(tool):
            name = getattr(tool, '__name__', str(tool))
            self._tools[name] = tool

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_tool_names(self) -> List[str]:
        """Get all tool names"""
        return list(self._tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for LLM binding)"""
        return self._tool_schemas

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name with given arguments"""
        tool = self._tools.get(name)
        if not tool:
            return f"Tool not found: {name}"

        try:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**args)
            else:
                result = tool(**args)
            return str(result) if result is not None else "Tool executed successfully"
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists"""
        return name in self._tools
