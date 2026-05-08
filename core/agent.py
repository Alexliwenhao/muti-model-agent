"""Multi-Model Agent Core Implementation

参考 Claude Code 架构:
- 简单的 while(tool_call) 循环
- 模型决定何时调用工具
- 8个核心工具
"""
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from config import Config, ModelConfig
from context.window import ContextWindow
from tools.toolbox import ToolBox

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    COMPLETED = "completed"
    ERROR = "error"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class AgentResponse:
    status: AgentStatus
    message: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    iterations: int = 0


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class MultiModelAgent:
    """Multi-Model Agent based on Claude Code architecture"""

    def __init__(
        self,
        config: Optional[Config] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        callback_handler: Optional[Any] = None
    ):
        self.config = config or Config()
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.callback_handler = callback_handler

        self._llm: Optional[Any] = None
        self._toolbox = ToolBox(tools or [])
        self._context_window = ContextWindow(
            max_tokens=self.config.agent.context_window
        )
        self._initialize_llm()

        self.status = AgentStatus.IDLE
        self.current_task: Optional[str] = None
        self.iteration_count = 0

    def _initialize_llm(self):
        primary = self.config.get_primary_model()
        if primary.provider == "anthropic":
            self._llm = ChatAnthropic(
                model=primary.model,
                anthropic_api_key=primary.api_key,
                temperature=primary.temperature,
                max_tokens=primary.max_tokens,
            )
        elif primary.provider == "openai":
            self._llm = ChatOpenAI(
                model=primary.model,
                api_key=primary.api_key,
                base_url=primary.base_url,
                temperature=primary.temperature,
            )
        self._llm = self._llm.bind_tools(self._toolbox.get_tool_schemas())

    def _get_default_system_prompt(self) -> str:
        return """You are a helpful AI coding assistant based on Claude Code architecture.

You have access to the following tools to interact with the filesystem and run commands:
- Bash: Execute shell commands
- Read: Read file contents
- Edit: Make targeted edits to files
- Write: Write or overwrite files
- Grep: Search for patterns in files
- Glob: Find files by pattern
- TodoWrite: Create and manage task lists

Guidelines:
1. Think step by step before taking actions
2. Use tools when you need to read, write, or modify files
3. Verify your changes before moving on
4. Provide clear explanations of what you're doing
5. If an action fails, try an alternative approach

Remember: You are in a CLI environment. Be concise and action-oriented."""

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def add_tools(self, tools: List[Any]):
        self._toolbox.add_tools(tools)
        if self._llm:
            self._llm = self._llm.bind_tools(self._toolbox.get_tool_schemas())

    def get_available_tools(self) -> List[str]:
        return self._toolbox.get_tool_names()

    async def run(self, task: str, max_iterations: Optional[int] = None) -> AgentResponse:
        """Execute a task using the agentic loop

        This is the core execution loop inspired by Claude Code:
        while (tool_call) - 模型决定何时调用工具
        """
        self.status = AgentStatus.RUNNING
        self.current_task = task
        max_iters = max_iterations or self.config.agent.max_iterations
        self.iteration_count = 0

        messages = [SystemMessage(content=self.system_prompt)]
        self._context_window.add_message(HumanMessage(content=task))

        tool_calls = []
        intermediate_steps = []

        try:
            while self.iteration_count < max_iters:
                self.iteration_count += 1
                logger.info(f"Iteration {self.iteration_count}/{max_iters}")

                context_messages = self._context_window.get_messages()
                full_messages = messages + context_messages

                response = await self._llm.ainvoke(full_messages)

                self._context_window.add_message(response)
                messages.append(response)

                if not hasattr(response, "tool_calls") or not response.tool_calls:
                    self.status = AgentStatus.COMPLETED
                    return AgentResponse(
                        status=AgentStatus.COMPLETED,
                        message=response.content if hasattr(response, "content") else str(response),
                        tool_calls=tool_calls,
                        intermediate_steps=intermediate_steps,
                        iterations=self.iteration_count
                    )

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    logger.info(f"Tool call: {tool_name}")
                    tool_calls.append({
                        "name": tool_name,
                        "args": tool_args
                    })

                    if tool_name == "Bash":
                        result = await self._execute_bash(tool_args)
                    elif tool_name == "Read":
                        result = self._execute_read(tool_args)
                    elif tool_name == "Write":
                        result = self._execute_write(tool_args)
                    elif tool_name == "Edit":
                        result = self._execute_edit(tool_args)
                    elif tool_name == "Grep":
                        result = self._execute_grep(tool_args)
                    elif tool_name == "Glob":
                        result = self._execute_glob(tool_args)
                    else:
                        result = await self._toolbox.execute_tool(tool_name, tool_args)

                    intermediate_steps.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result
                    })

                    tool_msg = {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.get("id", f"call_{len(tool_calls)}")
                    }
                    self._context_window.add_message(tool_msg)
                    messages.append(tool_msg)

                if self._context_window.should_compact():
                    messages = await self._compact_context(messages)
                    self._context_window.reset()

            self.status = AgentStatus.MAX_ITERATIONS
            return AgentResponse(
                status=AgentStatus.MAX_ITERATIONS,
                message="Maximum iterations reached",
                tool_calls=tool_calls,
                intermediate_steps=intermediate_steps,
                iterations=self.iteration_count
            )

        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent error: {e}")
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="",
                tool_calls=tool_calls,
                intermediate_steps=intermediate_steps,
                error=str(e),
                iterations=self.iteration_count
            )

    async def run_sync(self, task: str, max_iterations: Optional[int] = None) -> AgentResponse:
        """Synchronous wrapper for run"""
        return await self.run(task, max_iterations)

    async def _compact_context(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Compact context to stay within window limits"""
        if len(messages) <= 4:
            return messages

        summary_prompt = """Summarize the conversation so far, keeping:
1. The original task
2. Key findings or important information discovered
3. Current progress and pending tasks

Be concise - this summary will replace the full conversation history."""

        summary_response = await self._llm.ainvoke(
            messages[:-4] + [HumanMessage(content=summary_prompt)]
        )

        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.current_task or "Previous task"),
            summary_response,
            messages[-1]
        ]

    async def _execute_bash(self, args: Dict[str, Any]) -> str:
        """Execute bash command"""
        import subprocess
        command = args.get("command", "")
        cwd = args.get("cwd")
        timeout = args.get("timeout", 60)

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"\nStdout:\n{result.stdout}"
            if result.stderr:
                output += f"\nStderr:\n{result.stderr}"
            return output
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _execute_read(self, args: Dict[str, Any]) -> str:
        """Read file contents"""
        from pathlib import Path
        file_path = args.get("file_path")
        offset = args.get("offset", 0)
        limit = args.get("limit")

        try:
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                if offset > 0:
                    for _ in range(offset):
                        f.readline()
                content = f.read(limit) if limit else f.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _execute_write(self, args: Dict[str, Any]) -> str:
        """Write file contents"""
        from pathlib import Path
        file_path = args.get("file_path")
        content = args.get("content", "")

        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def _execute_edit(self, args: Dict[str, Any]) -> str:
        """Edit file contents"""
        from pathlib import Path
        file_path = args.get("file_path")
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")

        try:
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_str not in content:
                return f"Could not find the specified text to replace"

            new_content = content.replace(old_str, new_str)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return f"Successfully edited {file_path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"

    def _execute_grep(self, args: Dict[str, Any]) -> str:
        """Grep search in files"""
        import subprocess
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        regex = args.get("regex", True)

        try:
            cmd = ["grep", "-n"]
            if regex:
                cmd.extend(["-E", pattern])
            else:
                cmd.extend(["-F", pattern])
            cmd.append(path)

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout or "(No matches found)"
        except Exception as e:
            return f"Error grepping: {str(e)}"

    def _execute_glob(self, args: Dict[str, Any]) -> str:
        """Glob file search"""
        from pathlib import Path
        pattern = args.get("pattern", "*")
        path = args.get("path", ".")

        try:
            matches = list(Path(path).glob(pattern))
            return "\n".join([str(m) for m in matches]) or "(No matches found)"
        except Exception as e:
            return f"Error globbing: {str(e)}"
