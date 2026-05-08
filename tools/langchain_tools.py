"""Claude Code-style tools implemented with LangChain

8个核心工具:
1. Bash - 执行命令
2. Read - 读取文件
3. Edit - 编辑文件
4. Write - 写入文件
5. Grep - 搜索
6. Glob - 文件匹配
7. TodoWrite - 任务管理
8. Task - 子代理（已在agents模块实现）
"""
from typing import Optional, List, Type
from pathlib import Path
import subprocess
import asyncio

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class BashInput(BaseModel):
    command: str = Field(description="Shell command to execute")
    cwd: Optional[str] = Field(default=None, description="Working directory")
    timeout: Optional[int] = Field(default=60, description="Timeout in seconds")


class ReadInput(BaseModel):
    file_path: str = Field(description="Path to the file to read")
    offset: Optional[int] = Field(default=0, description="Line offset to start reading")
    limit: Optional[int] = Field(default=None, description="Maximum lines to read")


class WriteInput(BaseModel):
    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class EditInput(BaseModel):
    file_path: str = Field(description="Path to the file to edit")
    old_str: str = Field(description="Text to replace")
    new_str: str = Field(description="Replacement text")


class GrepInput(BaseModel):
    pattern: str = Field(description="Search pattern")
    path: str = Field(default=".", description="Directory to search in")
    regex: bool = Field(default=True, description="Use regex or literal match")


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern to match files")
    path: str = Field(default=".", description="Directory to search in")


class TodoWriteInput(BaseModel):
    todos: List[dict] = Field(description="List of todo items with id, content, status")


@tool("Bash", args_schema=BashInput, return_direct=True)
def bash_tool(command: str, cwd: Optional[str] = None, timeout: int = 60) -> str:
    """Execute a shell command. Use this to run git, npm, python, or other shell commands.

    Args:
        command: The shell command to execute
        cwd: Working directory (optional)
        timeout: Timeout in seconds (default 60)

    Returns:
        Command output or error message
    """
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
        return f"Error: {str(e)}"


@tool("Read", args_schema=ReadInput, return_direct=True)
def read_tool(file_path: str, offset: int = 0, limit: Optional[int] = None) -> str:
    """Read the contents of a file. Use this to view files before editing or to understand the codebase.

    Args:
        file_path: Path to the file to read
        offset: Line number to start reading from (0-indexed)
        limit: Maximum number of lines to read

    Returns:
        File contents or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        with open(path, 'r', encoding='utf-8') as f:
            if offset > 0:
                for _ in range(offset):
                    f.readline()
            content = f.read() if limit is None else ''.join([f.readline() for _ in range(limit)])

        if not content:
            return "(Empty file or end of file reached)"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool("Write", args_schema=WriteInput, return_direct=True)
def write_tool(file_path: str, content: str) -> str:
    """Write content to a file. Use this to create new files or overwrite existing files.

    Args:
        file_path: Path where to write the file
        content: Content to write

    Returns:
        Success or error message
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool("Edit", args_schema=EditInput, return_direct=True)
def edit_tool(file_path: str, old_str: str, new_str: str) -> str:
    """Make a targeted edit to a file. Use this to modify specific parts of a file.

    Args:
        file_path: Path to the file to edit
        old_str: The exact text to replace (must match exactly)
        new_str: The replacement text

    Returns:
        Success or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_str not in content:
            return f"Could not find the specified text to replace.\n\nExpected to find:\n{old_str}\n\nIn file: {file_path}"

        new_content = content.replace(old_str, new_str)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return f"Successfully edited {file_path}"
    except Exception as e:
        return f"Error editing file: {str(e)}"


@tool("Grep", args_schema=GrepInput, return_direct=True)
def grep_tool(pattern: str, path: str = ".", regex: bool = True) -> str:
    """Search for a pattern in files. Use this to find code, text, or function definitions.

    Args:
        pattern: Search pattern (regex or literal)
        path: Directory to search in
        regex: Use regex (True) or literal match (False)

    Returns:
        Matching lines with file:line:content format
    """
    try:
        cmd = ["grep", "-n"]
        if regex:
            cmd.extend(["-E", pattern])
        else:
            cmd.extend(["-F", pattern])
        cmd.append(path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            return result.stdout
        return "(No matches found)"
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool("Glob", args_schema=GlobInput, return_direct=True)
def glob_tool(pattern: str, path: str = ".") -> str:
    """Find files matching a pattern. Use this to discover files in the codebase.

    Args:
        pattern: Glob pattern (e.g., "*.py", "**/*.ts")
        path: Directory to search in

    Returns:
        List of matching file paths
    """
    try:
        matches = list(Path(path).glob(pattern))
        if not matches:
            return "(No matches found)"
        return "\n".join([str(m.relative_to(path)) if m.is_relative_to(Path(path).resolve()) else str(m) for m in matches])
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool("TodoWrite", args_schema=TodoWriteInput, return_direct=True)
def todo_write_tool(todos: List[dict]) -> str:
    """Create and manage a todo list. Use this to track progress on multi-step tasks.

    Args:
        todos: List of todo items, each with:
            - id: Unique identifier
            - content: Description of the task
            - status: "pending", "in_progress", or "completed"
            - priority: Optional priority level

    Returns:
        Confirmation message
    """
    if not todos:
        return "Todo list cleared"

    formatted = []
    for todo in todos:
        status_emoji = {
            "pending": "📋",
            "in_progress": "🔄",
            "completed": "✅"
        }.get(todo.get("status", "pending"), "📋")

        priority = todo.get("priority", "")
        priority_str = f" [{priority}]" if priority else ""

        formatted.append(f"{status_emoji} {todo.get('id', '?')}: {todo.get('content', '')}{priority_str}")

    return "Todo List:\n" + "\n".join(formatted)


def get_all_claude_code_tools() -> List[BaseTool]:
    """Get all Claude Code-style tools as LangChain tools"""
    return [
        bash_tool,
        read_tool,
        write_tool,
        edit_tool,
        grep_tool,
        glob_tool,
        todo_write_tool,
    ]
