"""Base class for agent tools"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Base input model for tools"""
    pass


class ToolOutput(BaseModel):
    """Base output model for tools"""
    success: bool = True
    result: Any = None
    error: Optional[str] = None


class BaseAgentTool(ABC):
    """Base class for Claude Code-style tools

    Claude Code 核心工具:
    - Bash: 执行命令
    - Read: 读取文件
    - Edit: 编辑文件
    - Write: 写入文件
    - Grep: 搜索
    - Glob: 文件匹配
    - TodoWrite: 任务管理
    """

    name: str = "base_tool"
    description: str = "Base tool description"
    input_model: type[ToolInput] = ToolInput

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> ToolOutput:
        """Execute the tool with given input"""
        pass

    async def __call__(self, **kwargs) -> str:
        """Make tool callable for LangChain"""
        try:
            result = await self.execute(kwargs)
            if result.success:
                return str(result.result)
            else:
                return f"Error: {result.error}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM binding"""
        schema = {
            "name": self.name,
            "description": self.description,
        }

        if self.input_model != ToolInput:
            schema["parameters"] = self.input_model.model_json_schema()
        else:
            schema["parameters"] = {
                "type": "object",
                "properties": {},
                "required": []
            }

        return schema
