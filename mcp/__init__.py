"""MCP (Model Context Protocol) 支持模块

MCP 是 Claude Code 的插件系统，允许连接外部工具和数据源。

参考 Claude Code MCP 架构:
- MCP 服务器作为独立进程运行
- 通过标准化协议与 Agent 通信
- 支持的工具: 数据库、GitHub、Slack、Web 搜索等
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import asyncio
import subprocess
import os

from pydantic import BaseModel


class MCPServerStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """MCP 服务器配置"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    scope: str = "project"
    enabled: bool = True


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


@dataclass
class MCPToolResult:
    """MCP 工具执行结果"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


class MCPClient:
    """MCP 客户端 - 管理与 MCP 服务器的连接

    Claude Code MCP 特性:
    - 支持多种 MCP 服务器 (filesystem, github, postgres 等)
    - 自动发现可用工具
    - 安全的权限控制
    """

    DEFAULT_SERVERS = {
        "filesystem": MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
            scope="project",
            description="文件系统访问"
        ),
        "git": MCPServerConfig(
            name="git",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-git"],
            scope="project",
            description="Git 操作"
        ),
        "memory": MCPServerConfig(
            name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            scope="user",
            description="持久化记忆存储"
        ),
    }

    def __init__(self, config_path: Optional[str] = None):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.available_tools: Dict[str, List[MCPTool]] = {}
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.server_status: Dict[str, MCPServerStatus] = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self._init_default_servers()

    def _init_default_servers(self):
        """初始化默认服务器"""
        for name, config in self.DEFAULT_SERVERS.items():
            self.register_server(config)
            self.server_status[name] = MCPServerStatus.DISCONNECTED

    def register_server(self, config: MCPServerConfig):
        """注册 MCP 服务器"""
        self.servers[config.name] = config
        self.available_tools[config.name] = []

    def load_config(self, config_path: str):
        """从配置文件加载 MCP 服务器配置"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            mcp_servers = config.get("mcpServers", {})
            for name, server_config in mcp_servers.items():
                config_obj = MCPServerConfig(
                    name=name,
                    command=server_config.get("command", ""),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    scope=server_config.get("scope", "project")
                )
                self.register_server(config_obj)
                self.server_status[name] = MCPServerStatus.DISCONNECTED
                
        except Exception as e:
            print(f"Failed to load MCP config: {e}")

    def save_config(self, config_path: str):
        """保存 MCP 服务器配置到文件"""
        config = {"mcpServers": {}}
        
        for name, server in self.servers.items():
            config["mcpServers"][name] = {
                "command": server.command,
                "args": server.args,
                "env": server.env,
                "scope": server.scope
            }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    async def connect_server(self, server_name: str) -> bool:
        """连接到 MCP 服务器"""
        if server_name not in self.servers:
            return False

        config = self.servers[server_name]
        self.server_status[server_name] = MCPServerStatus.CONNECTING

        try:
            env = os.environ.copy()
            env.update(config.env)

            process = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )

            self.server_processes[server_name] = process
            self.server_status[server_name] = MCPServerStatus.CONNECTED

            await self._discover_tools(server_name)
            return True

        except Exception as e:
            self.server_status[server_name] = MCPServerStatus.ERROR
            return False

    async def disconnect_server(self, server_name: str):
        """断开 MCP 服务器连接"""
        if server_name in self.server_processes:
            process = self.server_processes[server_name]
            process.terminate()
            del self.server_processes[server_name]
        
        self.server_status[server_name] = MCPServerStatus.DISCONNECTED

    async def _discover_tools(self, server_name: str):
        """发现服务器提供的工具"""
        tools = []
        
        if server_name == "filesystem":
            tools = [
                MCPTool(
                    name="read_file",
                    description="读取文件内容",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                    server_name=server_name
                ),
                MCPTool(
                    name="write_file",
                    description="写入文件内容",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}},
                    server_name=server_name
                ),
                MCPTool(
                    name="list_directory",
                    description="列出目录内容",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                    server_name=server_name
                ),
            ]
        elif server_name == "git":
            tools = [
                MCPTool(
                    name="git_status",
                    description="显示 Git 状态",
                    input_schema={"type": "object", "properties": {}},
                    server_name=server_name
                ),
                MCPTool(
                    name="git_log",
                    description="显示 Git 提交历史",
                    input_schema={"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}},
                    server_name=server_name
                ),
                MCPTool(
                    name="git_diff",
                    description="显示文件变更",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                    server_name=server_name
                ),
            ]
        elif server_name == "memory":
            tools = [
                MCPTool(
                    name="memory_create",
                    description="创建记忆条目",
                    input_schema={"type": "object", "properties": {"entity": {"type": "string"}, "data": {"type": "object"}}},
                    server_name=server_name
                ),
                MCPTool(
                    name="memory_search",
                    description="搜索记忆",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                    server_name=server_name
                ),
            ]

        self.available_tools[server_name] = tools

    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> MCPToolResult:
        """执行 MCP 工具"""
        import time
        start_time = time.time()

        if server_name not in self.server_processes:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Server {server_name} not connected"
            )

        process = self.server_processes[server_name]
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        try:
            request_str = json.dumps(request) + "\n"
            process.stdin.write(request_str)
            process.stdin.flush()

            response_line = process.stdout.readline()
            response = json.loads(response_line)

            if "error" in response:
                return MCPToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=response["error"].get("message", "Unknown error"),
                    execution_time=time.time() - start_time
                )

            return MCPToolResult(
                tool_name=tool_name,
                success=True,
                result=response.get("result", {}),
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def get_available_tools(self) -> List[MCPTool]:
        """获取所有可用工具"""
        all_tools = []
        for tools in self.available_tools.values():
            all_tools.extend(tools)
        return all_tools

    def get_tools_by_server(self, server_name: str) -> List[MCPTool]:
        """获取指定服务器的可用工具"""
        return self.available_tools.get(server_name, [])

    def get_status(self) -> Dict[str, Any]:
        """获取 MCP 状态"""
        return {
            "servers": {
                name: {
                    "status": status.value,
                    "tool_count": len(self.available_tools.get(name, []))
                }
                for name, status in self.server_status.items()
            },
            "total_tools": len(self.get_available_tools())
        }


class MCPIntegration:
    """MCP 集成到 Agent 系统

    将 MCP 工具无缝集成到 Agent 的工具集中
    """

    def __init__(self, mcp_client: Optional[MCPClient] = None):
        self.mcp_client = mcp_client or MCPClient()

    def get_tools_for_agent(self) -> List[Dict[str, Any]]:
        """获取适合 Agent 的 MCP 工具定义"""
        tools = []

        for tool in self.mcp_client.get_available_tools():
            tools.append({
                "name": f"mcp_{tool.server_name}_{tool.name}",
                "description": f"[{tool.server_name}] {tool.description}",
                "input_schema": tool.input_schema
            })

        return tools

    async def execute_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> str:
        """执行 MCP 工具（从 Agent 调用）"""
        parts = tool_name.split("_", 2)
        if len(parts) < 3:
            return f"Invalid tool name: {tool_name}"

        server_name = parts[1]
        actual_tool_name = parts[2]

        result = await self.mcp_client.execute_tool(
            server_name,
            actual_tool_name,
            arguments
        )

        if result.success:
            return json.dumps(result.result, indent=2)
        else:
            return f"Error: {result.error}"
