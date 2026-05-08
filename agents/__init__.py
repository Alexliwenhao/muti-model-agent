"""Sub-Agent system for Multi-Model Agent

Claude Code 子代理架构:
- 子代理在独立上下文中运行
- 只能返回摘要，防止上下文污染
- 支持 Task tool 工具创建子代理
- 深度限制为 1（子代理不能再创建子代理）
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import CallbackManager

from config import Config, ModelConfig


class SubAgentType(Enum):
    EXPLORE = "explore"
    PLAN = "plan"
    GENERAL = "general"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"


@dataclass
class SubAgentConfig:
    name: str
    description: str
    agent_type: SubAgentType
    model: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    max_iterations: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.agent_type.value,
            "model": self.model,
            "tools": self.tools,
            "system_prompt": self.system_prompt
        }


@dataclass
class SubAgentResult:
    agent_id: str
    agent_name: str
    status: str
    summary: str
    findings: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    tokens_used: int = 0


class SubAgent:
    """Sub-agent with isolated context

    Claude Code 子代理特点:
    - 独立上下文窗口
    - 只返回摘要给父代理
    - 深度限制为 1
    """

    def __init__(
        self,
        config: SubAgentConfig,
        parent_context: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.agent_id = str(uuid.uuid4())[:8]
        self.parent_context = parent_context or {}

        self.messages: List[Any] = []
        self.status = "initialized"
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None

        self._initialize_system_prompt()

    def _initialize_system_prompt(self):
        base_prompt = self.config.system_prompt or f"""You are {self.config.name}, a specialized sub-agent.

Your role: {self.config.description}

Guidelines:
1. Work autonomously within your specialized area
2. Return concise findings to the parent agent
3. Do not spawn additional sub-agents
4. Focus on your assigned task"""

        if self.parent_context:
            context_info = f"\n\nParent context:\n"
            for key, value in self.parent_context.items():
                context_info += f"- {key}: {value}\n"
            base_prompt += context_info

        self.messages.append(SystemMessage(content=base_prompt))

    def add_message(self, role: str, content: str):
        self.messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))

    def get_isolated_context(self) -> List[Any]:
        """Get messages for this agent's isolated context"""
        return self.messages.copy()

    def get_summary(self) -> str:
        """Generate summary for parent agent"""
        if len(self.messages) <= 2:
            return "Agent completed with minimal interaction."

        summary_prompt = f"""Summarize the work completed by {self.config.name}:

Task: {self.config.description}

Recent messages:
{chr(10).join([f'{type(m).__name__}: {str(m.content)[:200]}' for m in self.messages[-6:]])}

Provide a concise summary (2-3 sentences) suitable for returning to the parent agent."""

        return summary_prompt

    async def execute(self, task: str, llm: Any, tools: Optional[List[Any]] = None) -> SubAgentResult:
        """Execute the sub-agent task"""
        from datetime import datetime as dt

        self.status = "running"
        self.started_at = dt.now()
        self.add_message("user", task)

        try:
            response = await llm.ainvoke(self.messages)
            self.add_message("assistant", str(response.content))
            self.messages.append(response)

            self.status = "completed"
            self.ended_at = dt.now()

            return SubAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                status="completed",
                summary=str(response.content)[:500],
                findings=[str(response.content)],
                execution_time=(self.ended_at - self.started_at).total_seconds()
            )

        except Exception as e:
            self.status = "failed"
            self.ended_at = dt.now()

            return SubAgentResult(
                agent_id=self.agent_id,
                agent_name=self.config.name,
                status="failed",
                summary=f"Error: {str(e)}",
                findings=[],
                execution_time=(self.ended_at - self.started_at).total_seconds()
            )


class SubAgentManager:
    """Manages sub-agents for the main agent

    Claude Code 子代理管理:
    - 跟踪所有活跃子代理
    - 管理子代理生命周期
    - 收集和聚合结果
    """

    DEFAULT_AGENTS = {
        "explore": SubAgentConfig(
            name="Explorer",
            description="Explore codebase and find relevant files",
            agent_type=SubAgentType.EXPLORE,
            tools=["Read", "Glob", "Grep"]
        ),
        "plan": SubAgentConfig(
            name="Planner",
            description="Plan and analyze complex tasks",
            agent_type=SubAgentType.PLAN,
            tools=["Read", "Grep"]
        ),
        "general": SubAgentConfig(
            name="GeneralPurpose",
            description="Handle general-purpose multi-step operations",
            agent_type=SubAgentType.GENERAL,
            tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob"]
        )
    }

    def __init__(self, config: Config):
        self.config = config
        self.active_agents: Dict[str, SubAgent] = {}
        self.completed_results: List[SubAgentResult] = []

    def create_sub_agent(
        self,
        agent_config: Optional[SubAgentConfig] = None,
        agent_type: Optional[SubAgentType] = None,
        custom_tools: Optional[List[str]] = None,
        parent_context: Optional[Dict[str, Any]] = None
    ) -> SubAgent:
        """Create a new sub-agent"""
        if agent_config:
            config = agent_config
        elif agent_type:
            if agent_type.value in self.DEFAULT_AGENTS:
                config = self.DEFAULT_AGENTS[agent_type.value]
            else:
                config = SubAgentConfig(
                    name=agent_type.value,
                    description=f"Handle {agent_type.value} tasks",
                    agent_type=agent_type
                )
        else:
            config = self.DEFAULT_AGENTS["general"]

        if custom_tools:
            config.tools = custom_tools

        agent = SubAgent(config, parent_context)
        self.active_agents[agent.agent_id] = agent
        return agent

    def create_from_markdown(self, markdown_content: str) -> Optional[SubAgent]:
        """Create sub-agent from markdown definition (Claude Code style)"""
        import re

        yaml_match = re.search(r'---\n(.*?)\n---', markdown_content, re.DOTALL)
        if not yaml_match:
            return None

        yaml_content = yaml_match.group(1)
        config_dict = {}

        for line in yaml_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                config_dict[key.strip()] = value.strip()

        agent_config = SubAgentConfig(
            name=config_dict.get('name', 'CustomAgent'),
            description=config_dict.get('description', 'Custom sub-agent'),
            agent_type=SubAgentType(config_dict.get('type', 'general')),
            model=config_dict.get('model'),
            tools=config_dict.get('tools', '').split(', ') if config_dict.get('tools') else [],
            system_prompt=markdown_content.split('---', 2)[-1].strip()
        )

        return self.create_sub_agent(agent_config)

    def get_agent(self, agent_id: str) -> Optional[SubAgent]:
        """Get an active agent by ID"""
        return self.active_agents.get(agent_id)

    def complete_agent(self, agent_id: str, result: SubAgentResult):
        """Mark an agent as completed and store result"""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
        self.completed_results.append(result)

    def get_results(self) -> List[SubAgentResult]:
        """Get all completed results"""
        return self.completed_results

    def clear_results(self):
        """Clear completed results"""
        self.completed_results = []

    def get_active_count(self) -> int:
        """Get count of active agents"""
        return len(self.active_agents)
