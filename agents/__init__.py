"""Sub-Agent system for Multi-Model Agent

Claude Code sub-agent architecture:
- Sub-agents run in isolated contexts
- Only return summaries to prevent context pollution
- Support Task tool for creating sub-agents
- Depth limit of 1 (sub-agents cannot create more sub-agents)
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
    """Built-in sub-agent types matching Claude Code"""
    EXPLORE = "explore"       # One-shot: explore codebase
    PLAN = "plan"             # One-shot: plan and analyze
    GENERAL = "general"       # General purpose
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    WORKER = "worker"         # Coordinator mode worker
    VERIFICATION = "verification"  # Verify implementations


@dataclass
class SubAgentConfig:
    """Sub-agent configuration"""
    name: str
    description: str
    agent_type: SubAgentType
    model: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    max_iterations: int = 50
    is_one_shot: bool = False  # True for Explore, Plan types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.agent_type.value,
            "model": self.model,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "is_one_shot": self.is_one_shot
        }


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution"""
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

    Claude Code sub-agent characteristics:
    - Isolated context window
    - Only returns summary to parent
    - Depth limit of 1
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
        return self.messages.copy()

    def get_summary(self) -> str:
        if len(self.messages) <= 2:
            return "Agent completed with minimal interaction."

        summary_prompt = f"""Summarize the work completed by {self.config.name}:

Task: {self.config.description}

Recent messages:
{chr(10).join([f'{type(m).__name__}: {str(m.content)[:200]}' for m in self.messages[-6:]])}

Provide a concise summary (2-3 sentences) suitable for returning to the parent agent."""

        return summary_prompt

    async def execute(self, task: str, llm: Any, tools: Optional[List[Any]] = None) -> SubAgentResult:
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

    Claude Code sub-agent management:
    - Track all active sub-agents
    - Manage sub-agent lifecycle
    - Collect and aggregate results
    """

    DEFAULT_AGENTS = {
        "explore": SubAgentConfig(
            name="Explorer",
            description="Explore codebase and find relevant files",
            agent_type=SubAgentType.EXPLORE,
            tools=["Read", "Glob", "Grep"],
            is_one_shot=True
        ),
        "plan": SubAgentConfig(
            name="Planner",
            description="Plan and analyze complex tasks",
            agent_type=SubAgentType.PLAN,
            tools=["Read", "Grep"],
            is_one_shot=True
        ),
        "general": SubAgentConfig(
            name="GeneralPurpose",
            description="Handle general-purpose multi-step operations",
            agent_type=SubAgentType.GENERAL,
            tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob"]
        ),
        "verification": SubAgentConfig(
            name="Verifier",
            description="Verify code changes work correctly",
            agent_type=SubAgentType.VERIFICATION,
            tools=["Bash", "Read", "Grep"],
            is_one_shot=True
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
        return self.active_agents.get(agent_id)

    def complete_agent(self, agent_id: str, result: SubAgentResult):
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
        self.completed_results.append(result)

    def get_results(self) -> List[SubAgentResult]:
        return self.completed_results

    def clear_results(self):
        self.completed_results = []

    def get_active_count(self) -> int:
        return len(self.active_agents)


# Re-export team coordination components
from agents.team import Coordinator, TeamManager, Worker, WorkerStatus, TaskNotification

__all__ = [
    "SubAgent", "SubAgentManager", "SubAgentConfig", "SubAgentResult", "SubAgentType",
    "Coordinator", "TeamManager", "Worker", "WorkerStatus", "TaskNotification"
]
