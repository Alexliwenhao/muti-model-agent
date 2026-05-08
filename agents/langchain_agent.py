"""LangChain ReAct Agent implementation for Multi-Model Agent

LangChain 提供了强大的 Agent 构建能力:
- ConversationalAgent
- ReAct Agent
- Plan-and-Execute Agent
"""
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent, ConversationalChatAgent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from config import Config, ModelConfig
from tools.toolbox import ToolBox


class LangChainAgent:
    """LangChain-based Agent with ReAct and Conversational capabilities

    结合 Claude Code 架构和 LangChain 最佳实践:
    - ReAct (Reasoning + Acting) 模式
    - Conversational memory
    - Tool binding
    """

    def __init__(
        self,
        config: Config,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None
    ):
        self.config = config
        self.tools = tools or []
        self.system_prompt = system_prompt or self._get_default_prompt()

        self._llm = self._initialize_llm()
        self._agent = None
        self._executor = None
        self._memory: List[BaseMessage] = []

        self._build_agent()

    def _initialize_llm(self):
        primary = self.config.get_primary_model()
        if primary.provider == "anthropic":
            return ChatAnthropic(
                model=primary.model,
                anthropic_api_key=primary.api_key,
                temperature=primary.temperature,
                max_tokens=primary.max_tokens,
            )
        elif primary.provider == "openai":
            return ChatOpenAI(
                model=primary.model,
                api_key=primary.api_key,
                base_url=primary.base_url,
                temperature=primary.temperature,
            )
        return None

    def _get_default_prompt(self) -> str:
        return """You are a helpful AI assistant.

You have access to tools to help complete tasks. Follow these steps:

1. Think about what the task requires
2. Use tools when needed
3. Observe results and continue as needed
4. Provide a final answer

Be concise and action-oriented."""

    def _build_agent(self):
        """Build the LangChain agent"""
        if not self._llm:
            return

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        tool_schemas = [tool for tool in self.tools]

        self._agent = create_react_agent(
            self._llm,
            tool_schemas,
            prompt=prompt
        )

        self._executor = AgentExecutor(
            agent=self._agent,
            tools=tool_schemas,
            verbose=True,
            max_iterations=100,
            handle_parsing_errors=True,
        )

    def add_tool(self, tool: BaseTool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        self._build_agent()

    async def run(self, task: str, chat_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """Execute task using LangChain agent"""
        try:
            result = await self._executor.ainvoke({
                "input": task,
                "chat_history": chat_history or []
            })

            self._memory.append(HumanMessage(content=task))
            self._memory.append(AIMessage(content=str(result.get("output", ""))))

            return {
                "status": "success",
                "output": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "memory": self._memory
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "output": None
            }

    def run_sync(self, task: str, chat_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """Synchronous version of run"""
        return asyncio.run(self.run(task, chat_history))

    def get_memory(self) -> List[BaseMessage]:
        """Get conversation memory"""
        return self._memory

    def clear_memory(self):
        """Clear conversation memory"""
        self._memory = []


class PlanAndExecuteAgent:
    """Plan-and-Execute Agent using LangChain

    1. Plan: 将复杂任务分解为步骤
    2. Execute: 逐个执行步骤
    3. 适合复杂的多步骤任务
    """

    def __init__(
        self,
        config: Config,
        tools: Optional[List[BaseTool]] = None
    ):
        self.config = config
        self.tools = tools or []

        self._planner_llm = self._initialize_llm("planner")
        self._executor_llm = self._initialize_llm("executor")

    def _initialize_llm(self, role: str):
        primary = self.config.get_primary_model()
        if primary.provider == "anthropic":
            return ChatAnthropic(
                model=primary.model,
                anthropic_api_key=primary.api_key,
                temperature=0.7,
                max_tokens=2048,
            )
        return ChatOpenAI(
            model=primary.model,
            api_key=primary.api_key,
            temperature=0.7,
        )

    async def plan(self, task: str) -> List[str]:
        """Create execution plan"""
        planning_prompt = f"""Break down this task into clear, sequential steps:

Task: {task}

Provide a numbered list of steps (max 5 steps). Each step should be:
- Specific and actionable
- Can be completed independently
- Clearly defined

Steps:"""

        try:
            response = await self._planner_llm.ainvoke([HumanMessage(content=planning_prompt)])
            steps_text = response.content

            steps = []
            for line in steps_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    step = line.lstrip('0123456789.-) ')
                    if step:
                        steps.append(step)

            return steps
        except Exception as e:
            return [task]

    async def execute_step(self, step: str) -> str:
        """Execute a single step"""
        executor = AgentExecutor.from_agent_and_tools(
            agent=create_react_agent(self._executor_llm, self.tools),
            tools=self.tools,
            verbose=True
        )

        try:
            result = await executor.ainvoke({"input": step})
            return result.get("output", "Step completed")
        except Exception as e:
            return f"Error: {str(e)}"

    async def run(self, task: str) -> Dict[str, Any]:
        """Execute task with planning"""
        steps = await self.plan(task)
        results = []

        for i, step in enumerate(steps, 1):
            result = await self.execute_step(step)
            results.append({
                "step": i,
                "description": step,
                "result": result
            })

        final_output = "\n".join([f"Step {r['step']}: {r['result']}" for r in results])

        return {
            "status": "success",
            "task": task,
            "steps": steps,
            "results": results,
            "final_output": final_output
        }


class ConversationalAgent:
    """Conversational Agent with memory using LangChain"""

    def __init__(
        self,
        config: Config,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None
    ):
        self.config = config
        self.tools = tools or []

        self._llm = self._initialize_llm()
        self._memory: List[BaseMessage] = []
        self._system_prompt = system_prompt or self._get_default_prompt()

        self._build_agent()

    def _initialize_llm(self):
        primary = self.config.get_primary_model()
        if primary.provider == "anthropic":
            return ChatAnthropic(
                model=primary.model,
                anthropic_api_key=primary.api_key,
                temperature=0.7,
                max_tokens=2048,
            )
        return ChatOpenAI(
            model=primary.model,
            api_key=primary.api_key,
            temperature=0.7,
        )

    def _get_default_prompt(self) -> str:
        return """You are a helpful AI coding assistant with memory of our conversation.

Guidelines:
- Remember previous context in our conversation
- Build on previous responses when relevant
- Ask clarifying questions when needed
- Provide accurate, actionable assistance"""

    def _build_agent(self):
        """Build conversational agent"""
        prompt = ConversationalChatAgent.create_prompt(
            system_message=self._system_prompt,
            tools=self.tools
        )

        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self._llm,
            tools=self.tools,
            system_message=self._system_prompt
        )

        self._executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self._memory,
            max_iterations=100,
        )

    async def run(self, task: str) -> Dict[str, Any]:
        """Execute with conversation memory"""
        try:
            result = await self._executor.ainvoke({"input": task})
            return {
                "status": "success",
                "output": result.get("output", ""),
                "memory_length": len(self._memory)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_memory(self) -> List[BaseMessage]:
        """Get conversation memory"""
        return self._memory

    def clear_memory(self):
        """Clear conversation memory"""
        self._memory.clear()
