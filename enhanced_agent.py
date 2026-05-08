"""Enhanced Multi-Model Agent with LangChain and Claude Code Architecture

整合所有组件:
- Claude Code 风格的核心 Agent
- LangChain ReAct Agent
- 子代理系统
- Claude Code 工具集
"""
from typing import Optional, Dict, Any, List
import asyncio
import logging

from config import Config
from core.agent import MultiModelAgent, AgentStatus
from core.session import Session, SessionManager
from agents import SubAgentManager, SubAgentConfig, SubAgentType
from agents.langchain_agent import LangChainAgent, PlanAndExecuteAgent, ConversationalAgent
from tools.langchain_tools import get_all_claude_code_tools

logger = logging.getLogger(__name__)


class MultiModelAgentSystem:
    """完整的 Multi-Model Agent 系统

    Claude Code 架构特点:
    1. 简单的 while(tool_call) 循环
    2. 8个核心工具
    3. 子代理隔离执行
    4. 上下文窗口管理
    5. 多模型协同
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        use_langchain: bool = True,
        verbose: bool = True
    ):
        self.config = config or Config()
        self.verbose = verbose
        self.use_langchain = use_langchain

        self.claude_style_agent = MultiModelAgent(
            config=self.config,
            system_prompt=self._get_claude_code_system_prompt()
        )

        self.session_manager = SessionManager()
        self.sub_agent_manager = SubAgentManager(self.config)

        if use_langchain:
            tools = get_all_claude_code_tools()
            self.langchain_agent = LangChainAgent(
                config=self.config,
                tools=tools,
                system_prompt=self._get_langchain_system_prompt()
            )
            self.plan_agent = PlanAndExecuteAgent(
                config=self.config,
                tools=tools
            )
            self.conversational_agent = ConversationalAgent(
                config=self.config,
                tools=tools
            )
        else:
            self.langchain_agent = None
            self.plan_agent = None
            self.conversational_agent = None

    def _get_claude_code_system_prompt(self) -> str:
        return """You are Claude Code, an AI coding assistant.

You have access to tools to interact with the filesystem and run commands:
- Bash(command, cwd?, timeout?): Execute shell commands
- Read(file_path, offset?, limit?): Read file contents
- Edit(file_path, old_str, new_str): Make targeted edits
- Write(file_path, content): Write or overwrite files
- Grep(pattern, path?, regex?): Search for patterns
- Glob(pattern, path?): Find files by pattern
- TodoWrite(todos): Manage task list

Guidelines:
1. Think step by step before taking actions
2. Use tools to read, write, and modify files
3. Verify your changes work correctly
4. Be concise and action-oriented
5. If something fails, try an alternative approach
6. Use TodoWrite to track multi-step tasks

You are in a CLI environment. Work with the user's current directory."""

    def _get_langchain_system_prompt(self) -> str:
        return """You are a helpful AI assistant with access to powerful tools.

Available tools:
- Bash: Execute shell commands
- Read: Read file contents
- Write: Write files
- Edit: Edit files
- Grep: Search files
- Glob: Find files
- TodoWrite: Manage tasks

Guidelines:
1. Analyze the task before taking action
2. Use tools when needed
3. Verify results
4. Provide clear explanations"""

    async def run(
        self,
        task: str,
        agent_type: str = "claude",
        sub_agents: bool = True
    ) -> Dict[str, Any]:
        """Run task using selected agent type

        Args:
            task: Task to execute
            agent_type: "claude", "langchain", "plan", or "conversational"
            sub_agents: Whether to use sub-agents for complex tasks
        """
        session = self.session_manager.get_current_session()
        if not session:
            session = self.session_manager.create_session()

        session.add_message("user", task)

        if self.verbose:
            print(f"\n[Running with {agent_type} agent]\n")

        if agent_type == "claude":
            result = await self.claude_style_agent.run(task)
            return self._format_result(result, agent_type)

        elif agent_type == "langchain":
            if self.langchain_agent:
                result = await self.langchain_agent.run(task)
                return self._format_result(result, agent_type)

        elif agent_type == "plan":
            if self.plan_agent:
                result = await self.plan_agent.run(task)
                return self._format_result(result, agent_type)

        elif agent_type == "conversational":
            if self.conversational_agent:
                result = await self.conversational_agent.run(task)
                return self._format_result(result, agent_type)

        elif agent_type == "multi":
            return await self._run_multi_agent(task, sub_agents)

        return {"status": "error", "error": "Unknown agent type"}

    async def _run_multi_agent(self, task: str, use_sub_agents: bool) -> Dict[str, Any]:
        """Run task using multiple agents in collaboration"""
        results = {}

        explore_config = SubAgentConfig(
            name="CodeExplorer",
            description="Explore codebase and find relevant files for: " + task,
            agent_type=SubAgentType.EXPLORE,
            tools=["Read", "Glob", "Grep"]
        )

        explore_agent = self.sub_agent_manager.create_sub_agent(explore_config)

        primary_model = self.config.get_primary_model()
        if primary_model.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=primary_model.model,
                anthropic_api_key=primary_model.api_key
            )
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=primary_model.model, api_key=primary_model.api_key)

        explore_result = await explore_agent.execute(task, llm)
        self.sub_agent_manager.complete_agent(explore_agent.agent_id, explore_result)
        results["explore"] = explore_result.summary

        main_result = await self.claude_style_agent.run(task)
        results["main"] = {
            "status": main_result.status.value,
            "output": main_result.message,
            "iterations": main_result.iterations
        }

        return {
            "status": "success",
            "collaboration": True,
            "results": results,
            "sub_agents_used": use_sub_agents
        }

    async def run_sub_agent(
        self,
        task: str,
        agent_type: SubAgentType = SubAgentType.GENERAL,
        parent_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a standalone sub-agent"""
        agent = self.sub_agent_manager.create_sub_agent(
            agent_type=agent_type,
            parent_context=parent_context
        )

        primary_model = self.config.get_primary_model()
        if primary_model.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=agent.config.model or primary_model.model,
                anthropic_api_key=primary_model.api_key
            )
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=agent.config.model or primary_model.model,
                api_key=primary_model.api_key
            )

        result = await agent.execute(task, llm)
        self.sub_agent_manager.complete_agent(agent.agent_id, result)

        return {
            "status": result.status,
            "summary": result.summary,
            "findings": result.findings,
            "execution_time": result.execution_time
        }

    def _format_result(self, result: Any, agent_type: str) -> Dict[str, Any]:
        """Format result based on agent type"""
        if hasattr(result, "status"):
            return {
                "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                "output": result.message if hasattr(result, "message") else str(result),
                "agent_type": agent_type
            }
        elif isinstance(result, dict):
            result["agent_type"] = agent_type
            return result
        return {"status": "success", "output": str(result), "agent_type": agent_type}

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "agent_type": "claude_code_architecture",
            "langchain_enabled": self.use_langchain,
            "available_agents": ["claude", "langchain", "plan", "conversational", "multi"],
            "sub_agents": {
                "active": self.sub_agent_manager.get_active_count(),
                "completed": len(self.sub_agent_manager.get_results())
            },
            "sessions": len(self.session_manager.list_sessions()),
            "tools_available": self.claude_style_agent.get_available_tools()
        }


async def main():
    """Main entry point for the enhanced agent system"""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Model Agent System")
    parser.add_argument("--task", "-t", type=str, help="Task to execute")
    parser.add_argument("--agent", "-a", type=str, default="claude",
                       choices=["claude", "langchain", "plan", "conversational", "multi"],
                       help="Agent type to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    system = MultiModelAgentSystem(verbose=args.verbose)

    print("=" * 60)
    print("Multi-Model Agent System (Claude Code Architecture + LangChain)")
    print("=" * 60)
    print(f"Available agents: {', '.join(system.get_status()['available_agents'])}")
    print(f"Available tools: {len(system.get_status()['tools_available'])}")
    print("-" * 60)

    if args.task:
        result = await system.run(args.task, agent_type=args.agent)
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(result.get("output", result))
    else:
        print("\nInteractive mode. Type your task or /help for commands.")
        print("Type /exit to quit.")

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["/exit", "/quit"]:
                    print("Goodbye!")
                    break

                elif user_input.lower() == "/help":
                    print("""
Commands:
  /help          Show this help
  /exit          Exit
  /status        Show system status
  /agents        List available agents
  /tools         List available tools
  /session       Show current session
  /clear         Clear session
  /agent <type>  Switch agent type (claude/langchain/plan/conversational/multi)

Just type your task to start!
                    """)
                    continue

                elif user_input.lower() == "/status":
                    import json
                    print(json.dumps(system.get_status(), indent=2))
                    continue

                elif user_input.lower() == "/agents":
                    print(f"\nAvailable agents: {', '.join(system.get_status()['available_agents'])}")
                    continue

                elif user_input.lower() == "/tools":
                    tools = system.get_status()['tools_available']
                    print(f"\nAvailable tools ({len(tools)}):")
                    for t in tools:
                        print(f"  - {t}")
                    continue

                elif user_input.lower().startswith("/agent "):
                    agent_type = user_input[7:].strip()
                    if agent_type in system.get_status()['available_agents']:
                        print(f"Agent type set to: {agent_type}")
                        args.agent = agent_type
                    else:
                        print(f"Unknown agent type: {agent_type}")
                    continue

                result = await system.run(user_input, agent_type=args.agent)
                print("\n" + "-" * 40)
                print("Result:")
                print(result.get("output", result))

            except KeyboardInterrupt:
                print("\n\nInterrupted.")
            except Exception as e:
                print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
