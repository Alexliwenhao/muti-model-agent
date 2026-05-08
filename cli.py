#!/usr/bin/env python3
"""CLI for Multi-Model Agent

Inspired by Claude Code's terminal interface
"""
import asyncio
import sys
import os
from typing import Optional
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import MultiModelAgent, AgentStatus
from config import Config


class CLI:
    """Command-line interface for Multi-Model Agent"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.config = Config()
        self.agent = MultiModelAgent(
            config=self.config,
            system_prompt=self._get_system_prompt()
        )
        self.current_session_id = "default"

    def _get_system_prompt(self) -> str:
        return """You are Claude Code, an AI coding assistant.

You have access to tools to interact with the filesystem and run commands:
- Bash(command): Execute shell commands
- Read(file_path): Read file contents
- Edit(file_path, old_str, new_str): Make targeted edits
- Write(file_path, content): Write or overwrite files
- Grep(pattern, path): Search for patterns in files
- Glob(pattern, path): Find files by pattern

Guidelines:
1. Think step by step before taking actions
2. Use tools to read, write, and modify files
3. Verify your changes
4. Be concise and action-oriented
5. If something fails, try an alternative approach

You are in a CLI environment. Work with the user's current directory."""

    async def run(self):
        """Main CLI loop"""
        print("=" * 60)
        print("Multi-Model Agent (Claude Code Architecture)")
        print("=" * 60)
        print(f"Available tools: {', '.join(self.agent.get_available_tools())}")
        print("Type /help for commands, /exit to quit")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["/exit", "/quit", "/q"]:
                    print("Goodbye!")
                    break

                elif user_input.lower() == "/help":
                    self._print_help()

                elif user_input.lower() == "/status":
                    self._print_status()

                elif user_input.lower() == "/tools":
                    self._print_tools()

                elif user_input.lower().startswith("/model "):
                    model = user_input[7:].strip()
                    print(f"Model switching not yet implemented: {model}")

                elif user_input.lower().startswith("/session "):
                    session_id = user_input[9:].strip()
                    print(f"Session management: {session_id}")

                else:
                    await self._execute_task(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Use /exit to quit.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

    async def _execute_task(self, task: str):
        """Execute a task using the agent"""
        print(f"\n[Executing task...]\n")

        if self.verbose:
            print("-" * 40)

        response = await self.agent.run(task)

        if response.status == AgentStatus.COMPLETED:
            print("\n" + "=" * 60)
            print("RESULT:")
            print("=" * 60)
            print(response.message)

        elif response.status == AgentStatus.MAX_ITERATIONS:
            print("\nMaximum iterations reached.")
            print(f"Completed {response.iterations} iterations.")

        elif response.status == AgentStatus.ERROR:
            print(f"\nError: {response.error}")

        if self.verbose and response.tool_calls:
            print("\n" + "-" * 40)
            print(f"Tool calls made: {len(response.tool_calls)}")
            for tc in response.tool_calls[-5:]:
                print(f"  - {tc['name']}")

    def _print_help(self):
        """Print help message"""
        print("""
Available Commands:
  /help           Show this help message
  /exit           Exit the agent
  /status         Show agent status
  /tools          List available tools
  /model <name>   Switch to a different model
  /session <id>   Switch to a different session

Just type your task to start working!
        """)

    def _print_status(self):
        """Print agent status"""
        print(f"""
Agent Status:
  Model: {self.config.get_primary_model().name} ({self.config.get_primary_model().model})
  Session: {self.current_session_id}
  Status: {self.agent.status.value}
  Iterations: {self.agent.iteration_count}
        """)

    def _print_tools(self):
        """Print available tools"""
        tools = self.agent.get_available_tools()
        print(f"\nAvailable tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool}")


async def main():
    parser = argparse.ArgumentParser(description="Multi-Model Agent CLI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--task", "-t", type=str, help="Task to execute (non-interactive)")
    args = parser.parse_args()

    cli = CLI(verbose=args.verbose)

    if args.task:
        await cli._execute_task(args.task)
    else:
        await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
