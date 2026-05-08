#!/usr/bin/env python3
"""Complete CLI for Multi-Model Agent with Collaboration

整合所有功能的完整命令行界面
"""
import asyncio
import sys
import os
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from enhanced_agent import MultiModelAgentSystem
from collaboration import CollaborationEngine, CollaborationMode


class CollaborationCLI:
    """Complete CLI with multi-model collaboration support"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.config = Config()
        self.agent_system = MultiModelAgentSystem(
            config=self.config,
            verbose=verbose
        )
        self.collaboration_engine = CollaborationEngine(self.config)

    async def run_collaborative(
        self,
        task: str,
        mode: str = "parallel"
    ):
        """Run task with multi-model collaboration"""
        mode_map = {
            "parallel": CollaborationMode.PARALLEL,
            "sequential": CollaborationMode.SEQUENTIAL,
            "hierarchical": CollaborationMode.HIERARCHICAL,
            "debate": CollaborationMode.DEBATE,
        }

        collab_mode = mode_map.get(mode, CollaborationMode.PARALLEL)

        print(f"\n[Running {mode} collaboration]")
        print("-" * 40)

        result = await self.collaboration_engine.execute(
            task,
            mode=collab_mode
        )

        print(f"\n{'='*60}")
        print(f"COLLABORATION RESULT ({mode})")
        print(f"{'='*60}")
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Models used: {len(set(r.model_name for r in result.model_responses))}")

        if result.synthesis:
            print(f"\n{'='*60}")
            print("SYNTHESIZED ANSWER:")
            print("="*60)
            print(result.synthesis)

        print(f"\n{'='*60}")
        print("INDIVIDUAL MODEL RESPONSES:")
        print("="*60)

        for i, response in enumerate(result.model_responses, 1):
            print(f"\n[{i}] {response.model_name} (confidence: {response.confidence:.2f})")
            print("-" * 40)
            content = response.content[:500] + "..." if len(response.content) > 500 else response.content
            print(content)
            if response.error:
                print(f"Error: {response.error}")

        return result

    async def run_agent(
        self,
        task: str,
        agent_type: str = "claude"
    ):
        """Run task using specific agent"""
        print(f"\n[Running with {agent_type} agent]")

        result = await self.agent_system.run(
            task,
            agent_type=agent_type,
            sub_agents=True
        )

        print(f"\n{'='*60}")
        print(f"RESULT ({agent_type})")
        print("="*60)
        print(result.get("output", result))

        return result

    def print_status(self):
        """Print system status"""
        print(f"\n{'='*60}")
        print("SYSTEM STATUS")
        print("="*60)

        agent_status = self.agent_system.get_status()
        collab_status = self.collaboration_engine.get_model_status()

        print("\nAgent System:")
        print(f"  LangChain enabled: {agent_status['langchain_enabled']}")
        print(f"  Active sub-agents: {agent_status['sub_agents']['active']}")
        print(f"  Completed sub-agents: {agent_status['sub_agents']['completed']}")
        print(f"  Available agents: {', '.join(agent_status['available_agents'])}")

        print("\nCollaboration Engine:")
        print(f"  Total models: {collab_status['total_models']}")
        for name, info in collab_status['models'].items():
            print(f"    - {name}: {info['provider']} ({', '.join(info['strengths'])})")

        print("\nAvailable Tools:")
        for tool in agent_status['tools_available']:
            print(f"  - {tool}")


async def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Agent CLI with Collaboration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Write a hello world program"
  python main.py --collab --mode parallel "Analyze this codebase"
  python main.py --agent langchain "Plan a web app"
  python main.py --collab --mode hierarchical "Design a database schema"

Modes:
  parallel     - Multiple models work simultaneously
  sequential   - Models work in sequence
  hierarchical - Multi-level refinement
  debate       - Models debate and synthesize
        """
    )

    parser.add_argument("task", nargs="?", help="Task to execute")
    parser.add_argument("--agent", "-a", default="claude",
                       choices=["claude", "langchain", "plan", "conversational", "multi"],
                       help="Agent type to use")
    parser.add_argument("--collab", "-c", action="store_true",
                       help="Use multi-model collaboration")
    parser.add_argument("--mode", "-m", default="parallel",
                       choices=["parallel", "sequential", "hierarchical", "debate"],
                       help="Collaboration mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--status", "-s", action="store_true",
                       help="Show system status and exit")

    args = parser.parse_args()

    cli = CollaborationCLI(verbose=args.verbose)

    if args.status:
        cli.print_status()
        return

    print("=" * 60)
    print("Multi-Model Agent with Claude Code + LangChain")
    print("=" * 60)

    if args.task:
        if args.collab:
            await cli.run_collaborative(args.task, args.mode)
        else:
            await cli.run_agent(args.task, args.agent)
    else:
        cli.print_status()
        print("\nEnter interactive mode...")

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
  /status        System status
  /collab        Enable collaboration mode
  /mode <type>   Set collaboration mode
  /agent <type>  Set agent type

Collaboration modes: parallel, sequential, hierarchical, debate
Agent types: claude, langchain, plan, conversational, multi
                    """)
                    continue

                elif user_input.lower() == "/status":
                    cli.print_status()
                    continue

                elif user_input.lower() == "/collab":
                    args.collab = True
                    print("Collaboration mode enabled")
                    continue

                elif user_input.lower().startswith("/mode "):
                    mode = user_input[6:].strip()
                    if mode in ["parallel", "sequential", "hierarchical", "debate"]:
                        args.mode = mode
                        print(f"Mode set to: {mode}")
                    continue

                elif user_input.lower().startswith("/agent "):
                    agent = user_input[7:].strip()
                    if agent in ["claude", "langchain", "plan", "conversational", "multi"]:
                        args.agent = agent
                        print(f"Agent set to: {agent}")
                    continue

                if args.collab:
                    await cli.run_collaborative(user_input, args.mode)
                else:
                    await cli.run_agent(user_input, args.agent)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Use /exit to quit.")
            except EOFError:
                break
            except Exception as e:
                print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
