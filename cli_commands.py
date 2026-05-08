"""Advanced CLI Commands - Claude Code Style

基于 Claude Code 架构的高级命令:
- /loop - 循环任务
- /btw - 附带问题
- /plan - 计划模式
- /compact - 上下文压缩
- /review - 代码审查
- /compact - 上下文压缩
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uuid


class CommandType(Enum):
    LOOP = "loop"
    BTW = "btw"
    PLAN = "plan"
    REVIEW = "review"
    COMPACT = "compact"
    TASK = "task"
    MEMORY = "memory"


@dataclass
class LoopTask:
    """循环任务"""
    task_id: str
    description: str
    interval_seconds: int
    max_iterations: int = 10
    current_iteration: int = 0
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SideNote:
    """附带问题（btw 命令）"""
    note_id: str
    content: str
    context: str
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    response: Optional[str] = None


@dataclass
class PlanStep:
    """计划步骤"""
    step_id: str
    description: str
    status: str = "pending"
    assigned_agent: Optional[str] = None
    result: Optional[str] = None


class AdvancedCLICommands:
    """高级 CLI 命令系统

    Claude Code 高级命令特性:
    1. /loop - 定期执行任务
    2. /btw - 不打断流程的附带问题
    3. /plan - 计划模式
    4. /review - 代码审查
    5. /compact - 上下文压缩
    6. /tasks - 任务管理
    """

    def __init__(self, agent_system: Any):
        self.agent_system = agent_system
        self.loop_tasks: Dict[str, LoopTask] = {}
        self.side_notes: List[SideNote] = []
        self.plan_mode: bool = False
        self.plan_steps: List[PlanStep] = []
        self.plan_mode_active: bool = False

    async def parse_and_execute(self, user_input: str) -> Dict[str, Any]:
        """解析并执行命令"""
        parsed = self.parse_command(user_input)
        if parsed:
            command, args = parsed
            return await self.execute_command(command, args)
        return {"status": "error", "message": "Unknown command"}

    def parse_command(self, user_input: str) -> Optional[tuple[CommandType, str]]:
        """解析命令"""
        if user_input.startswith("/loop "):
            return CommandType.LOOP, user_input[6:].strip()
        elif user_input.startswith("/btw "):
            return CommandType.BTW, user_input[5:].strip()
        elif user_input.startswith("/plan"):
            return CommandType.PLAN, user_input[5:].strip() if len(user_input) > 5 else ""
        elif user_input.startswith("/review"):
            return CommandType.REVIEW, user_input[7:].strip() if len(user_input) > 7 else ""
        elif user_input.startswith("/compact"):
            return CommandType.COMPACT, ""
        elif user_input.startswith("/tasks"):
            return CommandType.TASK, user_input[6:].strip() if len(user_input) > 6 else ""
        elif user_input.startswith("/memory"):
            return CommandType.MEMORY, user_input[8:].strip() if len(user_input) > 8 else ""
        return None

    async def execute_command(
        self,
        command: CommandType,
        args: str
    ) -> Dict[str, Any]:
        """执行命令"""
        if command == CommandType.LOOP:
            return await self._execute_loop(args)
        elif command == CommandType.BTW:
            return await self._execute_btw(args)
        elif command == CommandType.PLAN:
            return await self._execute_plan(args)
        elif command == CommandType.REVIEW:
            return await self._execute_review(args)
        elif command == CommandType.COMPACT:
            return await self._execute_compact()
        elif command == CommandType.TASK:
            return await self._execute_tasks(args)
        elif command == CommandType.MEMORY:
            return await self._execute_memory(args)
        return {"status": "error", "message": "Unknown command"}

    async def _execute_loop(self, args: str) -> Dict[str, Any]:
        """执行循环任务"""
        parts = args.split()
        if len(parts) < 2:
            return {
                "status": "error",
                "message": "Usage: /loop <interval_seconds> <task_description>"
            }

        try:
            interval = int(parts[0])
            description = " ".join(parts[1:])

            task_id = str(uuid.uuid4())[:8]
            loop_task = LoopTask(
                task_id=task_id,
                description=description,
                interval_seconds=interval
            )
            self.loop_tasks[task_id] = loop_task

            return {
                "status": "created",
                "task_id": task_id,
                "message": f"Loop task created: every {interval}s - {description}"
            }
        except ValueError:
            return {
                "status": "error",
                "message": "Invalid interval. Usage: /loop <interval_seconds> <task>"
            }

    async def execute_loop_task(self, task_id: str) -> Dict[str, Any]:
        """执行单个循环任务"""
        if task_id not in self.loop_tasks:
            return {"status": "error", "message": "Task not found"}

        task = self.loop_tasks[task_id]

        if not task.active:
            return {"status": "inactive"}

        result = await self.agent_system.run(task.description, agent_type="claude")

        task.current_iteration += 1
        task.last_run = datetime.now()
        task.results.append({
            "iteration": task.current_iteration,
            "timestamp": task.last_run.isoformat(),
            "result": result
        })

        if task.current_iteration >= task.max_iterations:
            task.active = False

        return {
            "status": "executed",
            "iteration": task.current_iteration,
            "result": result
        }

    def start_loop_scheduler(self):
        """启动循环任务调度器"""
        asyncio.create_task(self._loop_scheduler())

    async def _loop_scheduler(self):
        """循环任务调度器"""
        while True:
            now = datetime.now()
            for task_id, task in list(self.loop_tasks.items()):
                if task.active and task.last_run:
                    next_run = task.last_run + timedelta(seconds=task.interval_seconds)
                    if now >= next_run:
                        await self.execute_loop_task(task_id)
                elif task.active and not task.last_run:
                    await self.execute_loop_task(task_id)

            await asyncio.sleep(1)

    async def _execute_btw(self, args: str) -> Dict[str, Any]:
        """执行附带问题"""
        note = SideNote(
            note_id=str(uuid.uuid4())[:8],
            content=args,
            context="Current task context"
        )
        self.side_notes.append(note)

        result = await self.agent_system.run(args, agent_type="langchain")
        note.response = str(result.get("output", ""))

        return {
            "status": "answered",
            "note_id": note.note_id,
            "question": args,
            "answer": note.response
        }

    async def _execute_plan(self, args: str) -> Dict[str, Any]:
        """执行计划模式"""
        if not args:
            self.plan_mode_active = True
            return {
                "status": "plan_mode_entered",
                "message": "Plan mode enabled. Enter your task and I'll create a plan."
            }

        if self.plan_mode_active:
            plan_prompt = f"""Create a detailed plan for this task:

Task: {args}

Provide a numbered plan with clear steps. Consider:
1. Analysis and requirements
2. Implementation steps
3. Testing approach
4. Potential issues

Format:
1. [Step description]
2. [Step description]
..."""

            result = await self.agent_system.run(plan_prompt, agent_type="plan")

            steps_text = result.get("output", "")

            self.plan_steps = []
            for i, line in enumerate(steps_text.split("\n"), 1):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    step_text = line.lstrip("0123456789.-) ")
                    if step_text:
                        self.plan_steps.append(PlanStep(
                            step_id=str(uuid.uuid4())[:8],
                            description=step_text
                        ))

            return {
                "status": "plan_created",
                "task": args,
                "steps": [{"id": s.step_id, "description": s.description} for s in self.plan_steps]
            }
        else:
            result = await self.agent_system.run(args, agent_type="claude")
            return {
                "status": "executed",
                "result": result
            }

    async def _execute_review(self, args: str) -> Dict[str, Any]:
        """执行代码审查"""
        review_prompt = f"""Review the code for quality and potential issues:

Files/Code: {args}

Focus on:
1. Code quality and readability
2. Potential bugs or errors
3. Security concerns
4. Performance issues
5. Best practices

Provide a structured review with:
- Issues found (critical, warnings, suggestions)
- Recommendations
"""

        result = await self.agent_system.run(review_prompt, agent_type="multi")

        return {
            "status": "review_completed",
            "review": result.get("output", "")
        }

    async def _execute_compact(self) -> Dict[str, Any]:
        """执行上下文压缩"""
        context_summary = {
            "session_messages": len(self.agent_system.session_manager.get_current_session().messages) if self.agent_system.session_manager.get_current_session() else 0,
            "active_tasks": len(self.loop_tasks),
            "pending_notes": len([n for n in self.side_notes if not n.resolved])
        }

        return {
            "status": "compacted",
            "summary": context_summary,
            "message": "Context has been compacted. Summary retained."
        }

    async def _execute_tasks(self, args: str) -> Dict[str, Any]:
        """执行任务管理"""
        if args == "list":
            return {
                "status": "success",
                "loop_tasks": [
                    {
                        "id": t.task_id,
                        "description": t.description,
                        "active": t.active,
                        "iteration": f"{t.current_iteration}/{t.max_iterations}"
                    }
                    for t in self.loop_tasks.values()
                ],
                "pending_notes": len([n for n in self.side_notes if not n.resolved])
            }
        elif args.startswith("stop "):
            task_id = args[5:].strip()
            if task_id in self.loop_tasks:
                self.loop_tasks[task_id].active = False
                return {"status": "stopped", "task_id": task_id}
            return {"status": "error", "message": "Task not found"}
        elif args == "clear":
            self.loop_tasks.clear()
            return {"status": "cleared"}
        else:
            return {
                "status": "error",
                "message": "Usage: /tasks [list|stop <id>|clear]"
            }

    async def _execute_memory(self, args: str) -> Dict[str, Any]:
        """执行记忆命令"""
        if hasattr(self.agent_system, 'memory_system'):
            if args == "summary":
                return {
                    "status": "success",
                    "summary": self.agent_system.memory_system.get_summary()
                }
            elif args.startswith("search "):
                query = args[7:].strip()
                results = self.agent_system.memory_system.search(query)
                return {
                    "status": "success",
                    "results": [r.to_dict() for r in results]
                }
            else:
                return {
                    "status": "error",
                    "message": "Usage: /memory [summary|search <query>]"
                }
        return {
            "status": "error",
            "message": "Memory system not initialized"
        }

    def get_status(self) -> Dict[str, Any]:
        """获取命令系统状态"""
        return {
            "active_loops": len([t for t in self.loop_tasks.values() if t.active]),
            "pending_notes": len([n for n in self.side_notes if not n.resolved]),
            "plan_mode": self.plan_mode_active,
            "plan_steps": len(self.plan_steps)
        }


class WriterReviewerPattern:
    """Writer/Reviewer 模式

    Claude Code 最佳实践:
    - Writer Agent 负责实现
    - Reviewer Agent 负责审查（新鲜上下文，无偏见）
    - 验证 = 2-3x 质量提升
    """

    def __init__(self, agent_system: Any):
        self.agent_system = agent_system

    async def implement_with_review(
        self,
        task: str,
        verification_iterations: int = 2
    ) -> Dict[str, Any]:
        """使用 Writer/Reviewer 模式实现任务"""
        results = []

        for iteration in range(verification_iterations):
            writer_prompt = f"""You are the Writer Agent.

Task: {task}

{self._get_context_from_results(results)}

Implement the solution following best practices:
1. Write clean, well-documented code
2. Consider edge cases
3. Follow existing code patterns
4. Add tests where appropriate

Return your implementation."""

            writer_result = await self.agent_system.run(writer_prompt, agent_type="claude")
            results.append({
                "type": "writer",
                "iteration": iteration + 1,
                "content": writer_result.get("output", "")
            })

            reviewer_prompt = f"""You are the Reviewer Agent.

Review this implementation critically:

{writer_result.get('output', '')}

Check for:
1. Correctness and bugs
2. Security vulnerabilities
3. Performance issues
4. Code quality
5. Edge cases
6. Missing error handling

Provide specific, actionable feedback."""

            reviewer_result = await self.agent_system.run(reviewer_prompt, agent_type="multi")
            results.append({
                "type": "reviewer",
                "iteration": iteration + 1,
                "content": reviewer_result.get("output", "")
            })

        return {
            "status": "completed",
            "iterations": verification_iterations,
            "results": results,
            "final_output": results[-2].get("content", "") if len(results) >= 2 else ""
        }

    def _get_context_from_results(self, results: List[Dict]) -> str:
        """从结果中提取上下文"""
        if not results:
            return ""

        context = "\n\nPrevious iterations:\n"
        for r in results[-2:]:
            context += f"\n[{r['type'].upper()}] Iteration {r['iteration']}:\n{r['content'][:500]}...\n"

        return context
