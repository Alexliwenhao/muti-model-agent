"""Team-based Multi-Agent Coordination System

Based on Claude Code Coordinator Mode Architecture:
- Coordinator orchestrates multiple workers
- Workers execute tasks autonomously
- Parallel execution with result synthesis
- Task notification system for async results
"""
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid

from agents import SubAgentManager, SubAgentConfig, SubAgentType, SubAgentResult


class WorkerStatus(Enum):
    """Worker agent status"""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TaskNotification:
    """Task notification from worker to coordinator (Claude Code XML format)"""
    task_id: str
    status: str
    summary: str
    result: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_xml(self) -> str:
        """Convert to Claude Code XML format"""
        xml = f"""<task-notification>
<task-id>{self.task_id}</task-id>
<status>{self.status}</status>
<summary>{self.summary}</summary>"""
        if self.result:
            xml += f"\n<result>{self.result}</result>"
        if self.usage:
            xml += f"""
<usage>
  <total_tokens>{self.usage.get('total_tokens', 0)}</total_tokens>
  <tool_uses>{self.usage.get('tool_uses', 0)}</tool_uses>
  <duration_ms>{self.usage.get('duration_ms', 0)}</duration_ms>
</usage>"""
        xml += "\n</task-notification>"
        return xml

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "result": self.result,
            "usage": self.usage,
            "error": self.error
        }


@dataclass
class Worker:
    """Worker agent that executes tasks"""
    worker_id: str
    description: str
    status: WorkerStatus = WorkerStatus.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_notification(self) -> TaskNotification:
        return TaskNotification(
            task_id=self.worker_id,
            status=self.status.value,
            summary=f"Worker '{self.description}' {self.status.value}",
            result=self.result,
            usage=self.usage,
            error=self.error
        )


class Coordinator:
    """Multi-Agent Coordinator

    Claude Code Coordinator Pattern:
    1. Coordinator (main agent) orchestrates workers
    2. Workers execute tasks in parallel
    3. Results arrive as task-notification messages
    4. Coordinator synthesizes and reports to user
    """

    def __init__(
        self,
        sub_agent_manager: SubAgentManager,
        llm_factory: Optional[Callable] = None
    ):
        self.sub_agent_manager = sub_agent_manager
        self.llm_factory = llm_factory
        self.workers: Dict[str, Worker] = {}
        self.completed_notifications: List[TaskNotification] = []
        self.on_notification: Optional[Callable[[TaskNotification], None]] = None

    def create_worker(
        self,
        description: str,
        subagent_type: SubAgentType = SubAgentType.GENERAL,
        tools: Optional[List[str]] = None
    ) -> Worker:
        """Create a new worker"""
        worker_id = f"agent-{str(uuid.uuid4())[:6]}"

        config = SubAgentConfig(
            name=description,
            description=description,
            agent_type=subagent_type,
            tools=tools or ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
        )

        agent = self.sub_agent_manager.create_sub_agent(config)

        worker = Worker(
            worker_id=worker_id,
            description=description
        )
        self.workers[worker_id] = worker

        return worker

    async def launch_worker(
        self,
        worker_id: str,
        prompt: str,
        continue_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Launch a worker with a task"""
        if worker_id not in self.workers:
            return {"error": f"Worker {worker_id} not found"}

        worker = self.workers[worker_id]
        worker.status = WorkerStatus.WORKING
        worker.started_at = datetime.now()

        try:
            agent = self.sub_agent_manager.get_agent(worker_id)
            if not agent:
                config = SubAgentConfig(
                    name=worker.description,
                    description=worker.description,
                    agent_type=SubAgentType.GENERAL
                )
                agent = self.sub_agent_manager.create_sub_agent(config, agent_config=config)

            if self.llm_factory:
                llm = self.llm_factory()

            result = await agent.execute(prompt, llm)
            self.sub_agent_manager.complete_agent(worker_id, result)

            worker.status = WorkerStatus.COMPLETED
            worker.completed_at = datetime.now()
            worker.result = result.summary
            worker.usage = {
                "total_tokens": 0,
                "tool_uses": 0,
                "duration_ms": result.execution_time * 1000
            }

            notification = worker.to_notification()
            self.completed_notifications.append(notification)

            if self.on_notification:
                self.on_notification(notification)

            return {
                "task_id": worker_id,
                "status": "launched",
                "notification": notification.to_xml()
            }

        except Exception as e:
            worker.status = WorkerStatus.FAILED
            worker.error = str(e)
            worker.completed_at = datetime.now()

            notification = worker.to_notification()
            self.completed_notifications.append(notification)

            return {
                "task_id": worker_id,
                "status": "error",
                "error": str(e)
            }

    async def launch_workers_parallel(
        self,
        tasks: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Launch multiple workers in parallel (fanning out)"""
        results = []

        for task in tasks:
            worker = self.create_worker(
                description=task.get("description", "Worker"),
                subagent_type=SubAgentType.GENERAL
            )

            result = await self.launch_worker(
                worker.worker_id,
                task.get("prompt", "")
            )
            results.append(result)

        return results

    def send_message(
        self,
        worker_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Continue a worker with a message (Claude Code pattern)"""
        if worker_id not in self.workers:
            return {"error": f"Worker {worker_id} not found"}

        worker = self.workers[worker_id]
        return {
            "status": "message_queued",
            "to": worker_id,
            "message": message
        }

    def stop_worker(self, worker_id: str) -> Dict[str, Any]:
        """Stop a running worker"""
        if worker_id not in self.workers:
            return {"error": f"Worker {worker_id} not found"}

        worker = self.workers[worker_id]
        worker.status = WorkerStatus.STOPPED
        worker.completed_at = datetime.now()

        return {
            "status": "stopped",
            "task_id": worker_id
        }

    def get_pending_notifications(self) -> List[TaskNotification]:
        """Get all completed task notifications"""
        return self.completed_notifications.copy()

    def clear_notifications(self):
        """Clear completed notifications"""
        self.completed_notifications.clear()

    def get_worker_status(self, worker_id: str) -> Optional[WorkerStatus]:
        """Get status of a specific worker"""
        if worker_id in self.workers:
            return self.workers[worker_id].status
        return None

    def get_all_workers(self) -> List[Dict[str, Any]]:
        """Get all workers and their status"""
        return [
            {
                "worker_id": w.worker_id,
                "description": w.description,
                "status": w.status.value,
                "created_at": w.created_at.isoformat(),
                "started_at": w.started_at.isoformat() if w.started_at else None,
                "completed_at": w.completed_at.isoformat() if w.completed_at else None
            }
            for w in self.workers.values()
        ]


class TeamManager:
    """Manages multiple teams of agents"""

    def __init__(self):
        self.teams: Dict[str, Coordinator] = {}
        self.default_team: Optional[Coordinator] = None

    def create_team(
        self,
        team_name: str,
        sub_agent_manager: SubAgentManager,
        llm_factory: Optional[Callable] = None
    ) -> Coordinator:
        """Create a new team"""
        coordinator = Coordinator(sub_agent_manager, llm_factory)
        self.teams[team_name] = coordinator
        return coordinator

    def get_team(self, team_name: str) -> Optional[Coordinator]:
        """Get a team by name"""
        return self.teams.get(team_name)

    def delete_team(self, team_name: str) -> bool:
        """Delete a team"""
        if team_name in self.teams:
            del self.teams[team_name]
            return True
        return False

    def get_all_teams(self) -> List[str]:
        """Get all team names"""
        return list(self.teams.keys())
