"""Session management for Multi-Model Agent"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from core.agent import AgentResponse


class SessionStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class SessionCheckpoint:
    checkpoint_id: str
    timestamp: datetime
    messages: List[Any]
    task_state: Dict[str, Any]
    description: str = ""


@dataclass
class Session:
    """Represents a conversation session with the agent

    Claude Code 的 Session 概念:
    - 支持分支切换
    - 支持恢复和 fork
    - Checkpoint 用于安全回滚
    """

    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.ACTIVE
    messages: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[SessionCheckpoint] = field(default_factory=list)
    current_task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()

    def create_checkpoint(self, task_state: Dict[str, Any], description: str = ""):
        checkpoint = SessionCheckpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            messages=self.messages.copy(),
            task_state=task_state.copy(),
            description=description
        )
        self.checkpoints.append(checkpoint)
        return checkpoint.checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                self.messages = cp.messages.copy()
                self.metadata.update(cp.task_state)
                self.updated_at = datetime.now()
                return True
        return False

    def fork(self, new_session_id: Optional[str] = None) -> 'Session':
        """Create a forked session"""
        new_id = new_session_id or str(uuid.uuid4())[:8]
        new_session = Session(
            session_id=new_id,
            messages=self.messages.copy(),
            current_task=self.current_task,
            metadata=self.metadata.copy()
        )
        return new_session

    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "message_count": len(self.messages),
            "checkpoint_count": len(self.checkpoints),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_task": self.current_task
        }


class SessionManager:
    """Manages multiple agent sessions"""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._current_session: Optional[str] = None

    def create_session(self, session_id: Optional[str] = None) -> Session:
        new_id = session_id or str(uuid.uuid4())[:8]
        session = Session(session_id=new_id)
        self._sessions[new_id] = session
        self._current_session = new_id
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_current_session(self) -> Optional[Session]:
        if self._current_session:
            return self._sessions.get(self._current_session)
        return None

    def switch_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._current_session = session_id
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [s.get_summary() for s in self._sessions.values()]

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            if self._current_session == session_id:
                self._current_session = None
            return True
        return False
