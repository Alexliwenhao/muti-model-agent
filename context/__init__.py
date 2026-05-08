"""Context window management inspired by Claude Code"""
from typing import List, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


@dataclass
class MessageToken:
    content: str
    token_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "unknown"


class ContextWindow:
    """Manage context window with auto-compaction support

    Claude Code 使用 200K token 上下文窗口
    在 ~75-92% 容量时自动压缩
    """

    def __init__(self, max_tokens: int = 200000, compact_threshold: float = 0.75):
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.messages: List[BaseMessage] = []
        self.message_tokens: List[MessageToken] = []
        self._total_tokens = 0

    def add_message(self, message: Union[BaseMessage, Dict[str, Any]]):
        if isinstance(message, dict):
            if message.get("role") == "tool":
                msg = HumanMessage(content=f"[Tool Result]: {message.get('content', '')}")
            else:
                msg = HumanMessage(content=str(message))
        else:
            msg = message

        self.messages.append(msg)
        tokens = self._estimate_tokens(msg)
        self.message_tokens.append(MessageToken(
            content=str(msg.content)[:1000],
            token_count=tokens,
            message_type=type(msg).__name__
        ))
        self._total_tokens += tokens

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

    def get_token_count(self) -> int:
        return self._total_tokens

    def should_compact(self) -> bool:
        return self._total_tokens > (self.max_tokens * self.compact_threshold)

    def reset(self):
        preserved = min(4, len(self.messages))
        self.messages = self.messages[-preserved:]
        self.message_tokens = self.message_tokens[-preserved:]
        self._total_tokens = sum(m.token_count for m in self.message_tokens)

    def clear(self):
        self.messages = []
        self.message_tokens = []
        self._total_tokens = 0

    def _estimate_tokens(self, message: BaseMessage) -> int:
        content = str(message.content)
        return len(content) // 4

    def get_context_summary(self) -> Dict[str, Any]:
        return {
            "total_tokens": self._total_tokens,
            "max_tokens": self.max_tokens,
            "usage_ratio": self._total_tokens / self.max_tokens,
            "message_count": len(self.messages),
            "should_compact": self.should_compact()
        }
