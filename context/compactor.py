"""Context compaction for long sessions

Based on Claude Code compaction patterns:
- Token budget management
- Smart message summarization
- Preserve critical context (task, file changes)
- Snip non-essential messages
"""
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


@dataclass
class CompactionConfig:
    """Configuration for context compaction"""
    max_tokens: int = 200000
    compact_threshold: float = 0.75  # Start compaction at 75%
    preserve_recent: int = 4  # Always keep last N messages
    preserve_system: bool = True
    preserve_task: bool = True
    preserve_tools: bool = False  # Only keep tool results that were successful
    summarization_fn: Optional[Callable[[List[BaseMessage]], str]] = None


@dataclass
class CompactionResult:
    """Result of a compaction operation"""
    original_tokens: int
    compacted_tokens: int
    messages_removed: int
    preserved_messages: List[BaseMessage]
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)


class MessageImportance:
    """Determine importance of different message types"""
    
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    MINIMAL = 10

    @classmethod
    def get_importance(cls, message: BaseMessage, is_recent: bool = False) -> Tuple[int, str]:
        """Get importance score for a message"""
        if isinstance(message, SystemMessage):
            return cls.HIGH, "system"
        
        if is_recent:
            return cls.CRITICAL, "recent"
        
        content = str(message.content).lower()
        
        if "error" in content or "failed" in content:
            return cls.MEDIUM, "error_message"
        
        if "commit" in content or "result" in content:
            return cls.MEDIUM, "result_message"
        
        if "task" in content and "notification" in content:
            return cls.MEDIUM, "task_notification"
        
        if len(content) < 50:
            return cls.LOW, "short_message"
        
        return cls.LOW, "general"


class ContextCompactor:
    """Context compaction system inspired by Claude Code

    Claude Code compaction features:
    - Token budget tracking
    - Message importance scoring
    - Smart summarization
    - Preserve critical context
    - XML-format summary output
    """

    def __init__(self, config: Optional[CompactionConfig] = None):
        self.config = config or CompactionConfig()

    def calculate_tokens(self, messages: List[BaseMessage]) -> int:
        """Calculate total tokens in messages"""
        total = 0
        for msg in messages:
            content = str(msg.content)
            total += len(content) // 4
        return total

    def should_compact(self, messages: List[BaseMessage]) -> bool:
        """Check if compaction is needed"""
        total = self.calculate_tokens(messages)
        return total > (self.config.max_tokens * self.config.compact_threshold)

    def get_compaction_strategy(self, messages: List[BaseMessage]) -> str:
        """Determine best compaction strategy based on context"""
        total = self.calculate_tokens(messages)
        usage_ratio = total / self.config.max_tokens

        if usage_ratio < 0.5:
            return "none"
        elif usage_ratio < 0.75:
            return "light"
        elif usage_ratio < 0.9:
            return "moderate"
        else:
            return "aggressive"

    def compact(
        self,
        messages: List[BaseMessage],
        task_description: Optional[str] = None
    ) -> CompactionResult:
        """Compact messages to fit within token budget"""
        original_tokens = self.calculate_tokens(messages)
        
        if not self.should_compact(messages):
            return CompactionResult(
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                messages_removed=0,
                preserved_messages=messages,
                summary=""
            )

        strategy = self.get_compaction_strategy(messages)
        
        if strategy == "none":
            return CompactionResult(
                original_tokens=original_tokens,
                compacted_tokens=original_tokens,
                messages_removed=0,
                preserved_messages=messages,
                summary=""
            )

        prioritized = self._prioritize_messages(messages)
        
        preserved = []
        removed_count = 0

        for i, (msg, importance, reason) in enumerate(prioritized):
            is_recent = i >= len(prioritized) - self.config.preserve_recent
            
            if self.config.preserve_system and isinstance(msg, SystemMessage):
                preserved.append(msg)
                continue
            
            if is_recent:
                preserved.append(msg)
                continue
            
            msg_tokens = len(str(msg.content)) // 4
            current_total = self.calculate_tokens(preserved) + msg_tokens
            
            if current_total < self.config.max_tokens * 0.9:
                preserved.append(msg)
            else:
                removed_count += 1

        summary = self._generate_summary(messages, preserved, task_description)

        compacted_tokens = self.calculate_tokens(preserved)

        return CompactionResult(
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            messages_removed=removed_count,
            preserved_messages=preserved,
            summary=summary
        )

    def _prioritize_messages(
        self,
        messages: List[BaseMessage]
    ) -> List[Tuple[BaseMessage, int, str]]:
        """Prioritize messages by importance"""
        prioritized = []
        
        for i, msg in enumerate(messages):
            importance, reason = MessageImportance.get_importance(
                msg,
                is_recent=(i >= len(messages) - self.config.preserve_recent)
            )
            prioritized.append((msg, importance, reason))
        
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized

    def _generate_summary(
        self,
        original: List[BaseMessage],
        compacted: List[BaseMessage],
        task_description: Optional[str] = None
    ) -> str:
        """Generate summary of compacted messages (Claude Code XML format)"""
        removed = len(original) - len(compacted)
        
        summary_lines = [
            "<compact_summary>",
            f"  <removed_messages>{removed}</removed_messages>",
            f"  <preserved_messages>{len(compacted)}</preserved_messages>"
        ]
        
        if task_description:
            summary_lines.insert(0, f"  <task>{task_description}</task>")
        
        action_types = self._categorize_messages(original)
        for action_type, count in action_types.items():
            summary_lines.append(f"  <{action_type}>{count}</{action_type}>")
        
        if self.config.summarization_fn:
            summary_text = self.config.summarization_fn(compacted)
            summary_lines.append(f"  <summary>{summary_text}</summary>")
        
        summary_lines.append("</compact_summary>")
        
        return "\n".join(summary_lines)

    def _categorize_messages(self, messages: List[BaseMessage]) -> Dict[str, int]:
        """Categorize messages by type"""
        categories = {
            "tool_results": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "system_messages": 0
        }
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                categories["system_messages"] += 1
            elif hasattr(msg, "type"):
                if "tool" in str(msg.type).lower():
                    categories["tool_results"] += 1
                elif "human" in str(msg.type).lower():
                    categories["user_messages"] += 1
                else:
                    categories["assistant_messages"] += 1
            else:
                content = str(msg.content).lower()
                if "[tool result]" in content:
                    categories["tool_results"] += 1
                elif "user" in str(type(msg).__name__).lower():
                    categories["user_messages"] += 1
                else:
                    categories["assistant_messages"] += 1
        
        return categories

    def get_token_budget_status(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Get current token budget status"""
        total = self.calculate_tokens(messages)
        usage_ratio = total / self.config.max_tokens
        
        return {
            "total_tokens": total,
            "max_tokens": self.config.max_tokens,
            "usage_ratio": usage_ratio,
            "remaining_tokens": self.config.max_tokens - total,
            "should_compact": self.should_compact(messages),
            "strategy": self.get_compaction_strategy(messages),
            "message_count": len(messages)
        }
