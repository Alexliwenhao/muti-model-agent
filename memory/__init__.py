"""Memory and Knowledge Graph System

持久化记忆系统，支持跨会话知识存储。

Claude Code Memory 特性:
- 实体和关系追踪
- 跨会话记忆
- 语义搜索
- CLAUDE.md 作为团队记忆
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path


class MemoryType(Enum):
    ENTITY = "entity"
    RELATION = "relation"
    FACTS = "facts"
    PREFERENCES = "preferences"
    CONTEXT = "context"


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.memory_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "access_count": self.access_count
        }


@dataclass
class KnowledgeGraph:
    """知识图谱节点"""
    entity_id: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class MemorySystem:
    """持久化记忆系统

    Claude Code 的记忆系统特点:
    1. 支持多种记忆类型
    2. 持久化存储到文件
    3. 语义搜索能力
    4. CLAUDE.md 集成
    """

    def __init__(self, memory_file: Optional[str] = None):
        self.memory_file = memory_file or ".claude/memory.json"
        self.memories: List[MemoryEntry] = []
        self.knowledge_graph: Dict[str, KnowledgeGraph] = {}
        
        self._ensure_directory()
        self._load()

    def _ensure_directory(self):
        """确保记忆目录存在"""
        path = Path(self.memory_file)
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        """从文件加载记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.memories = [
                    MemoryEntry(
                        id=m["id"],
                        memory_type=MemoryType(m.get("type", "entity")),
                        content=m["content"],
                        metadata=m.get("metadata", {}),
                        created_at=datetime.fromisoformat(m.get("created_at", datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(m.get("updated_at", datetime.now().isoformat())),
                        tags=m.get("tags", []),
                        access_count=m.get("access_count", 0)
                    )
                    for m in data.get("memories", [])
                ]

                self.knowledge_graph = {
                    k: KnowledgeGraph(
                        entity_id=v["entity_id"],
                        entity_type=v["entity_type"],
                        properties=v.get("properties", {}),
                        relations=v.get("relations", [])
                    )
                    for k, v in data.get("knowledge_graph", {}).items()
                }
            except Exception as e:
                print(f"Failed to load memory: {e}")

    def _save(self):
        """保存记忆到文件"""
        data = {
            "memories": [m.to_dict() for m in self.memories],
            "knowledge_graph": {
                k: {
                    "entity_id": v.entity_id,
                    "entity_type": v.entity_type,
                    "properties": v.properties,
                    "relations": v.relations
                }
                for k, v in self.knowledge_graph.items()
            },
            "last_updated": datetime.now().isoformat()
        }

        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.CONTEXT,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加记忆"""
        import uuid
        
        entry = MemoryEntry(
            id=str(uuid.uuid4())[:8],
            memory_type=memory_type,
            content=content,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.memories.append(entry)
        self._save()
        
        return entry.id

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """添加知识图谱实体"""
        self.knowledge_graph[entity_id] = KnowledgeGraph(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=properties or {}
        )
        self._save()

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str
    ):
        """添加实体关系"""
        if from_entity in self.knowledge_graph:
            self.knowledge_graph[from_entity].relations.append({
                "target": to_entity,
                "type": relation_type
            })
            self._save()

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """搜索记忆"""
        query_lower = query.lower()
        results = []
        
        for memory in self.memories:
            if query_lower in memory.content.lower():
                memory.access_count += 1
                results.append(memory)
                
                if len(results) >= limit:
                    break
        
        self._save()
        return results

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """获取最近的记忆"""
        sorted_memories = sorted(
            self.memories,
            key=lambda m: m.updated_at,
            reverse=True
        )
        return sorted_memories[:limit]

    def get_by_tag(self, tag: str) -> List[MemoryEntry]:
        """按标签获取记忆"""
        return [m for m in self.memories if tag in m.tags]

    def get_context_for_task(self, task: str) -> str:
        """获取与任务相关的上下文"""
        relevant = self.search(task, limit=5)
        
        if not relevant:
            return ""
        
        context = "Relevant information from memory:\n"
        for m in relevant:
            context += f"\n[{m.memory_type.value}] {m.content}\n"
        
        return context

    def update_memory(self, memory_id: str, content: str) -> bool:
        """更新记忆"""
        for memory in self.memories:
            if memory.id == memory_id:
                memory.content = content
                memory.updated_at = datetime.now()
                self._save()
                return True
        return False

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                del self.memories[i]
                self._save()
                return True
        return False

    def clear_all(self):
        """清除所有记忆"""
        self.memories = []
        self.knowledge_graph = {}
        self._save()

    def get_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        return {
            "total_memories": len(self.memories),
            "knowledge_graph_entities": len(self.knowledge_graph),
            "by_type": {
                m.memory_type.value: len([x for x in self.memories if x.memory_type == m.memory_type])
                for m in self.memories
            },
            "recent_memories": len(self.get_recent(5)),
            "most_accessed": max(
                (m.access_count for m in self.memories),
                default=0
            )
        }


class ClaudeMDManager:
    """CLAUDE.md 文件管理器

    Claude Code 最佳实践:
    - CLAUDE.md 作为团队记忆
    - 更新频率: 多周更新一次
    - 记录错误和经验教训
    - 保留 ~150 条指令上限
    """

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.claude_md = self.project_path / "CLAUDE.md"

    def exists(self) -> bool:
        """检查 CLAUDE.md 是否存在"""
        return self.claude_md.exists()

    def read(self) -> str:
        """读取 CLAUDE.md 内容"""
        if self.exists():
            with open(self.claude_md, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def write(self, content: str):
        """写入 CLAUDE.md"""
        with open(self.claude_md, 'w', encoding='utf-8') as f:
            f.write(content)

    def update_with_memory(
        self,
        memory_system: MemorySystem,
        lessons_learned: Optional[List[str]] = None
    ):
        """根据记忆更新 CLAUDE.md"""
        content = self.read() or "# CLAUDE.md\n\nThis file contains instructions for Claude Code.\n"

        sections = {
            "project_overview": "## Project Overview\n",
            "coding_standards": "## Coding Standards\n",
            "common_patterns": "## Common Patterns\n",
            "lessons_learned": "## Lessons Learned\n",
            "tools_available": "## Available Tools\n"
        }

        summary = memory_system.get_summary()
        
        if lessons_learned:
            content += "\n\n" + sections["lessons_learned"]
            for lesson in lessons_learned:
                content += f"- {lesson}\n"

        self.write(content)

    def create_default(self):
        """创建默认的 CLAUDE.md"""
        default_content = """# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) to help you work more effectively in this repository.

## Project Overview
[Describe your project here]

## Coding Standards
- Follow existing code style
- Write clear, descriptive comments
- Include type hints where appropriate
- Test all changes

## Common Patterns
- [Document common patterns used in this codebase]

## Available Tools
- Bash: Execute commands
- Read: Read files
- Write: Write files
- Edit: Edit files
- Grep: Search code
- Glob: Find files

## Guidelines
1. Think step by step before taking actions
2. Verify changes before committing
3. Update this file when learning new patterns
"""
        self.write(default_content)
