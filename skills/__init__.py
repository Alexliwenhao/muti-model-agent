"""Skills System - Reusable Workflows

Claude Code Skills:
- Reusable workflows that can be invoked via Skill tool
- Defined as markdown files with metadata
- Can be shared across projects
- Executed through SkillTool
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os
from pathlib import Path
import re

from langchain_core.tools import BaseTool
from pydantic import BaseModel


class SkillCategory(Enum):
    """Skill categories"""
    CODE = "code"
    ANALYSIS = "analysis"
    DEVOPS = "devops"
    RESEARCH = "research"
    UTILITY = "utility"
    CUSTOM = "custom"


@dataclass
class SkillDefinition:
    """Definition of a reusable skill"""
    name: str
    description: str
    category: SkillCategory
    prompt: str
    tags: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    author: Optional[str] = None
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Convert skill to Claude Code markdown format"""
        return f"""---
name: {self.name}
description: {self.description}
category: {self.category.value}
tags: {', '.join(self.tags)}
required_tools: {', '.join(self.required_tools)}
author: {self.author or ''}
version: {self.version}
---

{self.prompt}"""


@dataclass
class SkillExecution:
    """Result of skill execution"""
    skill_name: str
    status: str
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_uses: int = 0


class SkillToolInput(BaseModel):
    """Input schema for Skill tool"""
    name: str = {"description": "Name of the skill to execute"}
    arguments: Optional[Dict[str, Any]] = {"description": "Arguments for the skill"}


class SkillTool(BaseTool):
    """LangChain tool for executing skills"""
    name = "Skill"
    description = "Execute a reusable skill/workflow"

    def _run(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        from skills import SkillManager
        manager = SkillManager()
        result = manager.execute_skill(name, arguments or {})
        return str(result.output) if result.status == "success" else f"Error: {result.error}"


class SkillManager:
    """Manages skills for the agent

    Claude Code skills architecture:
    - Skills defined in skills/ directory
    - Markdown format with YAML header
    - Can be invoked via Skill tool
    - Support for arguments and templates
    """

    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir or "skills")
        self.skills: Dict[str, SkillDefinition] = {}
        self.load_skills()

    def load_skills(self):
        """Load all skills from the skills directory"""
        if not self.skills_dir.exists():
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_skills()
            return

        for skill_file in self.skills_dir.glob("*.md"):
            try:
                skill = self._load_skill_from_file(skill_file)
                if skill:
                    self.skills[skill.name.lower()] = skill
            except Exception as e:
                print(f"Failed to load skill {skill_file}: {e}")

    def _create_default_skills(self):
        """Create default skills"""
        default_skills = [
            SkillDefinition(
                name="CodeReview",
                description="Perform code review on specified files",
                category=SkillCategory.CODE,
                prompt="""Perform a comprehensive code review:

Files to review: {{files}}

Review checklist:
1. Code quality and readability
2. Potential bugs or errors
3. Security vulnerabilities
4. Performance issues
5. Best practices adherence

Provide detailed feedback with specific line numbers.""",
                tags=["review", "code", "quality"],
                required_tools=["Read", "Grep"]
            ),
            SkillDefinition(
                name="Commit",
                description="Create a git commit with proper message",
                category=SkillCategory.DEVOPS,
                prompt="""Create a git commit with a well-formatted message:

Changes made: {{changes}}

Commit message guidelines:
- Use imperative mood
- Limit subject line to 50 characters
- Add body with details if needed
- Reference issue numbers if applicable

Execute the commit and report the hash.""",
                tags=["git", "commit", "version control"],
                required_tools=["Bash"]
            ),
            SkillDefinition(
                name="Verify",
                description="Verify code changes work correctly",
                category=SkillCategory.CODE,
                prompt="""Verify code changes:

Files changed: {{files}}

Verification steps:
1. Run tests with the feature enabled
2. Run typechecks and investigate errors
3. Verify the change works as expected
4. Report any issues found

Be skeptical - if something looks off, dig deeper.""",
                tags=["test", "verify", "quality"],
                required_tools=["Bash", "Read"]
            ),
            SkillDefinition(
                name="Explore",
                description="Explore a codebase structure",
                category=SkillCategory.RESEARCH,
                prompt="""Explore the codebase structure:

Target: {{target}}

Explore steps:
1. List directory structure
2. Find key files and directories
3. Identify main entry points
4. Understand project architecture
5. Report findings with file paths and purpose.""",
                tags=["explore", "research", "codebase"],
                required_tools=["Glob", "Read", "Grep"]
            )
        ]

        for skill in default_skills:
            self.save_skill(skill)

    def _load_skill_from_file(self, file_path: Path) -> Optional[SkillDefinition]:
        """Load a skill from a markdown file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            return self._parse_markdown_skill(content)
        except Exception:
            return None

    def _parse_markdown_skill(self, content: str) -> Optional[SkillDefinition]:
        """Parse a skill from markdown content"""
        yaml_match = re.search(r'---\n(.*?)\n---', content, re.DOTALL)
        if not yaml_match:
            return None

        yaml_content = yaml_match.group(1)
        config = {}
        for line in yaml_content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                config[key.strip()] = value.strip()

        name = config.get('name')
        description = config.get('description')
        category = SkillCategory(config.get('category', 'custom'))
        tags = [t.strip() for t in config.get('tags', '').split(',')] if config.get('tags') else []
        required_tools = [t.strip() for t in config.get('required_tools', '').split(',')] if config.get('required_tools') else []
        author = config.get('author')
        version = config.get('version', '1.0')

        prompt = content.split('---', 2)[-1].strip()

        return SkillDefinition(
            name=name,
            description=description,
            category=category,
            prompt=prompt,
            tags=tags,
            required_tools=required_tools,
            author=author,
            version=version
        )

    def save_skill(self, skill: SkillDefinition):
        """Save a skill to file"""
        file_name = skill.name.lower().replace(' ', '_') + '.md'
        file_path = self.skills_dir / file_name
        file_path.write_text(skill.to_markdown(), encoding='utf-8')

    def execute_skill(
        self,
        skill_name: str,
        arguments: Dict[str, Any],
        agent: Optional[Any] = None
    ) -> SkillExecution:
        """Execute a skill"""
        import time
        start_time = time.time()

        skill_name_lower = skill_name.lower()
        if skill_name_lower not in self.skills:
            return SkillExecution(
                skill_name=skill_name,
                status="failed",
                output="",
                error=f"Skill '{skill_name}' not found"
            )

        skill = self.skills[skill_name_lower]
        prompt = self._render_prompt(skill.prompt, arguments)

        if agent:
            result = agent.run(prompt, agent_type="claude")
            output = str(result.get("output", ""))
            status = "success" if result.get("status") == "completed" else "failed"
        else:
            output = prompt
            status = "success"

        return SkillExecution(
            skill_name=skill_name,
            status=status,
            output=output,
            execution_time=time.time() - start_time
        )

    def _render_prompt(self, prompt: str, arguments: Dict[str, Any]) -> str:
        """Render prompt with arguments"""
        result = prompt
        for key, value in arguments.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        """Get a skill by name"""
        return self.skills.get(name.lower())

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all available skills"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category.value,
                "tags": s.tags,
                "version": s.version
            }
            for s in self.skills.values()
        ]

    def get_skills_by_category(self, category: SkillCategory) -> List[SkillDefinition]:
        """Get skills by category"""
        return [s for s in self.skills.values() if s.category == category]
