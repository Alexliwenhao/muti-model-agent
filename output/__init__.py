"""Structured Output System

Based on Claude Code patterns for consistent, structured agent outputs.
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class OutputFormat(Enum):
    """Output format types"""
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"


@dataclass
class StructuredOutput:
    """Structured output from agent"""
    format: OutputFormat
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"
    error: Optional[str] = None

    def to_text(self) -> str:
        """Convert to text format"""
        if self.format == OutputFormat.JSON:
            return json.dumps(self.content, indent=2)
        elif self.format == OutputFormat.XML:
            return self._to_xml()
        elif self.format == OutputFormat.MARKDOWN:
            return self._to_markdown()
        return str(self.content)

    def _to_xml(self) -> str:
        """Convert to XML format"""
        if isinstance(self.content, dict):
            return self._dict_to_xml(self.content)
        return f"<output>{self.content}</output>"

    def _dict_to_xml(self, d: Dict, root: str = "response") -> str:
        """Convert dict to XML"""
        xml = f"<{root}>"
        for k, v in d.items():
            key = re.sub(r'[^a-zA-Z0-9_]', '_', str(k))
            if isinstance(v, dict):
                xml += self._dict_to_xml(v, key)
            elif isinstance(v, list):
                for item in v:
                    xml += f"<{key}>{item}</{key}>"
            else:
                xml += f"<{key}>{v}</{key}>"
        xml += f"</{root}>"
        return xml

    def _to_markdown(self) -> str:
        """Convert to markdown format"""
        if isinstance(self.content, dict):
            lines = ["## Response", ""]
            self._dict_to_markdown(self.content, lines, 0)
            return "\n".join(lines)
        return str(self.content)

    def _dict_to_markdown(self, d: Dict, lines: List, depth: int):
        """Convert dict to markdown recursively"""
        for k, v in d.items():
            prefix = "#" * min(depth + 2, 6)
            if isinstance(v, dict):
                lines.append(f"{prefix} {k}")
                self._dict_to_markdown(v, lines, depth + 1)
            elif isinstance(v, list):
                lines.append(f"{prefix} {k}")
                for item in v:
                    lines.append(f"- {item}")
            else:
                lines.append(f"{prefix} {k}: {v}")


@dataclass
class OutputSchema:
    """Schema for structured output"""
    name: str
    description: str
    fields: Dict[str, Dict[str, Any]]
    required_fields: List[str] = field(default_factory=list)

    def validate(self, data: Any) -> tuple[bool, Optional[str]]:
        """Validate output against schema"""
        if not isinstance(data, dict):
            return False, "Output must be a dictionary"

        for field_name in self.required_fields:
            if field_name not in data:
                return False, f"Missing required field: {field_name}"

        for field_name, field_schema in self.fields.items():
            if field_name in data:
                expected_type = field_schema.get("type")
                if expected_type and not isinstance(data[field_name], eval(expected_type)):
                    return False, f"Field {field_name} must be of type {expected_type}"

        return True, None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format"""
        properties = {}
        required = []

        for name, schema in self.fields.items():
            prop = {"description": schema.get("description", "")}
            type_map = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}
            prop["type"] = type_map.get(schema.get("type", "str"), "string")
            properties[name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": self.required_fields
        }


class OutputFormatter:
    """Formats agent outputs according to specified formats"""

    TASK_RESULT_SCHEMA = OutputSchema(
        name="task_result",
        description="Result of an agent task",
        fields={
            "status": {"type": "str", "description": "Task status"},
            "result": {"type": "str", "description": "Task result"},
            "metadata": {"type": "dict", "description": "Additional metadata"}
        },
        required_fields=["status", "result"]
    )

    CODE_REVIEW_SCHEMA = OutputSchema(
        name="code_review",
        description="Code review result",
        fields={
            "issues": {"type": "list", "description": "Found issues"},
            "suggestions": {"type": "list", "description": "Suggestions"},
            "overall": {"type": "str", "description": "Overall assessment"}
        },
        required_fields=["issues", "overall"]
    )

    IMPLEMENTATION_SCHEMA = OutputSchema(
        name="implementation",
        description="Implementation result",
        fields={
            "files_changed": {"type": "list", "description": "List of changed files"},
            "commits": {"type": "list", "description": "Git commits made"},
            "tests": {"type": "str", "description": "Test results"}
        },
        required_fields=["files_changed"]
    )

    @staticmethod
    def format_as(
        data: Any,
        format_type: OutputFormat,
        schema: Optional[OutputSchema] = None
    ) -> StructuredOutput:
        """Format output as specified type"""
        if format_type == OutputFormat.JSON:
            content = data if isinstance(data, dict) else {"content": data}
        elif format_type == OutputFormat.XML:
            content = data if isinstance(data, dict) else {"content": data}
        else:
            content = data

        output = StructuredOutput(
            format=format_type,
            content=content
        )

        if schema:
            is_valid, error = schema.validate(data)
            output.validation_status = "valid" if is_valid else "invalid"
            output.error = error

        return output

    @staticmethod
    def parse_structured(text: str, schema: OutputSchema) -> Optional[Dict]:
        """Parse structured output from text (JSON or schema format)"""
        text = text.strip()

        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        lines = text.split("\n")
        result = {}
        current_key = None
        current_list = []

        for line in lines:
            line = line.strip()
            if ":" in line and not line.startswith("-"):
                if current_key:
                    result[current_key] = current_list if current_list else line.split(":", 1)[1].strip()
                key, value = line.split(":", 1)
                current_key = key.strip()
                current_list = []
                if value.strip():
                    result[current_key] = value.strip()
            elif line.startswith("-") or line.startswith("*"):
                current_list.append(line.lstrip("-* ").strip())

        if current_key:
            result[current_key] = current_list if current_list else result.get(current_key)

        return result

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict]:
        """Extract JSON object from mixed text"""
        json_patterns = [
            r'\{[^{}]*\}',
            r'\{[^{}]*\{[^{}]*\}[^{}]*\}',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None
