"""Configuration management for Multi-Model Agent"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 4096
    capabilities: List[str] = field(default_factory=list)


@dataclass
class CollaborationConfig:
    max_concurrent_tasks: int = 3
    task_decomposition_threshold: int = 3
    model_selection_strategy: str = "capability_based"
    fallback_enabled: bool = True
    max_retries: int = 2
    timeout_seconds: int = 300


@dataclass
class AgentConfig:
    name: str = "MultiModelAgent"
    description: str = "A multi-model collaborative agent based on Claude Code architecture"
    max_iterations: int = 100
    verbose: bool = True
    context_window: int = 200000
    auto_compact_threshold: float = 0.75


class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config_dir = Path(config_path) if config_path else Path(__file__).parent.parent / "config"
        self.models: Dict[str, ModelConfig] = {}
        self.collaboration = CollaborationConfig()
        self.agent = AgentConfig()
        self._load_default_config()
        self._load_from_env()

    def _load_default_config(self):
        self.models["anthropic"] = ModelConfig(
            name="anthropic",
            provider="anthropic",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20241022",
            capabilities=["reasoning", "code_generation", "analysis", "creative_writing"]
        )
        self.models["openai"] = ModelConfig(
            name="openai",
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            capabilities=["reasoning", "code_generation", "analysis", "creative_writing"]
        )
        self.models["local"] = ModelConfig(
            name="local",
            provider="openai",
            api_key=os.getenv("LOCAL_API_KEY", "not-needed"),
            base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("LOCAL_MODEL", "deepseek-coder"),
            capabilities=["code_generation", "fast_response"]
        )

    def _load_from_env(self):
        if os.getenv("CLAUDE_MODEL"):
            self.models["anthropic"].model = os.getenv("CLAUDE_MODEL")
        if os.getenv("OPENAI_MODEL"):
            self.models["openai"].model = os.getenv("OPENAI_MODEL")

    def get_model(self, name: str) -> Optional[ModelConfig]:
        return self.models.get(name)

    def get_all_models(self) -> Dict[str, ModelConfig]:
        return self.models

    def get_primary_model(self) -> ModelConfig:
        return self.models.get("anthropic", self.models.get("openai"))

    def get_capabilities_for_model(self, model_name: str) -> List[str]:
        model = self.models.get(model_name)
        return model.capabilities if model else []

    def get_models_with_capability(self, capability: str) -> List[str]:
        return [
            name for name, model in self.models.items()
            if capability in model.capabilities
        ]
