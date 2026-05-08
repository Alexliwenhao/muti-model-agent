"""Configuration Management - Claude CLI Format

支持 Claude CLI 的配置格式:
- baseUrl: API 端点
- apiKey: API 密钥
- 模型配置
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ClaudeConfig:
    """Claude CLI 格式的配置"""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "claude-3-5-sonnet-20241022"
    default_haiku_model: Optional[str] = None
    default_opus_model: Optional[str] = None
    default_sonnet_model: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaudeConfig":
        """从字典创建配置"""
        return cls(
            base_url=data.get("baseUrl") or data.get("base_url"),
            api_key=data.get("apiKey") or data.get("api_key"),
            model=data.get("model", "claude-3-5-sonnet-20241022"),
            default_haiku_model=data.get("defaultHaikuModel"),
            default_opus_model=data.get("defaultOpusModel"),
            default_sonnet_model=data.get("defaultSonnetModel"),
            timeout=data.get("timeout"),
            max_retries=data.get("maxRetries", 3)
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "baseUrl": self.base_url,
            "apiKey": self.api_key,
            "model": self.model,
            "defaultHaikuModel": self.default_haiku_model,
            "defaultOpusModel": self.default_opus_model,
            "defaultSonnetModel": self.default_sonnet_model,
            "timeout": self.timeout,
            "maxRetries": self.max_retries
        }


@dataclass
class ProviderConfig:
    """提供商配置"""
    name: str
    provider_type: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    default_model: str = ""
    available_models: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ProviderConfig":
        """从字典创建配置"""
        return cls(
            name=name,
            provider_type=data.get("type", "openai-compatible"),
            base_url=data.get("baseUrl") or data.get("base_url"),
            api_key=data.get("apiKey") or data.get("api_key"),
            default_model=data.get("model", ""),
            available_models=data.get("models", []),
            capabilities=data.get("capabilities", []),
            enabled=data.get("enabled", True)
        )


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    capabilities: List[str] = field(default_factory=list)
    enabled: bool = True

    @classmethod
    def from_provider(cls, provider: str, config: ProviderConfig) -> "ModelConfig":
        """从提供商配置创建模型配置"""
        return cls(
            name=provider,
            provider=provider,
            model=config.default_model,
            base_url=config.base_url,
            api_key=config.api_key,
            capabilities=config.capabilities
        )


class ConfigLoader:
    """配置加载器 - 支持 Claude CLI 格式"""

    ENV_VAR_MAPPING = {
        "ANTHROPIC_BASE_URL": "base_url",
        "ANTHROPIC_API_KEY": "api_key",
        "ANTHROPIC_MODEL": "model",
        "OPENAI_BASE_URL": "base_url",
        "OPENAI_API_KEY": "api_key",
        "OPENAI_MODEL": "model",
    }

    @staticmethod
    def load_from_json(file_path: str) -> Dict[str, Any]:
        """从 JSON 文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}

        anthropic_config = {}
        if os.getenv("ANTHROPIC_BASE_URL"):
            anthropic_config["baseUrl"] = os.getenv("ANTHROPIC_BASE_URL").strip('`')
        if os.getenv("ANTHROPIC_API_KEY"):
            anthropic_config["apiKey"] = os.getenv("ANTHROPIC_API_KEY")
        if os.getenv("ANTHROPIC_MODEL"):
            anthropic_config["model"] = os.getenv("ANTHROPIC_MODEL")
        if os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL"):
            anthropic_config["defaultSonnetModel"] = os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL")
        if os.getenv("ANTHROPIC_DEFAULT_OPUS_MODEL"):
            anthropic_config["defaultOpusModel"] = os.getenv("ANTHROPIC_DEFAULT_OPUS_MODEL")
        if os.getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            anthropic_config["defaultHaikuModel"] = os.getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL")

        if anthropic_config:
            config["anthropic"] = anthropic_config

        for prefix in ["OPENAI", "LOCAL", "GROQ", "DASHSCOPE", "ZHIPU", "MOONSHOT", "DEEPSEEK", "OLLAMA", "LMSTUDIO"]:
            provider_key = prefix.lower()
            provider_config = {}

            base_url_env = f"{prefix}_BASE_URL"
            if os.getenv(base_url_env):
                provider_config["baseUrl"] = os.getenv(base_url_env)

            api_key_env = f"{prefix}_API_KEY"
            if os.getenv(api_key_env):
                provider_config["apiKey"] = os.getenv(api_key_env)

            model_env = f"{prefix}_MODEL"
            if os.getenv(model_env):
                provider_config["model"] = os.getenv(model_env)

            if provider_config:
                config[provider_key] = provider_config

        return config

    @staticmethod
    def load_claude_config() -> ClaudeConfig:
        """加载 Claude CLI 格式的配置"""
        config_data = {}

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if base_url:
            config_data["baseUrl"] = base_url.strip('`')

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            config_data["apiKey"] = api_key

        model = os.getenv("ANTHROPIC_MODEL")
        if model:
            config_data["model"] = model

        return ClaudeConfig.from_dict(config_data)


class ProviderRegistry:
    """提供商注册表"""

    DEFAULT_PROVIDERS = {
        "anthropic": ProviderConfig(
            name="anthropic",
            provider_type="anthropic",
            default_model="claude-3-5-sonnet-20241022",
            available_models=[
                "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307"
            ],
            capabilities=["reasoning", "code_generation", "analysis", "creative_writing"]
        ),
        "openai": ProviderConfig(
            name="openai",
            provider_type="openai",
            default_model="gpt-4o",
            available_models=[
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ],
            capabilities=["reasoning", "code_generation", "analysis", "creative_writing"]
        ),
        "deepseek": ProviderConfig(
            name="deepseek",
            provider_type="openai-compatible",
            base_url="https://api.deepseek.com/v1",
            default_model="deepseek-coder",
            available_models=[
                "deepseek-coder",
                "deepseek-chat"
            ],
            capabilities=["code_generation", "reasoning", "analysis"]
        ),
        "qwen": ProviderConfig(
            name="qwen",
            provider_type="openai-compatible",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            default_model="qwen-plus",
            available_models=[
                "qwen-plus",
                "qwen-turbo",
                "qwen-max"
            ],
            capabilities=["reasoning", "code_generation", "analysis", "creative_writing"]
        ),
        "glm": ProviderConfig(
            name="glm",
            provider_type="openai-compatible",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            default_model="glm-4",
            available_models=[
                "glm-4",
                "glm-4-flash",
                "glm-3-turbo"
            ],
            capabilities=["reasoning", "code_generation", "analysis"]
        ),
        "kimi": ProviderConfig(
            name="kimi",
            provider_type="openai-compatible",
            base_url="https://api.moonshot.cn/v1",
            default_model="moonshot-v1-8k",
            available_models=[
                "moonshot-v1-8k",
                "moonshot-v1-32k",
                "moonshot-v1-128k"
            ],
            capabilities=["reasoning", "code_generation", "long_context"]
        ),
        "ollama": ProviderConfig(
            name="ollama",
            provider_type="openai-compatible",
            base_url="http://localhost:11434/v1",
            default_model="llama3",
            available_models=[
                "llama3",
                "llama3:70b",
                "codellama",
                "mistral",
                "mixtral",
                "deepseek-coder"
            ],
            capabilities=["code_generation", "fast_response", "local"]
        ),
        "lmstudio": ProviderConfig(
            name="lmstudio",
            provider_type="openai-compatible",
            base_url="http://localhost:1234/v1",
            default_model="local-model",
            capabilities=["code_generation", "local"]
        )
    }

    @classmethod
    def get_provider(cls, name: str) -> Optional[ProviderConfig]:
        """获取提供商配置"""
        return cls.DEFAULT_PROVIDERS.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """列出所有提供商"""
        return list(cls.DEFAULT_PROVIDERS.keys())

    @classmethod
    def register_provider(cls, name: str, config: ProviderConfig):
        """注册新的提供商"""
        cls.DEFAULT_PROVIDERS[name] = config


class Config:
    """统一的配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_dir = Path(config_path) if config_path else Path(__file__).parent

        self.claude_config: Optional[ClaudeConfig] = None
        self.providers: Dict[str, ProviderConfig] = {}
        self.models: Dict[str, ModelConfig] = {}

        self._load_claude_env()
        self._load_all_providers()
        self._load_from_env()

    def _load_claude_env(self):
        """加载 Claude CLI 环境变量"""
        self.claude_config = ConfigLoader.load_claude_config()

    def _load_all_providers(self):
        """加载所有提供商配置"""
        for name, provider in ProviderRegistry.DEFAULT_PROVIDERS.items():
            config = self._merge_with_env(name, provider)
            self.providers[name] = config

            model_config = ModelConfig.from_provider(name, config)
            self.models[name] = model_config

    def _merge_with_env(self, name: str, provider: ProviderConfig) -> ProviderConfig:
        """合并环境变量配置"""
        prefix = name.upper()

        if os.getenv(f"{prefix}_BASE_URL"):
            provider.base_url = os.getenv(f"{prefix}_BASE_URL")
        if os.getenv(f"{prefix}_API_KEY"):
            provider.api_key = os.getenv(f"{prefix}_API_KEY")
        if os.getenv(f"{prefix}_MODEL"):
            provider.default_model = os.getenv(f"{prefix}_MODEL")

        return provider

    def _load_from_env(self):
        """从环境变量加载额外配置"""
        env_config = ConfigLoader.load_from_env()

        for provider_name, config_data in env_config.items():
            if provider_name not in self.providers:
                provider = ProviderConfig.from_dict(provider_name, config_data)
                self.providers[provider_name] = provider
                self.models[provider_name] = ModelConfig.from_provider(provider_name, provider)

    def get_claude_config(self) -> ClaudeConfig:
        """获取 Claude 配置"""
        return self.claude_config or ClaudeConfig()

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """获取提供商配置"""
        return self.providers.get(name)

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self.models.get(name)

    def get_primary_model(self) -> ModelConfig:
        """获取主模型配置"""
        if self.claude_config and self.claude_config.api_key:
            return ModelConfig(
                name="anthropic",
                provider="anthropic",
                model=self.claude_config.model,
                base_url=self.claude_config.base_url,
                api_key=self.claude_config.api_key,
                capabilities=["reasoning", "code_generation", "analysis", "creative_writing"]
            )

        for name in ["anthropic", "openai", "deepseek"]:
            if name in self.models and self.models[name].api_key:
                return self.models[name]

        return self.models.get("ollama", ModelConfig(
            name="default",
            provider="ollama",
            model="llama3"
        ))

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """获取所有模型"""
        return self.models

    def get_enabled_providers(self) -> List[str]:
        """获取已启用的提供商"""
        return [name for name, p in self.providers.items() if p.enabled and p.api_key]

    def get_capabilities_for_model(self, model_name: str) -> List[str]:
        """获取模型能力"""
        if model_name in self.models:
            return self.models[model_name].capabilities
        return []

    def get_models_with_capability(self, capability: str) -> List[str]:
        """获取具有特定能力的模型"""
        return [
            name for name, model in self.models.items()
            if capability in model.capabilities and model.enabled
        ]

    def to_claude_json(self) -> str:
        """导出为 Claude CLI JSON 格式"""
        config = {
            "autoUpdatesChannel": "latest",
            "dangerouslySkipPermissions": True,
            "skipDangerousModePermissionPrompt": True,
            "env": {}
        }

        if self.claude_config:
            if self.claude_config.base_url:
                config["env"]["ANTHROPIC_BASE_URL"] = f"`{self.claude_config.base_url}`"
            if self.claude_config.api_key:
                config["env"]["ANTHROPIC_API_KEY"] = self.claude_config.api_key
            if self.claude_config.model:
                config["env"]["ANTHROPIC_MODEL"] = self.claude_config.model

        return json.dumps(config, indent=2)

    def save_claude_config(self, path: str):
        """保存为 Claude CLI 配置格式"""
        config = self.to_claude_json()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(config)
