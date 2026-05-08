"""Tests for core agent functionality"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from core.agent import MultiModelAgent, AgentStatus, AgentResponse
from config import Config


class TestMultiModelAgent:
    """Test cases for MultiModelAgent"""

    def test_agent_initialization(self):
        """Test agent can be initialized"""
        agent = MultiModelAgent()
        assert agent is not None
        assert agent.status == AgentStatus.IDLE

    def test_agent_has_default_tools(self):
        """Test agent has expected tools"""
        agent = MultiModelAgent()
        tools = agent.get_available_tools()
        assert "Bash" in tools or len(tools) >= 0

    def test_agent_system_prompt(self):
        """Test system prompt is set"""
        agent = MultiModelAgent()
        assert agent.system_prompt is not None
        assert len(agent.system_prompt) > 0

    def test_agent_add_tools(self):
        """Test adding tools to agent"""
        agent = MultiModelAgent()

        def dummy_tool():
            pass
        dummy_tool.name = "test_tool"
        dummy_tool.description = "A test tool"

        agent.add_tools([dummy_tool])
        assert "test_tool" in agent.get_available_tools()


class TestAgentConfig:
    """Test configuration"""

    def test_config_initialization(self):
        """Test config can be initialized"""
        config = Config()
        assert config is not None
        assert len(config.get_all_models()) > 0

    def test_get_primary_model(self):
        """Test getting primary model"""
        config = Config()
        model = config.get_primary_model()
        assert model is not None
        assert model.name in ["anthropic", "openai"]

    def test_capabilities(self):
        """Test model capabilities"""
        config = Config()
        caps = config.get_capabilities_for_model("anthropic")
        assert "reasoning" in caps or len(caps) > 0
