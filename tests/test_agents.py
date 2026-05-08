"""Tests for sub-agent system"""
import pytest
import asyncio
from agents import SubAgentManager, SubAgent, SubAgentConfig, SubAgentType, SubAgentResult


class TestSubAgent:
    """Test cases for SubAgent"""

    def test_sub_agent_creation(self):
        """Test sub-agent can be created"""
        config = SubAgentConfig(
            name="TestAgent",
            description="Test agent",
            agent_type=SubAgentType.GENERAL
        )
        agent = SubAgent(config)
        assert agent is not None
        assert agent.agent_id is not None
        assert agent.status == "initialized"

    def test_sub_agent_has_isolated_context(self):
        """Test sub-agent has isolated context"""
        config = SubAgentConfig(
            name="TestAgent",
            description="Test agent",
            agent_type=SubAgentType.GENERAL
        )
        agent = SubAgent(config)
        context = agent.get_isolated_context()
        assert isinstance(context, list)


class TestSubAgentManager:
    """Test cases for SubAgentManager"""

    def test_create_default_sub_agent(self):
        """Test creating default sub-agent"""
        from config import Config
        manager = SubAgentManager(Config())

        agent = manager.create_sub_agent(agent_type=SubAgentType.EXPLORE)
        assert agent is not None
        assert agent.config.name == "Explorer"

    def test_create_custom_sub_agent(self):
        """Test creating custom sub-agent"""
        from config import Config
        manager = SubAgentManager(Config())

        config = SubAgentConfig(
            name="CustomAgent",
            description="Custom test agent",
            agent_type=SubAgentType.GENERAL,
            tools=["Read", "Write"]
        )

        agent = manager.create_sub_agent(agent_config=config)
        assert agent.config.name == "CustomAgent"
        assert agent.config.tools == ["Read", "Write"]

    def test_active_agents_tracking(self):
        """Test tracking active agents"""
        from config import Config
        manager = SubAgentManager(Config())

        agent = manager.create_sub_agent(agent_type=SubAgentType.GENERAL)
        assert manager.get_active_count() == 1

        manager.complete_agent(agent.agent_id, SubAgentResult(
            agent_id=agent.agent_id,
            agent_name=agent.config.name,
            status="completed",
            summary="Test completed"
        ))

        assert manager.get_active_count() == 0
        assert len(manager.get_results()) == 1
