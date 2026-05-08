"""Tests for collaboration engine"""
import pytest
import asyncio
from collaboration import (
    CollaborationEngine, CollaborationMode, CollaborationResult,
    FinalResult, ModelRegistry
)
from config import Config


class TestModelRegistry:
    """Test model registry"""

    def test_registry_initialization(self):
        """Test registry can be initialized"""
        config = Config()
        registry = ModelRegistry(config)
        assert registry is not None
        assert len(registry.models) > 0

    def test_get_model_for_capability(self):
        """Test getting model for capability"""
        config = Config()
        registry = ModelRegistry(config)
        model = registry.get_model_for_capability("reasoning")
        assert model is not None or len(registry.models) == 0


class TestCollaborationEngine:
    """Test collaboration engine"""

    def test_engine_initialization(self):
        """Test engine can be initialized"""
        config = Config()
        engine = CollaborationEngine(config)
        assert engine is not None
        assert engine.model_registry is not None

    def test_get_model_status(self):
        """Test getting model status"""
        config = Config()
        engine = CollaborationEngine(config)
        status = engine.get_model_status()
        assert "total_models" in status
        assert "models" in status

    def test_confidence_calculation(self):
        """Test confidence calculation"""
        config = Config()
        engine = CollaborationEngine(config)

        high_content = "This is a detailed response.\nWith multiple lines.\nAnd good structure."
        low_content = "Short"

        high_conf = engine._calculate_confidence(high_content)
        low_conf = engine._calculate_confidence(low_content)

        assert high_conf > low_conf
        assert 0 <= high_conf <= 1
        assert 0 <= low_conf <= 1
