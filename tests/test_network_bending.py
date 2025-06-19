#!/usr/bin/env python

"""Tests for `network_bending` package."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

# Import the nodes from the package
from src.network_bending.nodes import NetworkBending, NetworkBendingAdvanced, ModelMixer


class SimpleModel(nn.Module):
    """A simple test model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.linear = nn.Linear(32 * 8 * 8, 10)
        self.norm = nn.BatchNorm2d(32)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


@pytest.fixture
def mock_model():
    """Create a mock ComfyUI model wrapper"""
    model = Mock()
    model.model = SimpleModel()
    model.clone = Mock(return_value=model)
    return model


@pytest.fixture
def mock_prompt_server(monkeypatch):
    """Mock the PromptServer for testing"""
    mock_server = Mock()
    mock_instance = Mock()
    mock_instance.send_sync = Mock()
    mock_server.instance = mock_instance
    
    # Create a mock module for server
    import sys
    from types import ModuleType
    server_module = ModuleType('server')
    server_module.PromptServer = mock_server
    sys.modules['server'] = server_module
    
    return mock_instance


class TestNetworkBending:
    """Test the NetworkBending node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = NetworkBending.INPUT_TYPES()
        
        assert "required" in input_types
        assert "model" in input_types["required"]
        assert "operation" in input_types["required"]
        assert "intensity" in input_types["required"]
        assert "target_layers" in input_types["required"]
        assert "seed" in input_types["required"]
        
        # Check operation list
        operations = input_types["required"]["operation"][0]
        assert "add_noise" in operations
        assert "scale_weights" in operations
        assert "prune_weights" in operations
    
    def test_add_noise_operation(self, mock_model, mock_prompt_server):
        """Test add_noise operation"""
        node = NetworkBending()
        
        # Run the operation
        result = node.bend_network(
            model=mock_model,
            operation="add_noise",
            intensity=0.1,
            target_layers="all",
            seed=42
        )
        
        # Check that model was cloned
        mock_model.clone.assert_called_once()
        
        # Check that result is returned
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 1
    
    def test_target_layer_filtering(self, mock_model, mock_prompt_server):
        """Test that target layer filtering works"""
        node = NetworkBending()
        
        # Test with specific layer pattern
        result = node.bend_network(
            model=mock_model,
            operation="add_noise",
            intensity=0.1,
            target_layers="conv",
            seed=42
        )
        
        # Verify feedback was sent
        mock_prompt_server.send_sync.assert_called()
        call_args = mock_prompt_server.send_sync.call_args
        assert call_args[0][0] == "network_bending.feedback"
        assert "conv" in str(call_args[0][1]["modified_layers"])
    
    def test_scale_weights_operation(self, mock_model, mock_prompt_server):
        """Test scale_weights operation"""
        node = NetworkBending()
        
        result = node.bend_network(
            model=mock_model,
            operation="scale_weights",
            intensity=0.7,  # Should scale by 1.4
            target_layers="linear",
            seed=42
        )
        
        assert result is not None
    
    def test_seed_reproducibility(self, mock_model, mock_prompt_server):
        """Test that setting seed produces reproducible results"""
        node = NetworkBending()
        
        # Get initial weights
        initial_weights = {}
        for name, param in mock_model.model.named_parameters():
            initial_weights[name] = param.data.clone()
        
        # Run with seed
        result1 = node.bend_network(
            model=mock_model,
            operation="add_noise",
            intensity=0.1,
            target_layers="all",
            seed=12345
        )
        
        # Weights should have changed
        for name, param in mock_model.model.named_parameters():
            assert not torch.allclose(initial_weights[name], param.data)


class TestModelMixer:
    """Test the ModelMixer node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = ModelMixer.INPUT_TYPES()
        
        assert "required" in input_types
        assert "model_a" in input_types["required"]
        assert "model_b" in input_types["required"]
        assert "mix_mode" in input_types["required"]
        assert "mix_ratio" in input_types["required"]
    
    def test_linear_interpolation(self, mock_model):
        """Test linear interpolation mixing"""
        node = ModelMixer()
        
        # Create two mock models
        model_a = mock_model
        model_b = Mock()
        model_b.model = SimpleModel()
        
        # Set different weights for model_b
        for param in model_b.model.parameters():
            param.data.fill_(2.0)
        
        result = node.mix_models(
            model_a=model_a,
            model_b=model_b,
            mix_mode="linear_interpolation",
            mix_ratio=0.5
        )
        
        assert result is not None
        assert isinstance(result, tuple)


class TestNetworkBendingAdvanced:
    """Test the NetworkBendingAdvanced node"""
    
    def test_input_types(self):
        """Test that INPUT_TYPES returns correct structure"""
        input_types = NetworkBendingAdvanced.INPUT_TYPES()
        
        assert "required" in input_types
        assert "model" in input_types["required"]
        assert "operation" in input_types["required"]
        assert "intensity" in input_types["required"]
        assert "preserve_functionality" in input_types["required"]
        
        # Check advanced operations
        operations = input_types["required"]["operation"][0]
        assert "layer_swap" in operations
        assert "activation_replace" in operations
        assert "weight_transpose" in operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
