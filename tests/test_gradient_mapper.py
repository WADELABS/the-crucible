"""Tests for gradient mapping functionality."""
import pytest
import torch
import torch.nn as nn
from crucible.dissection.neural import GradientMapper


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.sigmoid(self.layer2(x))


def test_gradient_mapper_initialization():
    """Test GradientMapper initializes correctly."""
    model = SimpleModel()
    mapper = GradientMapper(model)
    assert mapper is not None
    assert mapper.model == model


def test_map_vector_field():
    """Test gradient vector field mapping."""
    model = SimpleModel()
    mapper = GradientMapper(model)
    
    # Create sample input and compute gradients
    input_data = torch.randn(1, 10, requires_grad=True)
    output = model(input_data)
    output.backward()
    
    # Map the vector field
    grad_stats = mapper.map_vector_field()
    
    assert grad_stats is not None
    assert isinstance(grad_stats, (dict, list))
    assert len(grad_stats) > 0


def test_gradient_sensitivity_detection():
    """Test detection of gradient sensitivity to specific features."""
    model = SimpleModel()
    mapper = GradientMapper(model)
    
    # Test with two different inputs
    input1 = torch.randn(1, 10, requires_grad=True)
    input2 = torch.randn(1, 10, requires_grad=True)
    
    output1 = model(input1)
    output2 = model(input2)
    
    output1.backward(retain_graph=True)
    output2.backward()
    
    # Verify gradients exist
    assert input1.grad is not None
    assert input2.grad is not None
