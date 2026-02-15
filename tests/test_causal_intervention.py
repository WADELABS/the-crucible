"""Tests for causal intervention hooks."""
import pytest
import torch
import torch.nn as nn
from crucible.dissection.neural import CausalIntervener


class InterventionTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.sigmoid(self.layer3(x))


def test_causal_intervener_initialization():
    """Test CausalIntervener initializes correctly."""
    model = InterventionTestModel()
    intervener = CausalIntervener(model)
    assert intervener is not None


def test_apply_ablation_hook():
    """Test applying ablation hooks to specific layers."""
    model = InterventionTestModel()
    intervener = CausalIntervener(model)
    
    # Apply ablation to layer1
    hook_handle = intervener.apply_ablation_hook("layer1")
    assert hook_handle is not None
    
    # Run inference with ablation
    input_data = torch.randn(1, 10)
    output = model(input_data)
    assert output is not None
    assert output.shape == (1, 1)


def test_intervention_changes_output():
    """Test that intervention actually changes model behavior."""
    model = InterventionTestModel()
    intervener = CausalIntervener(model)
    
    input_data = torch.randn(1, 10)
    
    # Get baseline output
    output_baseline = model(input_data).detach()
    
    # Apply intervention
    intervener.apply_ablation_hook("layer1")
    output_intervened = model(input_data).detach()
    
    # Outputs should differ when intervention is applied
    assert not torch.allclose(output_baseline, output_intervened, rtol=1e-5)
