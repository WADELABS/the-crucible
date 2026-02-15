"""Pytest configuration and shared fixtures."""
import pytest
import torch


@pytest.fixture
def sample_model():
    """Fixture providing a simple test model."""
    import torch.nn as nn
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))
    
    return TestModel()


@pytest.fixture
def sample_input():
    """Fixture providing sample input data."""
    return torch.randn(1, 10)


@pytest.fixture
def sample_metrics():
    """Fixture providing sample metrics for testing."""
    return {
        "model_id": "test_model",
        "gradient_sensitivity": 0.5,
        "entropy": 2.5,
        "vulnerabilities_found": 1,
        "compliance_score": 0.8
    }
