import torch
import numpy as np
from typing import Dict, List
import logging

class InformationBottleneck:
    """
    Layer 4: Information Bottleneck Analysis.
    Calculates mutual information between input and hidden layers 
    to see if the model is 'leaking' biased attributes into decision nodes.
    """
    
    def __init__(self):
        logging.info("Information Bottleneck Analyzer initialized.")

    def calculate_entropy(self, activations: torch.Tensor) -> float:
        """Estimate Shannon entropy of activations."""
        # Simple discretization for demo purposes
        probs = torch.histc(activations, bins=10, min=-1.0, max=1.0) / activations.numel()
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return entropy

class SymbolicExtractor:
    """
    Layer 5: Formal Logical Extraction.
    Attempts to extract discrete rules from continuous weights.
    """
    
    def __init__(self):
        logging.info("Symbolic Rule Extractor initialized.")

    def extract_logic(self, weights: torch.Tensor) -> List[str]:
        """Convert weights into simplified IF-THEN rules."""
        # Simplified: If weight > threshold, it's a decision feature
        rules = []
        threshold = 0.5
        for i, w in enumerate(weights.flatten()[:5]):
            if w > threshold:
                rules.append(f"IF feature_{i} > {threshold} THEN approve_credit")
        return rules
