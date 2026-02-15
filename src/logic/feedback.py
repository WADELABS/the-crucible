"""
src/logic/feedback.py
Feedback Injection: Refining substrate configs based on stress-test failures.
"""

from typing import Dict, Any, List
import yaml
import os

class FeedbackInjection:
    """
    The 'Feedback Loop': Neutralizing vulnerabilities at the source.
    """
    def __init__(self, target_repos: List[str]):
        self.target_repos = target_repos

    def inject_feedback(self, test_result: Dict[str, Any]):
        """
        Analyzes a failure and injects configuration adjustments into the target pillar.
        """
        if test_result.get("veracity_score", 1.0) < 0.5:
            self._refine_thresholds(test_result)

    def _refine_thresholds(self, test_result: Dict[str, Any]):
        """
        Aggressively tightens thresholds in the target repo's config.yaml.
        """
        target = test_result.get("origin_pillar", "unknown")
        print(f"DEBUG: Stress-test failure in {target}. Injecting threshold refinement...")
        
        # Mock logic for updating a config.yaml
        config_path = f"C:/Users/sheew/.gemini/antigravity/scratch/{target}/config.yaml"
        if os.path.exists(config_path):
             with open(config_path, 'r') as f:
                 config = yaml.safe_load(f) or {}
             
             # Tighten the 'uncertainty' or 'utilization' thresholds
             if 'automation' in config:
                 config['automation']['utilization']['target_maximum'] *= 0.9
             
             with open(config_path, 'w') as f:
                 yaml.dump(config, f)
             print(f"✓ Feedback injected: High-fidelity threshold tightened for {target}.")
