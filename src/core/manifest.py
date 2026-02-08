from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import yaml
import jsonschema

@dataclass
class ManifestConfig:
    type: str
    hypothesis: str
    methodology: str
    success_criteria: str
    failure_modes: List[str]
    instrumentation: List[str]

class Manifest:
    """
    Principle 1: The Manifest as Computational Contract.
    
    A Manifest is not merely a configuration file; it is an executable contract 
    between intention and implementation. It encodes a hypothesis, a methodology,
    and strict criteria for success and failure.
    """
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "manifest": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "methodology": {"type": "string"},
                    "success_criteria": {"type": "string"},
                    "failure_modes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "instrumentation": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["type", "hypothesis", "methodology", "success_criteria", "failure_modes", "instrumentation"]
            }
        },
        "required": ["manifest"]
    }

    def __init__(self, content: Dict[str, Any]):
        self._validate(content)
        self.config = self._parse(content)
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'Manifest':
        """Load a manifest from a YAML string."""
        try:
            content = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML content: {e}")
        return cls(content)

    @classmethod
    def from_file(cls, filepath: str) -> 'Manifest':
        """Load a manifest from a YAML file."""
        with open(filepath, 'r') as f:
            return cls.from_yaml(f.read())

    def _validate(self, content: Dict[str, Any]):
        """Validate the manifest against the core schema."""
        try:
            jsonschema.validate(instance=content, schema=self.SCHEMA)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Manifest violates computational contract: {e.message}")

    def _parse(self, content: Dict[str, Any]) -> ManifestConfig:
        """Parse raw dictionary into structured ManifestConfig."""
        m = content['manifest']
        return ManifestConfig(
            type=m['type'],
            hypothesis=m['hypothesis'],
            methodology=m['methodology'],
            success_criteria=m['success_criteria'],
            failure_modes=m['failure_modes'],
            instrumentation=m['instrumentation']
        )
    
    def execute_contract(self) -> str:
        """
        Symbolic execution of the contract. 
        In a full system, this would orchestrate the experiment.
        """
        return (
            f"Executing Contract: {self.config.type}\n"
            f"  Hypothesis: {self.config.hypothesis}\n"
            f"  Methodology: {self.config.methodology}"
        )

    def __repr__(self):
        return f"<Manifest(type='{self.config.type}', hypothesis='{self.config.hypothesis}')>"
