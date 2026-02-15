"""
src/logic/validator.py
The Material Consistency Engine: Detecting 'hallucinated' history.
"""

from typing import Dict, Any, List

# Standardizing for Wadelabs Substrate
try:
    from .feedback import FeedbackInjection
except ImportError:
    from src.logic.feedback import FeedbackInjection

class ForensicValidator:
    """
    The 'Gauntlet': Probing the veracity of story-told value.
    """
    def __init__(self):
        # Database of era-specific material markers
        self.era_markers = {
            "1950s": {
                "hardware": ["FLAT HEAD SCREW", "BRASS HINGE"],
                "prohibited": ["PHILIPS HEAD SCREW", "PLASTIC CAM LOCK", "MDF"],
                "finish": ["SHELLAC", "LACQUER"]
            },
            "1920s": {
                "hardware": ["SQUARE DRIVE", "SLOTTED SCREW"],
                "prohibited": ["PLYWOOD", "ALLEN BOLT"],
                "construction": ["DOVETAIL JOINT"]
            }
        }
        self.feedback = FeedbackInjection(target_repos=["credstack", "inbox-sanitizer"])

    def verify_listing(self, listing: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the three-stage forensic audit. Triggering feedback on failure."""
        claimed_era = listing.get("claimed_era", "unknown")
        observed_features = listing.get("observed_features", [])
        
        # Phase 1: Material Check
        veracity_report = self._check_anachronisms(claimed_era, observed_features)
        
        # Trigger FEEDBACK INJECTION on failure
        if veracity_report.get("status") == "REJECTED":
            veracity_report["origin_pillar"] = listing.get("origin_pillar", "unknown")
            self.feedback.inject_feedback(veracity_report)
            
        return veracity_report

    def _check_anachronisms(self, claimed_era: str, observed_features: List[str]) -> Dict[str, Any]:
        """Validates if the observed features match the claimed historical era."""
        if claimed_era not in self.era_markers:
            return {"status": "UNKNOWN_ERA", "veracity_score": 0.5}

        markers = self.era_markers[claimed_era]
        anachronisms = [f.upper() for f in observed_features if f.upper() in markers["prohibited"]]
        
        if anachronisms:
            return {
                "status": "REJECTED",
                "veracity_score": 0.2,
                "anachronisms_detected": anachronisms,
                "message": f"Claim rejected: {', '.join(anachronisms)} did not exist in the {claimed_era}."
            }
            
        return {
            "status": "PROBABLE",
            "veracity_score": 0.9,
            "message": f"No immediate anachronisms detected for the {claimed_era}."
        }
