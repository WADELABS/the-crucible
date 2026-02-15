"""
src/logic/validator.py
The Material Consistency Engine: Detecting 'hallucinated' history.
"""

from typing import Dict, List

class MaterialValidator:
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

    def verify_claim(self, claimed_era: str, observed_features: List[str]) -> Dict:
        """
        Validates if the observed features match the claimed historical era.
        """
        if claimed_era not in self.era_markers:
            return {"status": "UNKNOWN_ERA", "message": f"Era '{claimed_era}' not in verification database."}

        markers = self.era_markers[claimed_era]
        anachronisms = [f.upper() for f in observed_features if f.upper() in markers["prohibited"]]
        
        if anachronisms:
            return {
                "status": "REJECTED",
                "anachronisms_detected": anachronisms,
                "message": f"Claim rejected: {', '.join(anachronisms)} did not exist in the {claimed_era}."
            }
            
        return {
            "status": "PROBABLE",
            "message": f"No immediate anachronisms detected for the {claimed_era}."
        }
