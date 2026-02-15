"""
src/auditor/registry_check.py
The Story Auditor: Fact-checking the fabrication.
"""

from typing import Dict, Optional, Any, List

class StoryAuditor:
    def __init__(self):
        # Mock database of historical business registries
        self.registry_database = {
            "MILLER'S WOODSHOP": {"founded": 1985, "location": "VERMONT", "status": "CLOSED"},
            "HERMAN MILLER": {"founded": 1905, "location": "MICHIGAN", "status": "ACTIVE"}
        }

    def verify_business(self, business_name: str, claimed_year: Any) -> Dict:
        """
        Verifies if a business existed during the claimed production year.
        """
        name_upper = business_name.upper()
        if name_upper not in self.registry_database:
            return {
                "status": "UNVERIFIED",
                "message": f"No record of '{business_name}' found in historical registries."
            }
            
        record = self.registry_database[name_upper]
        try:
            year_int = int(claimed_year)
        except (ValueError, TypeError):
            return {"status": "ERROR", "message": "Claimed year must be a numeric value."}

        if int(year_int) < int(record["founded"]):
            return {
                "status": "FABRICATED",
                "message": f"Conflict detected: {business_name} was founded in {record['founded']}, but claim is for {year_int}."
            }
            
        return {
            "status": "VERIFIED",
            "message": f"Confirmed: {business_name} was active in {claimed_year}."
        }

    def verify_price_salience(self, claimed_original_price: float, current_asking_price: float) -> str:
        """
        Heuristic for detecting 'hallucinated' original value.
        """
        if current_asking_price > claimed_original_price * 10:
            return "SIGNAL: High appreciation detected. Requires deep provenance check."
        return "SIGNAL: Normal depreciation/appreciation curve."
