"""
tests/test_forensics.py
The IKEA Test: Verifying veracity against mass-produced hallucinations.
"""

import unittest
from src.forensics.label_detector import LabelDetector
from src.logic.validator import MaterialValidator
from src.auditor.registry_check import StoryAuditor

class TheIkeaTest(unittest.TestCase):
    def setUp(self):
        self.detector = LabelDetector()
        self.validator = MaterialValidator()
        self.auditor = StoryAuditor()

    def test_mass_production_detection(self):
        """Test detection of hidden IKEA/China markers in OCR text."""
        ocr_text = "Beautiful hand-carved frame. Distressed finish. [MADE IN CHINA] marker on back."
        result = self.detector.scan_text(ocr_text)
        self.assertEqual(result["status"], "FLAGGED")
        self.assertIn("MADE IN CHINA", result["markers_detected"])

    def test_material_anachronism(self):
        """Test detection of modern hardware in a vintage claim."""
        claimed_era = "1950s"
        observed = ["MDF", "Philips Head Screw", "Maple Wood"]
        result = self.validator.verify_claim(claimed_era, observed)
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("PHILIPS HEAD SCREW", result["anachronisms_detected"])

    def test_story_fabrication(self):
        """Test detection of business registry conflicts."""
        business = "Miller's Woodshop"
        year = 1980
        result = self.auditor.verify_business(business, year)
        self.assertEqual(result["status"], "FABRICATED") # Founded in 1985

if __name__ == "__main__":
    unittest.main()
