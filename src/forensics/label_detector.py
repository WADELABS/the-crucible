"""
src/forensics/label_detector.py
The Label Detector: Stripping away 'distressed' hallucinations.
"""

import re
from typing import List, Dict

class LabelDetector:
    def __init__(self):
        # High-probability scam markers
        self.scam_markers = [
            "MADE IN CHINA",
            "IKEA",
            "ASSEMBLED IN MEXICO",
            "MDF",
            "PARTICLE BOARD",
            "WAYFAIR",
            "OVERSTOCK"
        ]

    def scan_text(self, ocr_results: str) -> Dict:
        """
        Scans OCR text for markers that contradict 'Vintage' or 'Handmade' claims.
        """
        findings = []
        text_upper = ocr_results.upper()
        
        for marker in self.scam_markers:
            if marker in text_upper:
                findings.append(marker)
        
        status = "FLAGGED" if findings else "CLEAN"
        return {
            "status": status,
            "markers_detected": findings,
            "risk_level": "HIGH" if findings else "LOW",
            "message": f"Detected {len(findings)} mass-production markers." if findings else "No obvious mass-production markers found."
        }

    def detect_anachronistic_hardware(self, detected_objects: List[str]) -> List[str]:
        """
        Detects hardware that shouldn't exist in specific eras.
        Example: Philips-head screws in a 1920s claim.
        """
        modern_hardware = ["PHILIPS HEAD SCREW", "PLASTIC CAM LOCK", "EURO HINGE"]
        return [h for h in modern_hardware if h in [obj.upper() for obj in detected_objects]]
