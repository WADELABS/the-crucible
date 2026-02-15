"""CLI Dashboard for The Crucible analysis results."""
import argparse
import json
from pathlib import Path
from typing import Dict, Any


class CrucibleDashboard:
    """Display analysis results in a formatted CLI dashboard."""
    
    def __init__(self, report_path: str):
        """Initialize dashboard with report data."""
        self.report_path = Path(report_path)
        self.data = self._load_report()
    
    def _load_report(self) -> Dict[str, Any]:
        """Load report JSON."""
        with open(self.report_path, 'r') as f:
            return json.load(f)
    
    def display(self):
        """Display the dashboard."""
        self._print_header()
        self._print_summary()
        self._print_vulnerabilities()
        self._print_recommendations()
        self._print_footer()
    
    def _print_header(self):
        """Print dashboard header."""
        print("\n" + "="*70)
        print(" " * 20 + "THE CRUCIBLE DASHBOARD")
        print("="*70 + "\n")
    
    def _print_summary(self):
        """Print summary section."""
        print("📊 SUMMARY")
        print("-" * 70)
        print(f"Model ID: {self.data.get('model_id', 'N/A')}")
        print(f"Compliance Score: {self.data.get('compliance_score', 0):.2%}")
        print(f"Risk Level: {self.data.get('risk_level', 'unknown').upper()}")
        print(f"Total Vulnerabilities: {len(self.data.get('vulnerabilities', []))}")
        print()
    
    def _print_vulnerabilities(self):
        """Print vulnerabilities section."""
        print("🔒 DETECTED VULNERABILITIES")
        print("-" * 70)
        
        vulns = self.data.get('vulnerabilities', [])
        if not vulns:
            print("✅ No vulnerabilities detected!")
        else:
            for i, vuln in enumerate(vulns, 1):
                severity = vuln.get('severity', 'unknown')
                icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                print(f"{icon} {i}. {vuln.get('type', 'Unknown')} "
                      f"[{severity.upper()}] - {vuln.get('location', 'N/A')}")
        print()
    
    def _print_recommendations(self):
        """Print recommendations section."""
        print("💡 RECOMMENDATIONS")
        print("-" * 70)
        
        recs = self.data.get('recommendations', [])
        if not recs:
            print("No specific recommendations.")
        else:
            for i, rec in enumerate(recs, 1):
                print(f"{i}. {rec}")
        print()
    
    def _print_footer(self):
        """Print dashboard footer."""
        print("="*70)
        print(f"Report generated: {self.data.get('timestamp', 'N/A')}")
        print("="*70 + "\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="The Crucible Dashboard - Visualize AI security analysis"
    )
    parser.add_argument(
        '--report',
        required=True,
        help='Path to the JSON report file'
    )
    
    args = parser.parse_args()
    
    dashboard = CrucibleDashboard(args.report)
    dashboard.display()


if __name__ == '__main__':
    main()
