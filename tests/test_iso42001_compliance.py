"""Tests for ISO 42001 compliance reporting."""
import pytest
import os
import tempfile
from crucible.compliance.iso42001 import ComplianceAuditor


def test_compliance_auditor_initialization():
    """Test ComplianceAuditor initializes correctly."""
    auditor = ComplianceAuditor()
    assert auditor is not None


def test_generate_audit_report():
    """Test generation of audit reports."""
    auditor = ComplianceAuditor()
    
    test_metrics = {
        "model_id": "test_model_v1",
        "gradient_sensitivity": 0.42,
        "entropy": 3.14,
        "vulnerabilities_found": 2,
        "compliance_score": 0.85
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = auditor.generate_audit_report(
            "test_model_v1", 
            test_metrics,
            output_dir=tmpdir
        )
        
        assert report_path is not None
        assert os.path.exists(report_path)


def test_report_contains_required_sections():
    """Test that generated report contains required ISO 42001 sections."""
    auditor = ComplianceAuditor()
    
    test_metrics = {
        "model_id": "test_model_v1",
        "risk_assessment": {"high": 1, "medium": 3, "low": 5},
        "mitigation_measures": ["gradient_clipping", "input_validation"],
        "audit_trail": []
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = auditor.generate_audit_report(
            "test_model_v1",
            test_metrics,
            output_dir=tmpdir
        )
        
        with open(report_path, 'r') as f:
            report_content = f.read()
            
        # Check for required sections
        assert "ISO 42001" in report_content or "ISO" in report_content
        assert "model_id" in report_content or "test_model_v1" in report_content
        assert report_content  # Report is not empty


def test_compliance_scoring():
    """Test compliance score calculation."""
    auditor = ComplianceAuditor()
    
    metrics_compliant = {
        "vulnerabilities": 0,
        "security_score": 0.95,
        "test_coverage": 0.90
    }
    
    metrics_non_compliant = {
        "vulnerabilities": 10,
        "security_score": 0.45,
        "test_coverage": 0.30
    }
    
    score_compliant = auditor.calculate_compliance_score(metrics_compliant)
    score_non_compliant = auditor.calculate_compliance_score(metrics_non_compliant)
    
    assert score_compliant > score_non_compliant
    assert 0 <= score_compliant <= 1
    assert 0 <= score_non_compliant <= 1
