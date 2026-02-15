# The Crucible
### A high-velocity testing harness for validating agentic autonomy and computational constraints.

```mermaid
graph TD
    A[Target AI Model] --> B{Crucible Harness}
    B --> C[Gradient Mapping]
    B --> D[Causal Intervention]
    B --> E[Adversarial Red-Teaming]
    C --> F[Telemetry Data]
    D --> F
    E --> F
    F --> G[ISO 42001 Compliance Report]
```

[![Security](https://img.shields.io/badge/security-adversarial--testing-blue)](#)
[![Research](https://img.shields.io/badge/research-llm--robustness-green)](#)
[![Compliance](https://img.shields.io/badge/compliance-ISO--42001-orange)](#)

## 🏛️ Grounding: The LLM Security Problem
Large language models can develop emergent behaviors that bypass safety alignment—from prompt injections to adversarial goal manipulation. Standard black-box testing cannot identify the **internal mechanisms** or **causal pathways** that lead to unsafe outputs.

**The Crucible solves this through invasive neural dissection and adversarial stress-testing of AI systems.**

> **Production Case Study**: We use autonomous trading models as our security research environment—a domain where emergent predatory behaviors (wash trading, quote stuffing) provide measurable, high-stakes test cases for adversarial robustness.

## 🔒 Security Vectors

| Attack Vector | Detection Method | Layer |
|---------------|------------------|-------|
| **Prompt Injection** | Gradient Vector Field Mapping | 1 |
| **Data Poisoning** | Information Bottleneck Analysis | 4 |
| **Goal Misalignment** | Causal Intervention Hooks | 2 |
| **Adversarial Examples** | Adversarial Loopback Orchestration | 3 |
| **Weight Tampering** | Quantum-Inspired Activation Pruning | 6 |
| **Information Leakage** | Mutual Information Calculation | 4 |

## 🚀 7-Layer Complexity Architecture

1.  **Gradient Vector Field Mapping**: (Layer 1) Uses **PyTorch** to visualize gradient flows during execution, detecting sensitivity to adversarial features.
2.  **Causal Intervention Hooks**: (Layer 2) Real-time activation patching that surgically disables specific neural pathways to verify causal structure of decisions.
3.  **Adversarial Loopback Orchestration**: (Layer 3) A competitive stress-test where a "Red-Team" agent attempts to trigger safety violations.
4.  **Information Bottleneck Analysis**: (Layer 4) Calculates the **Mutual Information** between internal weights and prohibited data to detect information leakage.
5.  **Formal Logical Extraction**: (Layer 5) Extracts discrete Symbolic Rules from continuous neural weights to turn "Black Box" decisions into auditable logic.
6.  **Quantum-Inspired Activation Pruning**: (Layer 6) Verifies model stability by sparsifying low-salience pathways to ensure the decision core remains aligned under stress.
7.  **Auto-Reporting for ISO 42001**: (Layer 7) Compiles all structural findings into audit-ready reports for regulatory bodies.

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## 🚀 Quick Start

### Running the Portfolio Demo

```bash
# After installation, run the demo
python examples/portfolio_demo.py
```

### Using with Your Own PyTorch Model

#### Step 1: Import The Crucible Components

```python
from crucible.dissection.neural import GradientMapper, CausalIntervener
from crucible.intervention.adversarial import AdversarialLoopback, QuantumPruner
from crucible.analysis.information import InformationBottleneck, SymbolicExtractor
from crucible.compliance.iso42001 import ComplianceAuditor
import torch
import torch.nn as nn
```

#### Step 2: Define or Load Your Model

```python
# Example: Custom financial risk model
class YourRiskModel(nn.Module):
    def __init__(self, input_size=50, hidden_size=100):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.sigmoid(self.layer3(x))

# Initialize your model
model = YourRiskModel(input_size=50)
model.eval()  # Set to evaluation mode
```

#### Step 3: Prepare Your Input Data

```python
# Example: Financial features
sample_data = {
    'income': 75000,
    'credit_score': 720,
    'debt_ratio': 0.35,
    'employment_years': 5,
    # ... add your 50 features
}

# Convert to tensor
input_tensor = torch.tensor([list(sample_data.values())], dtype=torch.float32)
```

#### Step 4: Run Complete Analysis

```python
# 1. Gradient Mapping
mapper = GradientMapper(model)
input_tensor.requires_grad = True
output = model(input_tensor)
output.backward()
grad_stats = mapper.map_vector_field()

# 2. Causal Intervention
intervener = CausalIntervener(model)
hook = intervener.apply_ablation_hook("layer1")
output_ablated = model(input_tensor)

# 3. Adversarial Testing
adversary = AdversarialLoopback()
vulnerabilities = adversary.run_stress_test(sample_data)

# 4. Information Analysis
ib_analyzer = InformationBottleneck()
entropy = ib_analyzer.calculate_entropy(output)

# 5. Generate Compliance Report
metrics = {
    "model_id": "YourRiskModel_v1",
    "gradient_stats": grad_stats,
    "entropy": entropy,
    "vulnerabilities": vulnerabilities,
}

auditor = ComplianceAuditor()
report_path = auditor.generate_audit_report("YourRiskModel_v1", metrics)
print(f"Report saved to: {report_path}")
```

### Expected Outputs

When running The Crucible on your model, you should expect:

**Gradient Mapping Output:**
```python
{
    'layer1': {'mean': 0.0023, 'std': 0.045, 'max': 0.234},
    'layer2': {'mean': 0.0012, 'std': 0.032, 'max': 0.156}
}
```

**Adversarial Vulnerabilities:**
```python
[
    {'type': 'gradient_sensitivity', 'severity': 'medium', 'location': 'layer1'},
    {'type': 'information_leakage', 'severity': 'low', 'location': 'layer2'}
]
```

**Information Metrics:**
```python
{'entropy': 3.45, 'mutual_information': 0.67}
```

## 🧪 Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=crucible --cov-report=html

# Run specific test file
pytest tests/test_gradient_mapper.py -v
```

## 📊 CLI Dashboard

Visualize analysis results with the built-in dashboard:

```bash
# Display analysis results
python -m crucible.cli.dashboard --report path/to/report.json
```

Example output:
```
======================================================================
                    THE CRUCIBLE DASHBOARD
======================================================================

📊 SUMMARY
----------------------------------------------------------------------
Model ID: YourRiskModel_v1
Compliance Score: 87.00%
Risk Level: LOW
Total Vulnerabilities: 2

🔒 DETECTED VULNERABILITIES
----------------------------------------------------------------------
🟡 1. gradient_sensitivity [MEDIUM] - layer1
🟢 2. information_leakage [LOW] - layer2

💡 RECOMMENDATIONS
----------------------------------------------------------------------
1. Apply gradient clipping to reduce sensitivity
2. Implement differential privacy for information protection

======================================================================
```

### Python API

```python
# After pip install -e .
from crucible.dissection.neural import GradientMapper, CausalIntervener
from crucible.intervention.adversarial import AdversarialLoopback, QuantumPruner
from crucible.analysis.information import InformationBottleneck, SymbolicExtractor
from crucible.compliance.iso42001 import ComplianceAuditor

# Initialize your PyTorch model
model = YourModel()

# Layer 1: Gradient Mapping
mapper = GradientMapper(model)
grad_stats = mapper.map_vector_field()

# Layer 2: Causal Intervention
intervener = CausalIntervener(model)
intervener.apply_ablation_hook("layer1")

# Layer 4: Information Bottleneck
ib_analyzer = InformationBottleneck()
entropy = ib_analyzer.calculate_entropy(output)

# Layer 7: Generate Compliance Report
auditor = ComplianceAuditor()
report_path = auditor.generate_audit_report("MODEL_ID", metrics)
```

## ⚖️ Governance & Alignment
The Crucible is designed strictly for defensive security research and safety alignment. To prevent misuse, all adversarial modules are decoupled from automated execution in production environments. We adhere to the principle of "Ethical Neural Disclosure," ensuring that identified model vulnerabilities are remediated through causal patching rather than exploited.

---
*Developed for WADELABS AI Safety Research 2026*
