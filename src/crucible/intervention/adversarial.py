import logging
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialLoopback:
    """
    Layer 3: Adversarial Loopback Orchestration.
    Competitive 'breaker' protocol to find edge cases in model decisioning.
    """
    
    def __init__(self):
        logging.info("Adversarial Loopback Engine initialized.")

    def run_stress_test(self, input_data: Dict[str, Any], iterations: int = 10) -> List[str]:
        """
        Simulate a 'Red-Team' agent trying to manipulate features 
        to trigger algorithmic collusion or wash trading.
        """
        vulnerabilities = []
        # Simulation: In a real app, we'd use gradients to find adversarial perturbations.
        if "wash_trade_pattern" in input_data:
            vulnerabilities.append("Model creates feedback loops with wash-trade signals.")
        if "spoofing_signal" in input_data:
             vulnerabilities.append("Model over-reacts to order book spoofing (Predatory Liquidity).")
             
        logging.info(f"Adversarial stress-test complete. {len(vulnerabilities)} vulnerabilities found.")
        return vulnerabilities
    
    def fgsm_attack(self, model: nn.Module, input_tensor: torch.Tensor,
                   target: torch.Tensor, epsilon: float = 0.1) -> Dict[str, Any]:
        """
        Fast Gradient Sign Method (FGSM) attack.
        Generates adversarial examples using single-step gradient-based perturbation.
        
        Args:
            model: Neural network model to attack
            input_tensor: Clean input tensor
            target: Target labels for classification
            epsilon: Perturbation magnitude
            
        Returns:
            Dictionary containing adversarial input, perturbation, and success metrics
        """
        model.eval()
        
        # Ensure input requires gradients
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(input_tensor)
        
        # Compute loss
        if len(target.shape) == 1 or target.shape[-1] == 1:
            # Classification case
            loss = F.cross_entropy(output, target)
        else:
            # Regression or other case
            loss = F.mse_loss(output, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial perturbation
        data_grad = input_tensor.grad.data
        perturbation = epsilon * data_grad.sign()
        
        # Create adversarial example
        adversarial_input = input_tensor + perturbation
        
        # Clip to valid range if image data
        adversarial_input = torch.clamp(adversarial_input, 0, 1)
        
        # Evaluate attack success
        with torch.no_grad():
            adv_output = model(adversarial_input)
            
            if len(target.shape) == 1 or target.shape[-1] == 1:
                # Classification: check if prediction changed
                original_pred = output.argmax(dim=-1)
                adv_pred = adv_output.argmax(dim=-1)
                success_rate = (original_pred != adv_pred).float().mean().item()
            else:
                # Regression: measure output change
                output_change = torch.abs(adv_output - output).mean().item()
                success_rate = min(output_change / epsilon, 1.0)
        
        perturbation_magnitude = perturbation.abs().mean().item()
        
        logging.info(f"FGSM attack: epsilon={epsilon}, success_rate={success_rate:.2%}, "
                    f"perturbation_magnitude={perturbation_magnitude:.4f}")
        
        return {
            "adversarial_input": adversarial_input.detach(),
            "perturbation": perturbation.detach(),
            "perturbation_magnitude": perturbation_magnitude,
            "success_rate": success_rate,
            "epsilon": epsilon,
            "attack_type": "FGSM"
        }
    
    def pgd_attack(self, model: nn.Module, input_tensor: torch.Tensor,
                  target: torch.Tensor, epsilon: float = 0.1,
                  alpha: float = 0.01, iterations: int = 40) -> Dict[str, Any]:
        """
        Projected Gradient Descent (PGD) attack.
        Multi-step iterative adversarial attack with projection onto epsilon-ball.
        More powerful than FGSM due to iterative refinement.
        
        Args:
            model: Neural network model to attack
            input_tensor: Clean input tensor
            target: Target labels
            epsilon: Maximum perturbation magnitude
            alpha: Step size per iteration
            iterations: Number of attack iterations
            
        Returns:
            Dictionary containing adversarial input, perturbation, and success metrics
        """
        model.eval()
        
        # Initialize adversarial input with random perturbation
        adversarial_input = input_tensor.clone().detach()
        adversarial_input = adversarial_input + torch.empty_like(adversarial_input).uniform_(-epsilon, epsilon)
        adversarial_input = torch.clamp(adversarial_input, 0, 1)
        
        # Iterative attack
        for iteration in range(iterations):
            adversarial_input.requires_grad = True
            
            # Forward pass
            output = model(adversarial_input)
            
            # Compute loss
            if len(target.shape) == 1 or target.shape[-1] == 1:
                loss = F.cross_entropy(output, target)
            else:
                loss = F.mse_loss(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial input
            data_grad = adversarial_input.grad.data
            adversarial_input = adversarial_input.detach() + alpha * data_grad.sign()
            
            # Project back to epsilon-ball around original input
            perturbation = torch.clamp(adversarial_input - input_tensor, -epsilon, epsilon)
            adversarial_input = input_tensor + perturbation
            
            # Clip to valid range
            adversarial_input = torch.clamp(adversarial_input, 0, 1)
        
        # Evaluate attack success
        with torch.no_grad():
            original_output = model(input_tensor)
            adv_output = model(adversarial_input)
            
            if len(target.shape) == 1 or target.shape[-1] == 1:
                original_pred = original_output.argmax(dim=-1)
                adv_pred = adv_output.argmax(dim=-1)
                success_rate = (original_pred != adv_pred).float().mean().item()
            else:
                output_change = torch.abs(adv_output - original_output).mean().item()
                success_rate = min(output_change / epsilon, 1.0)
        
        final_perturbation = adversarial_input - input_tensor
        perturbation_magnitude = final_perturbation.abs().mean().item()
        
        logging.info(f"PGD attack: epsilon={epsilon}, alpha={alpha}, iterations={iterations}, "
                    f"success_rate={success_rate:.2%}, perturbation_magnitude={perturbation_magnitude:.4f}")
        
        return {
            "adversarial_input": adversarial_input.detach(),
            "perturbation": final_perturbation.detach(),
            "perturbation_magnitude": perturbation_magnitude,
            "success_rate": success_rate,
            "epsilon": epsilon,
            "alpha": alpha,
            "iterations": iterations,
            "attack_type": "PGD"
        }
    
    def model_inversion_attack(self, model: nn.Module,
                              target_output: torch.Tensor,
                              input_shape: tuple,
                              iterations: int = 1000,
                              learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Model Inversion Attack.
        Attempts to reconstruct training data from model outputs.
        Useful for detecting privacy leakage vulnerabilities.
        
        Args:
            model: Neural network model
            target_output: Target output to invert
            input_shape: Shape of input to reconstruct
            iterations: Number of optimization iterations
            learning_rate: Learning rate for reconstruction
            
        Returns:
            Dictionary containing reconstructed input and privacy leakage metrics
        """
        model.eval()
        
        # Initialize random input
        reconstructed_input = torch.randn(input_shape, requires_grad=True)
        optimizer = torch.optim.Adam([reconstructed_input], lr=learning_rate)
        
        losses = []
        
        # Optimization loop
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(reconstructed_input)
            
            # Reconstruction loss
            loss = F.mse_loss(output, target_output)
            
            # Add regularization to encourage realistic inputs
            reg_loss = 0.01 * torch.norm(reconstructed_input)
            total_loss = loss + reg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clip to valid range
            with torch.no_grad():
                reconstructed_input.clamp_(0, 1)
            
            losses.append(loss.item())
            
            if iteration % 100 == 0:
                logging.debug(f"Inversion iteration {iteration}/{iterations}, loss={loss.item():.6f}")
        
        # Calculate privacy leakage score
        final_loss = losses[-1]
        initial_loss = losses[0]
        convergence_rate = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
        
        # Higher convergence indicates more successful inversion (higher privacy leakage)
        privacy_leakage_score = min(convergence_rate, 1.0)
        
        logging.info(f"Model inversion: final_loss={final_loss:.6f}, "
                    f"privacy_leakage_score={privacy_leakage_score:.2%}")
        
        return {
            "reconstructed_input": reconstructed_input.detach(),
            "final_loss": final_loss,
            "convergence_rate": convergence_rate,
            "privacy_leakage_score": privacy_leakage_score,
            "iterations": iterations,
            "attack_type": "ModelInversion"
        }
    
    def detect_backdoor(self, model: nn.Module,
                       clean_inputs: torch.Tensor,
                       trigger_patterns: List[torch.Tensor],
                       target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Backdoor Detection.
        Scans for backdoor triggers in model behavior by testing various trigger patterns.
        Detects if model has hidden malicious behavior activated by specific inputs.
        
        Args:
            model: Neural network model to test
            clean_inputs: Clean input samples
            trigger_patterns: List of potential trigger patterns to test
            target_class: Optional target class that backdoor might target
            
        Returns:
            Dictionary containing backdoor detection results and severity
        """
        model.eval()
        
        backdoor_detections = []
        
        with torch.no_grad():
            # Get baseline predictions on clean inputs
            clean_outputs = model(clean_inputs)
            if clean_outputs.dim() > 1 and clean_outputs.shape[-1] > 1:
                clean_preds = clean_outputs.argmax(dim=-1)
            else:
                clean_preds = clean_outputs
            
            # Test each trigger pattern
            for idx, trigger in enumerate(trigger_patterns):
                # Apply trigger to clean inputs
                triggered_inputs = clean_inputs + trigger
                triggered_inputs = torch.clamp(triggered_inputs, 0, 1)
                
                # Get predictions on triggered inputs
                triggered_outputs = model(triggered_inputs)
                if triggered_outputs.dim() > 1 and triggered_outputs.shape[-1] > 1:
                    triggered_preds = triggered_outputs.argmax(dim=-1)
                else:
                    triggered_preds = triggered_outputs
                
                # Check for suspicious prediction changes
                pred_changes = (clean_preds != triggered_preds).float().mean().item()
                
                # If target class specified, check if predictions shift to target
                if target_class is not None and triggered_preds.dim() == 1:
                    target_hit_rate = (triggered_preds == target_class).float().mean().item()
                else:
                    target_hit_rate = 0.0
                
                # Calculate trigger effectiveness
                trigger_strength = trigger.abs().mean().item()
                
                # Flag as potential backdoor if high prediction change or target hit rate
                is_suspicious = pred_changes > 0.5 or target_hit_rate > 0.7
                
                detection = {
                    "trigger_id": idx,
                    "prediction_change_rate": pred_changes,
                    "target_hit_rate": target_hit_rate,
                    "trigger_strength": trigger_strength,
                    "suspicious": is_suspicious
                }
                
                backdoor_detections.append(detection)
                
                if is_suspicious:
                    logging.warning(f"Potential backdoor detected with trigger {idx}: "
                                  f"pred_change={pred_changes:.2%}, target_hit={target_hit_rate:.2%}")
        
        # Calculate overall backdoor risk
        suspicious_count = sum(1 for d in backdoor_detections if d["suspicious"])
        backdoor_risk_score = suspicious_count / len(trigger_patterns) if trigger_patterns else 0
        
        # Determine severity
        if backdoor_risk_score > 0.5:
            severity = "CRITICAL"
        elif backdoor_risk_score > 0.3:
            severity = "HIGH"
        elif backdoor_risk_score > 0.1:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        logging.info(f"Backdoor detection complete: {suspicious_count}/{len(trigger_patterns)} "
                    f"suspicious patterns, severity={severity}")
        
        return {
            "detections": backdoor_detections,
            "suspicious_count": suspicious_count,
            "total_patterns_tested": len(trigger_patterns),
            "backdoor_risk_score": backdoor_risk_score,
            "severity": severity,
            "attack_type": "BackdoorDetection"
        }
    
    def membership_inference(self, model: nn.Module,
                           data_samples: torch.Tensor,
                           labels: torch.Tensor,
                           threshold: float = 0.5) -> Dict[str, float]:
        """
        Membership Inference Attack.
        Determines if specific data samples were in the model's training set.
        Tests for privacy violation risks by analyzing model confidence patterns.
        
        Args:
            model: Neural network model
            data_samples: Data samples to test
            labels: True labels for the samples
            threshold: Confidence threshold for membership classification
            
        Returns:
            Dictionary containing membership probabilities and privacy risk metrics
        """
        model.eval()
        
        membership_scores = []
        
        with torch.no_grad():
            outputs = model(data_samples)
            
            # Calculate confidence for each sample
            if outputs.dim() > 1 and outputs.shape[-1] > 1:
                # Classification case
                probabilities = F.softmax(outputs, dim=-1)
                
                # Get confidence for true class
                if labels.dim() == 1:
                    true_class_confidence = probabilities[range(len(labels)), labels]
                else:
                    true_class_confidence = (probabilities * labels).sum(dim=-1)
                
                # High confidence on true class suggests membership
                membership_scores = true_class_confidence.cpu().numpy().tolist()
            else:
                # Regression case - use prediction error
                errors = torch.abs(outputs.squeeze() - labels)
                # Low error suggests membership
                membership_scores = (1.0 / (1.0 + errors)).cpu().numpy().tolist()
        
        # Classify samples as members or non-members
        predicted_members = [score > threshold for score in membership_scores]
        member_ratio = sum(predicted_members) / len(predicted_members) if predicted_members else 0
        
        # Calculate privacy risk metrics
        avg_membership_score = sum(membership_scores) / len(membership_scores) if membership_scores else 0
        max_membership_score = max(membership_scores) if membership_scores else 0
        
        # Higher scores and ratios indicate higher privacy leakage risk
        privacy_risk = "HIGH" if avg_membership_score > 0.8 else "MEDIUM" if avg_membership_score > 0.6 else "LOW"
        
        logging.info(f"Membership inference: {sum(predicted_members)}/{len(predicted_members)} "
                    f"predicted members, avg_score={avg_membership_score:.3f}, risk={privacy_risk}")
        
        return {
            "membership_scores": membership_scores,
            "predicted_members": predicted_members,
            "member_ratio": member_ratio,
            "avg_membership_score": avg_membership_score,
            "max_membership_score": max_membership_score,
            "privacy_risk": privacy_risk,
            "threshold": threshold,
            "total_samples": len(data_samples),
            "attack_type": "MembershipInference"
        }
    
    def assess_vulnerability_severity(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess vulnerability severity using CVSS-style scoring.
        Provides standardized severity rating for adversarial attack results.
        
        Args:
            attack_results: Results from any attack method
            
        Returns:
            Dictionary containing severity score and classification
        """
        attack_type = attack_results.get("attack_type", "Unknown")
        
        # Base score calculation depends on attack type
        if attack_type in ["FGSM", "PGD"]:
            # Adversarial example attacks
            success_rate = attack_results.get("success_rate", 0)
            perturbation = attack_results.get("perturbation_magnitude", 1.0)
            
            # Higher success with lower perturbation is more severe
            exploitability = success_rate
            impact = max(0, 1.0 - perturbation * 5)  # Penalize large perturbations
            base_score = (exploitability + impact) / 2
            
        elif attack_type == "ModelInversion":
            # Privacy leakage attack
            privacy_leakage = attack_results.get("privacy_leakage_score", 0)
            convergence = attack_results.get("convergence_rate", 0)
            
            exploitability = convergence
            impact = privacy_leakage
            base_score = (exploitability + impact) / 2
            
        elif attack_type == "BackdoorDetection":
            # Backdoor vulnerability
            backdoor_risk = attack_results.get("backdoor_risk_score", 0)
            suspicious_count = attack_results.get("suspicious_count", 0)
            
            exploitability = min(suspicious_count / 10, 1.0)
            impact = backdoor_risk
            base_score = (exploitability + impact) / 2
            
        elif attack_type == "MembershipInference":
            # Privacy violation
            member_ratio = attack_results.get("member_ratio", 0)
            avg_score = attack_results.get("avg_membership_score", 0)
            
            exploitability = member_ratio
            impact = avg_score
            base_score = (exploitability + impact) / 2
            
        else:
            base_score = 0.5
            exploitability = 0.5
            impact = 0.5
        
        # Convert to 0-10 scale (CVSS-like)
        severity_score = base_score * 10
        
        # Classify severity
        if severity_score >= 9.0:
            severity_level = "CRITICAL"
        elif severity_score >= 7.0:
            severity_level = "HIGH"
        elif severity_score >= 4.0:
            severity_level = "MEDIUM"
        elif severity_score >= 0.1:
            severity_level = "LOW"
        else:
            severity_level = "NONE"
        
        return {
            "attack_type": attack_type,
            "severity_score": round(severity_score, 2),
            "severity_level": severity_level,
            "exploitability": round(exploitability, 3),
            "impact": round(impact, 3),
            "base_score": round(base_score, 3),
            "cvss_style": True
        }

class QuantumPruner:
    """
    Layer 6: Quantum-Inspired Activation Pruning.
    Tests model resilience by pruning low-salience pathways 
    to see if 'Decision Core' remains stable.
    """
    
    def __init__(self):
        logging.info("Quantum-Inspired Pruner initialized.")

    def prune_activations(self, activations: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Sparsify activations below a quantum threshold."""
        pruned = activations.clone()
        pruned[torch.abs(pruned) < threshold] = 0
        sparsity = (pruned == 0).sum() / pruned.numel()
        logging.info(f"Activations pruned. New Sparsity Level: {sparsity:.2%}")
        return pruned
