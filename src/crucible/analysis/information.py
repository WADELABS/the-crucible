import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class InformationBottleneck:
    """
    Layer 4: Information Bottleneck Analysis.
    Calculates mutual information between input and hidden layers 
    to see if the model is 'leaking' biased attributes into decision nodes.
    """
    
    def __init__(self):
        logging.info("Information Bottleneck Analyzer initialized.")

    def calculate_entropy(self, activations: torch.Tensor) -> float:
        """Estimate Shannon entropy of activations."""
        # Simple discretization for demo purposes
        probs = torch.histc(activations, bins=10, min=-1.0, max=1.0) / activations.numel()
        probs = probs[probs > 0]
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return entropy
    
    def calculate_mutual_information(self, X: torch.Tensor, Y: torch.Tensor,
                                    bins: int = 20) -> float:
        """
        Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
        Measures information leakage between layers or features.
        
        Args:
            X: First variable tensor
            Y: Second variable tensor
            bins: Number of bins for discretization
            
        Returns:
            Mutual information value
        """
        # Convert to numpy for histogram operations
        X_np = X.detach().cpu().numpy().flatten()
        Y_np = Y.detach().cpu().numpy().flatten()
        
        # Ensure same length
        min_len = min(len(X_np), len(Y_np))
        X_np = X_np[:min_len]
        Y_np = Y_np[:min_len]
        
        # Compute 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(X_np, Y_np, bins=bins)
        
        # Convert to probability distribution
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Calculate entropies
        # H(X)
        px_nonzero = px[px > 0]
        hx = -np.sum(px_nonzero * np.log2(px_nonzero))
        
        # H(Y)
        py_nonzero = py[py > 0]
        hy = -np.sum(py_nonzero * np.log2(py_nonzero))
        
        # H(X,Y)
        pxy_nonzero = pxy[pxy > 0]
        hxy = -np.sum(pxy_nonzero * np.log2(pxy_nonzero))
        
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        mutual_info = hx + hy - hxy
        
        logging.debug(f"Mutual Information: I(X;Y)={mutual_info:.4f}, H(X)={hx:.4f}, "
                     f"H(Y)={hy:.4f}, H(X,Y)={hxy:.4f}")
        
        return float(mutual_info)
    
    def kl_divergence(self, P: torch.Tensor, Q: torch.Tensor,
                     bins: int = 50, epsilon: float = 1e-10) -> float:
        """
        Calculate Kullback-Leibler divergence KL(P||Q).
        Measures how one probability distribution differs from another.
        Useful for detecting distribution shifts.
        
        Args:
            P: First distribution tensor
            Q: Second distribution tensor
            bins: Number of bins for histogram estimation
            epsilon: Small constant to avoid log(0)
            
        Returns:
            KL divergence value
        """
        # Convert to numpy
        P_np = P.detach().cpu().numpy().flatten()
        Q_np = Q.detach().cpu().numpy().flatten()
        
        # Compute histograms with same bin edges
        min_val = min(P_np.min(), Q_np.min())
        max_val = max(P_np.max(), Q_np.max())
        
        P_hist, edges = np.histogram(P_np, bins=bins, range=(min_val, max_val), density=True)
        Q_hist, _ = np.histogram(Q_np, bins=bins, range=(min_val, max_val), density=True)
        
        # Normalize to probability distributions
        P_prob = P_hist / (np.sum(P_hist) + epsilon)
        Q_prob = Q_hist / (np.sum(Q_hist) + epsilon)
        
        # Add epsilon to avoid log(0)
        P_prob = P_prob + epsilon
        Q_prob = Q_prob + epsilon
        
        # Calculate KL divergence: sum(P * log(P/Q))
        kl_div = np.sum(P_prob * np.log(P_prob / Q_prob))
        
        logging.debug(f"KL Divergence: KL(P||Q)={kl_div:.4f}")
        
        return float(kl_div)
    
    def quantify_privacy_leakage(self, model: nn.Module,
                                input_data: torch.Tensor,
                                sensitive_indices: List[int],
                                protected_indices: List[int],
                                bins: int = 20) -> Dict[str, float]:
        """
        Measure how much protected information leaks into model decisions.
        Calculates mutual information between sensitive features and protected attributes.
        
        Args:
            model: Neural network model
            input_data: Input dataset
            sensitive_indices: Indices of sensitive features in input
            protected_indices: Indices of protected attributes in input
            bins: Number of bins for MI calculation
            
        Returns:
            Dictionary of per-attribute leakage scores
        """
        model.eval()
        
        leakage_scores = {}
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(input_data)
            
            # For each protected attribute, measure leakage
            for prot_idx in protected_indices:
                protected_attr = input_data[:, prot_idx]
                
                # Measure mutual information with outputs
                mi_with_output = self.calculate_mutual_information(
                    protected_attr, outputs.flatten(), bins=bins
                )
                
                leakage_scores[f"protected_attr_{prot_idx}_to_output"] = mi_with_output
                
                # Measure mutual information with sensitive features
                for sens_idx in sensitive_indices:
                    sensitive_feature = input_data[:, sens_idx]
                    mi_sensitive = self.calculate_mutual_information(
                        protected_attr, sensitive_feature, bins=bins
                    )
                    leakage_scores[f"protected_{prot_idx}_to_sensitive_{sens_idx}"] = mi_sensitive
        
        # Calculate aggregate leakage score
        if leakage_scores:
            avg_leakage = sum(leakage_scores.values()) / len(leakage_scores)
            max_leakage = max(leakage_scores.values())
            
            leakage_scores["average_leakage"] = avg_leakage
            leakage_scores["max_leakage"] = max_leakage
            
            # Classify risk level
            if max_leakage > 1.0:
                risk_level = "HIGH"
            elif max_leakage > 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            leakage_scores["risk_level"] = risk_level
            
            logging.info(f"Privacy leakage assessment: avg={avg_leakage:.4f}, "
                        f"max={max_leakage:.4f}, risk={risk_level}")
        
        return leakage_scores
    
    def map_information_flow(self, model: nn.Module,
                            input_tensor: torch.Tensor,
                            layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Trace how information propagates through network layers.
        Identifies bottlenecks and information-rich pathways.
        
        Args:
            model: Neural network model
            input_tensor: Sample input
            layer_names: Optional list of layer names to analyze
            
        Returns:
            Dictionary containing information flow metrics and pathways
        """
        model.eval()
        
        # Hook to capture intermediate activations
        activations = {}
        hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for all layers
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                if layer_names is None or name in layer_names:
                    hook = module.register_forward_hook(get_activation(name))
                    hooks.append(hook)
                    layer_count += 1
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze information flow
        flow_metrics = {}
        layer_sequence = list(activations.keys())
        
        for i, layer_name in enumerate(layer_sequence):
            activation = activations[layer_name]
            
            # Calculate entropy for this layer
            entropy = self.calculate_entropy(activation)
            
            # Calculate information retention from input
            if i == 0:
                mi_from_input = self.calculate_mutual_information(
                    input_tensor.flatten(), activation.flatten()
                )
                retention_rate = 1.0
            else:
                prev_activation = activations[layer_sequence[i-1]]
                mi_from_input = self.calculate_mutual_information(
                    input_tensor.flatten(), activation.flatten()
                )
                mi_from_prev = self.calculate_mutual_information(
                    prev_activation.flatten(), activation.flatten()
                )
                
                # Retention rate: how much information is preserved
                if mi_from_input > 0:
                    retention_rate = mi_from_prev / mi_from_input if i > 0 else 1.0
                else:
                    retention_rate = 0.0
            
            # Identify bottlenecks (low entropy, low MI)
            is_bottleneck = entropy < 2.0 and mi_from_input < 0.5
            
            flow_metrics[layer_name] = {
                "entropy": float(entropy),
                "mi_from_input": float(mi_from_input),
                "retention_rate": float(retention_rate),
                "is_bottleneck": is_bottleneck,
                "activation_shape": list(activation.shape)
            }
        
        # Identify critical pathways (high MI preservation)
        critical_layers = [
            name for name, metrics in flow_metrics.items()
            if metrics["mi_from_input"] > 1.0
        ]
        
        # Find overall bottleneck layer
        bottleneck_layers = [
            name for name, metrics in flow_metrics.items()
            if metrics["is_bottleneck"]
        ]
        
        summary = {
            "total_layers_analyzed": len(flow_metrics),
            "critical_pathways": critical_layers,
            "bottleneck_layers": bottleneck_layers,
            "layer_metrics": flow_metrics,
            "information_preserved_at_output": float(
                flow_metrics[layer_sequence[-1]]["mi_from_input"] if layer_sequence else 0
            )
        }
        
        logging.info(f"Information flow mapped: {len(flow_metrics)} layers, "
                    f"{len(critical_layers)} critical pathways, "
                    f"{len(bottleneck_layers)} bottlenecks")
        
        return summary

class SymbolicExtractor:
    """
    Layer 5: Formal Logical Extraction.
    Attempts to extract discrete rules from continuous weights.
    """
    
    def __init__(self):
        logging.info("Symbolic Rule Extractor initialized.")

    def extract_logic(self, weights: torch.Tensor) -> List[str]:
        """Convert weights into simplified IF-THEN rules."""
        # Simplified: If weight > threshold, it's a decision feature
        rules = []
        threshold = 0.5
        for i, w in enumerate(weights.flatten()[:5]):
            if w > threshold:
                rules.append(f"IF feature_{i} > {threshold} THEN approve_credit")
        return rules
    
    def extract_decision_tree(self, model: nn.Module,
                             input_samples: torch.Tensor,
                             labels: torch.Tensor,
                             max_depth: int = 5) -> Dict[str, Any]:
        """
        Approximate neural network with interpretable decision tree.
        Provides model-agnostic interpretability.
        
        Args:
            model: Neural network to approximate
            input_samples: Sample inputs for training tree
            labels: True labels or model predictions
            max_depth: Maximum depth of decision tree
            
        Returns:
            Dictionary containing tree structure and fidelity score
        """
        model.eval()
        
        # Get model predictions
        with torch.no_grad():
            model_outputs = model(input_samples)
            
            # Handle different output types
            if model_outputs.dim() > 1 and model_outputs.shape[-1] > 1:
                # Classification
                model_predictions = model_outputs.argmax(dim=-1).cpu().numpy()
            else:
                # Regression or binary
                model_predictions = model_outputs.squeeze().cpu().numpy()
                # Discretize for tree
                if len(model_predictions.shape) == 0 or model_predictions.shape[0] == 1:
                    model_predictions = np.array([model_predictions])
                if np.issubdtype(model_predictions.dtype, np.floating):
                    # For regression, discretize into bins
                    model_predictions = np.digitize(
                        model_predictions,
                        bins=np.linspace(model_predictions.min(), model_predictions.max(), 5)
                    )
        
        # Convert inputs to numpy
        X = input_samples.cpu().numpy()
        if len(X.shape) > 2:
            # Flatten if needed
            X = X.reshape(X.shape[0], -1)
        
        # Train decision tree
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X, model_predictions)
        
        # Calculate fidelity (how well tree approximates model)
        tree_predictions = tree.predict(X)
        fidelity_score = accuracy_score(model_predictions, tree_predictions)
        
        # Extract tree structure
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        # Build readable rules
        def recurse_tree(node_id, depth=0):
            rules = []
            if children_left[node_id] == children_right[node_id]:
                # Leaf node
                value = tree.tree_.value[node_id]
                predicted_class = np.argmax(value)
                rules.append({
                    "type": "leaf",
                    "depth": depth,
                    "predicted_class": int(predicted_class),
                    "samples": int(tree.tree_.n_node_samples[node_id])
                })
            else:
                # Decision node
                rules.append({
                    "type": "decision",
                    "depth": depth,
                    "feature": int(feature[node_id]),
                    "threshold": float(threshold[node_id]),
                    "samples": int(tree.tree_.n_node_samples[node_id])
                })
                # Recurse left and right
                rules.extend(recurse_tree(children_left[node_id], depth + 1))
                rules.extend(recurse_tree(children_right[node_id], depth + 1))
            return rules
        
        tree_structure = recurse_tree(0)
        
        # Extract simple rules
        simple_rules = self._tree_to_rules(tree, X.shape[1])
        
        logging.info(f"Decision tree extraction: depth={tree.get_depth()}, "
                    f"n_leaves={tree.get_n_leaves()}, fidelity={fidelity_score:.3f}")
        
        return {
            "tree_depth": tree.get_depth(),
            "n_leaves": tree.get_n_leaves(),
            "n_nodes": n_nodes,
            "fidelity_score": float(fidelity_score),
            "tree_structure": tree_structure,
            "simple_rules": simple_rules,
            "feature_importances": tree.feature_importances_.tolist()
        }
    
    def _tree_to_rules(self, tree: DecisionTreeClassifier, n_features: int,
                       max_rules: int = 10) -> List[str]:
        """Convert decision tree to human-readable rules."""
        from sklearn.tree import _tree
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        tree_ = tree.tree_
        
        def recurse(node, rule_str=""):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]
                
                left_rule = f"{rule_str}({name} <= {threshold:.3f})"
                right_rule = f"{rule_str}({name} > {threshold:.3f})"
                
                yield from recurse(tree_.children_left[node], left_rule + " AND ")
                yield from recurse(tree_.children_right[node], right_rule + " AND ")
            else:
                value = tree_.value[node]
                predicted_class = np.argmax(value)
                yield f"IF {rule_str.rstrip(' AND ')} THEN class={predicted_class}"
        
        rules = list(recurse(0))
        return rules[:max_rules]
    
    def mine_rules_with_confidence(self, weights: torch.Tensor,
                                  bias: Optional[torch.Tensor] = None,
                                  threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extract IF-THEN rules with statistical confidence from model weights.
        
        Args:
            weights: Model weights tensor
            bias: Optional bias tensor
            threshold: Threshold for rule activation
            
        Returns:
            List of rules with confidence scores
        """
        rules = []
        
        # Flatten weights for analysis
        if weights.dim() > 2:
            # For conv layers, analyze filters
            weights_flat = weights.reshape(weights.shape[0], -1)
        else:
            weights_flat = weights
        
        # Analyze each output neuron
        for neuron_idx in range(min(weights_flat.shape[0], 20)):  # Limit to first 20
            neuron_weights = weights_flat[neuron_idx]
            
            # Find most important features (highest absolute weights)
            important_indices = torch.argsort(torch.abs(neuron_weights), descending=True)[:5]
            important_weights = neuron_weights[important_indices]
            
            # Calculate confidence based on weight magnitude
            weight_magnitudes = torch.abs(important_weights)
            confidence = (weight_magnitudes / (weight_magnitudes.sum() + 1e-10)).cpu().numpy()
            
            # Build rule
            conditions = []
            for feat_idx, weight, conf in zip(important_indices, important_weights, confidence):
                feat_idx = feat_idx.item()
                weight_val = weight.item()
                
                if abs(weight_val) > threshold:
                    operator = ">" if weight_val > 0 else "<"
                    conditions.append({
                        "feature": f"feature_{feat_idx}",
                        "operator": operator,
                        "threshold": threshold,
                        "weight": float(weight_val),
                        "confidence": float(conf)
                    })
            
            if conditions:
                # Build rule string
                rule_parts = [
                    f"{cond['feature']} {cond['operator']} {cond['threshold']}"
                    for cond in conditions
                ]
                rule_str = "IF (" + " AND ".join(rule_parts) + f") THEN activate_neuron_{neuron_idx}"
                
                # Calculate overall rule confidence
                overall_confidence = sum(c["confidence"] for c in conditions) / len(conditions)
                
                # Estimate support (simplified)
                support = int(100 * (1.0 - threshold))
                
                rules.append({
                    "rule": rule_str,
                    "confidence": float(overall_confidence),
                    "support": support,
                    "conditions": conditions,
                    "neuron_id": neuron_idx
                })
        
        # Sort by confidence
        rules.sort(key=lambda x: x["confidence"], reverse=True)
        
        logging.info(f"Extracted {len(rules)} rules with confidence scores")
        
        return rules
    
    def linear_approximation(self, model: nn.Module,
                           input_space: torch.Tensor,
                           target_input: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Find best linear approximation in local regions.
        Useful for explaining individual predictions (local interpretability).
        
        Args:
            model: Neural network model
            input_space: Sample points around which to approximate
            target_input: Specific input to explain (uses center if None)
            
        Returns:
            Dictionary containing linear coefficients and approximation quality
        """
        model.eval()
        
        if target_input is None:
            # Use mean of input space as target
            target_input = input_space.mean(dim=0, keepdim=True)
        
        # Get model predictions
        with torch.no_grad():
            target_output = model(target_input)
            space_outputs = model(input_space)
        
        # Flatten inputs and outputs
        X = input_space.cpu().numpy()
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        y = space_outputs.cpu().numpy()
        if len(y.shape) > 1:
            y = y.reshape(y.shape[0], -1)
        
        # Fit linear model using least squares
        from numpy.linalg import lstsq
        
        # Add bias term
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        
        # Solve for coefficients
        if y.shape[1] > 1:
            # Multi-output
            coefficients = []
            r_squared_scores = []
            
            for output_idx in range(min(y.shape[1], 10)):  # Limit outputs
                coef, residuals, rank, s = lstsq(X_with_bias, y[:, output_idx], rcond=None)
                
                # Calculate R-squared
                y_pred = X_with_bias @ coef
                ss_res = np.sum((y[:, output_idx] - y_pred) ** 2)
                ss_tot = np.sum((y[:, output_idx] - y[:, output_idx].mean()) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-10))
                
                coefficients.append(coef.tolist())
                r_squared_scores.append(float(r_squared))
        else:
            # Single output
            coef, residuals, rank, s = lstsq(X_with_bias, y.flatten(), rcond=None)
            
            # Calculate R-squared
            y_pred = X_with_bias @ coef
            ss_res = np.sum((y.flatten() - y_pred) ** 2)
            ss_tot = np.sum((y.flatten() - y.flatten().mean()) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            coefficients = [coef.tolist()]
            r_squared_scores = [float(r_squared)]
        
        # Extract feature importance from coefficients
        feature_importance = {}
        for output_idx, coef in enumerate(coefficients):
            # Exclude bias term
            feature_weights = np.abs(coef[:-1])
            top_features = np.argsort(feature_weights)[-5:][::-1]
            
            feature_importance[f"output_{output_idx}"] = [
                {
                    "feature_id": int(feat_idx),
                    "coefficient": float(coef[feat_idx]),
                    "importance": float(feature_weights[feat_idx])
                }
                for feat_idx in top_features
            ]
        
        avg_r_squared = sum(r_squared_scores) / len(r_squared_scores)
        
        # Determine approximation quality
        if avg_r_squared > 0.9:
            quality = "EXCELLENT"
        elif avg_r_squared > 0.7:
            quality = "GOOD"
        elif avg_r_squared > 0.5:
            quality = "MODERATE"
        else:
            quality = "POOR"
        
        logging.info(f"Linear approximation: R²={avg_r_squared:.3f}, quality={quality}")
        
        return {
            "coefficients": coefficients,
            "r_squared_scores": r_squared_scores,
            "average_r_squared": float(avg_r_squared),
            "approximation_quality": quality,
            "feature_importance": feature_importance,
            "n_samples_used": X.shape[0]
        }
