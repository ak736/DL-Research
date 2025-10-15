"""
Evaluation Metrics for Fisher-Guided LoRA Project
Phase 1 Foundation - Used by ALL phases

Metrics:
- Accuracy: Percentage of correct predictions
- ECE: Expected Calibration Error (confidence vs accuracy gap)
- Brier Score: Quality of probability predictions

Output Format (Standardized for all phases):
{
    'accuracy': float,
    'ece': float,
    'brier_score': float,
    'num_samples': int,
    'predictions': List[str],    # Optional
    'confidences': List[float]   # Optional
}
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MetricsCalculator:
    """
    Calculate evaluation metrics for medical QA models.

    Key Features:
    - Accuracy calculation
    - Expected Calibration Error (ECE)
    - Brier score
    - Save/load results in standardized JSON format
    """

    def __init__(self, num_bins: int = 10):
        """
        Initialize metrics calculator.

        Args:
            num_bins: Number of bins for ECE calculation (default: 10)
        """
        self.num_bins = num_bins

    def calculate_accuracy(self,
                           predictions: List[str],
                           targets: List[str]) -> float:
        """
        Calculate accuracy.

        Args:
            predictions: List of predicted answers ('A', 'B', 'C', 'D')
            targets: List of ground truth answers

        Returns:
            Accuracy as float between 0 and 1
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")

        correct = sum(p == t for p, t in zip(predictions, targets))
        accuracy = correct / len(predictions)

        return accuracy

    def calculate_ece(self,
                      predictions: List[str],
                      targets: List[str],
                      confidences: List[float]) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures the gap between model confidence and actual accuracy.
        Lower is better (< 0.08 is good for medical AI).

        Args:
            predictions: List of predicted answers
            targets: List of ground truth answers  
            confidences: List of confidence scores (0-1)

        Returns:
            ECE score (0 = perfectly calibrated, 1 = worst)
        """
        if not (len(predictions) == len(targets) == len(confidences)):
            raise ValueError("All inputs must have same length")

        # Convert to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        confidences = np.array(confidences)

        # Create bins
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                # Calculate accuracy in this bin
                accuracy_in_bin = np.mean(
                    predictions[in_bin] == targets[in_bin])

                # Calculate average confidence in this bin
                avg_confidence_in_bin = np.mean(confidences[in_bin])

                # Add weighted difference to ECE
                ece += np.abs(avg_confidence_in_bin -
                              accuracy_in_bin) * prop_in_bin

        return float(ece)

    def calculate_brier_score(self,
                              predictions: List[str],
                              targets: List[str],
                              probabilities: List[Dict[str, float]]) -> float:
        """
        Calculate Brier Score.

        Measures quality of probabilistic predictions.
        Lower is better (0 = perfect, 1 = worst).

        Args:
            predictions: List of predicted answers (not used, but kept for consistency)
            targets: List of ground truth answers
            probabilities: List of dicts with probabilities for each option
                          e.g., [{'A': 0.7, 'B': 0.2, 'C': 0.05, 'D': 0.05}, ...]

        Returns:
            Brier score (lower is better)
        """
        if len(targets) != len(probabilities):
            raise ValueError("Targets and probabilities must have same length")

        brier_sum = 0.0

        for target, probs in zip(targets, probabilities):
            # For each option, calculate (predicted_prob - true_prob)^2
            for option, prob in probs.items():
                true_prob = 1.0 if option == target else 0.0
                brier_sum += (prob - true_prob) ** 2

        # Average over all samples
        brier_score = brier_sum / len(targets)

        return float(brier_score)

    def calculate_all_metrics(self,
                              predictions: List[str],
                              targets: List[str],
                              confidences: List[float],
                              probabilities: Optional[List[Dict[str, float]]] = None) -> Dict:
        """
        Calculate all metrics at once.

        Args:
            predictions: Predicted answers
            targets: Ground truth answers
            confidences: Confidence scores (max probability)
            probabilities: Full probability distributions (optional, for Brier)

        Returns:
            Dictionary with all metrics
        """
        results = {
            'accuracy': self.calculate_accuracy(predictions, targets),
            'ece': self.calculate_ece(predictions, targets, confidences),
            'num_samples': len(predictions)
        }

        # Add Brier score if probabilities provided
        if probabilities is not None:
            results['brier_score'] = self.calculate_brier_score(
                predictions, targets, probabilities
            )

        return results

    def calculate_forgetting(self,
                             before_accuracy: float,
                             after_accuracy: float) -> float:
        """
        Calculate catastrophic forgetting metric.

        Forgetting = how much performance dropped on old tasks.

        Args:
            before_accuracy: Accuracy before fine-tuning
            after_accuracy: Accuracy after fine-tuning

        Returns:
            Forgetting amount (0 = no forgetting, positive = forgot)
        """
        forgetting = before_accuracy - after_accuracy
        return float(max(0, forgetting))  # Clamp to 0 minimum

    def save_results(self,
                     results: Dict,
                     filepath: str):
        """
        Save results to JSON file.

        Args:
            results: Dictionary with metrics
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 Saved results to {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """
        Load results from JSON file.

        Args:
            filepath: Path to results file

        Returns:
            Dictionary with metrics
        """
        with open(filepath, 'r') as f:
            results = json.load(f)

        print(f"📂 Loaded results from {filepath}")
        return results

    def print_results(self, results: Dict, title: str = "Results"):
        """
        Print results in a nice format.

        Args:
            results: Dictionary with metrics
            title: Title to display
        """
        print("\n" + "="*60)
        print(f"📊 {title}")
        print("="*60)

        # Print main metrics
        if 'accuracy' in results:
            print(f"  Accuracy:     {results['accuracy']:.2%}")

        if 'ece' in results:
            ece_status = "✅" if results['ece'] < 0.08 else "⚠️"
            print(f"  ECE:          {results['ece']:.4f} {ece_status}")

        if 'brier_score' in results:
            print(f"  Brier Score:  {results['brier_score']:.4f}")

        if 'forgetting' in results:
            forget_status = "✅" if results['forgetting'] < 0.15 else "⚠️"
            print(
                f"  Forgetting:   {results['forgetting']:.2%} {forget_status}")

        if 'num_samples' in results:
            print(f"  Samples:      {results['num_samples']}")

        print("="*60 + "\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_evaluate(predictions: List[str],
                   targets: List[str],
                   confidences: List[float],
                   probabilities: Optional[List[Dict[str, float]]] = None,
                   save_path: Optional[str] = None) -> Dict:
    """
    Quick evaluation with all metrics.

    Args:
        predictions: Predicted answers
        targets: Ground truth answers
        confidences: Confidence scores
        probabilities: Full probability distributions (optional)
        save_path: Path to save results (optional)

    Returns:
        Dictionary with all metrics
    """
    calc = MetricsCalculator()
    results = calc.calculate_all_metrics(
        predictions, targets, confidences, probabilities)

    # Print results
    calc.print_results(results)

    # Save if path provided
    if save_path:
        calc.save_results(results, save_path)

    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Test metrics calculator.
    """
    print("🔬 Testing Metrics Calculator\n")

    # Example data
    predictions = ['A', 'B', 'C', 'A', 'D', 'B', 'C', 'A', 'B', 'A']
    targets = ['A', 'B', 'C', 'B', 'D', 'B', 'A', 'A', 'B', 'A']
    confidences = [0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.95, 0.88, 0.92]

    # Probability distributions
    probabilities = [
        {'A': 0.9, 'B': 0.05, 'C': 0.03, 'D': 0.02},
        {'A': 0.1, 'B': 0.8, 'C': 0.05, 'D': 0.05},
        {'A': 0.1, 'B': 0.1, 'C': 0.7, 'D': 0.1},
        {'A': 0.6, 'B': 0.2, 'C': 0.1, 'D': 0.1},
        {'A': 0.05, 'B': 0.05, 'C': 0.05, 'D': 0.85},
        {'A': 0.1, 'B': 0.75, 'C': 0.1, 'D': 0.05},
        {'A': 0.65, 'B': 0.2, 'C': 0.1, 'D': 0.05},
        {'A': 0.95, 'B': 0.02, 'C': 0.02, 'D': 0.01},
        {'A': 0.05, 'B': 0.88, 'C': 0.05, 'D': 0.02},
        {'A': 0.92, 'B': 0.04, 'C': 0.02, 'D': 0.02},
    ]

    # Calculate metrics
    calc = MetricsCalculator()

    print("1️⃣ Testing individual metrics:")
    accuracy = calc.calculate_accuracy(predictions, targets)
    print(f"  Accuracy: {accuracy:.2%}")

    ece = calc.calculate_ece(predictions, targets, confidences)
    print(f"  ECE: {ece:.4f}")

    brier = calc.calculate_brier_score(predictions, targets, probabilities)
    print(f"  Brier Score: {brier:.4f}\n")

    print("2️⃣ Testing calculate_all_metrics:")
    results = calc.calculate_all_metrics(
        predictions, targets, confidences, probabilities)
    calc.print_results(results, "Test Results")

    print("3️⃣ Testing forgetting calculation:")
    before_acc = 0.85
    after_acc = 0.72
    forgetting = calc.calculate_forgetting(before_acc, after_acc)
    print(f"  Before: {before_acc:.2%}")
    print(f"  After:  {after_acc:.2%}")
    print(f"  Forgetting: {forgetting:.2%}\n")

    print("4️⃣ Testing save/load:")
    results['forgetting'] = forgetting
    calc.save_results(results, "outputs/test_metrics.json")
    loaded = calc.load_results("outputs/test_metrics.json")
    print(f"  Loaded accuracy: {loaded['accuracy']:.2%}\n")

    print("✅ Metrics calculator test complete!")
