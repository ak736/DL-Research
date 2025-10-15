"""
Test script for metrics.py
Run this to verify metrics calculation works correctly.

Usage:
    python test_metrics.py
"""

from src.utils.metrics import MetricsCalculator, quick_evaluate
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_metrics():
    """Test all metrics functionality."""

    print("="*70)
    print("🧪 TESTING METRICS CALCULATOR")
    print("="*70 + "\n")

    # Example predictions (simulating model output)
    predictions = ['A', 'B', 'C', 'A', 'D', 'B', 'C', 'A', 'B', 'A']
    targets = ['A', 'B', 'C', 'B', 'D', 'B', 'A', 'A', 'B', 'A']
    confidences = [0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.95, 0.88, 0.92]

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

    print("1️⃣ Testing quick_evaluate function...")
    results = quick_evaluate(
        predictions=predictions,
        targets=targets,
        confidences=confidences,
        probabilities=probabilities,
        save_path="outputs/test_results.json"
    )

    print("2️⃣ Verifying results...")
    assert 'accuracy' in results, "Missing accuracy"
    assert 'ece' in results, "Missing ECE"
    assert 'brier_score' in results, "Missing Brier score"
    print("  ✓ All metrics calculated\n")

    print("3️⃣ Testing forgetting calculation...")
    calc = MetricsCalculator()
    forgetting = calc.calculate_forgetting(
        before_accuracy=0.85, after_accuracy=0.72)
    print(f"  Forgetting: {forgetting:.2%}")
    assert forgetting == 0.13, "Forgetting calculation incorrect"
    print("  ✓ Forgetting calculation correct\n")

    print("="*70)
    print("✅ ALL METRICS TESTS PASSED!")
    print("="*70)
    print("\n💡 Metrics ready for Phase 1 training! 🚀")

    return True


if __name__ == "__main__":
    success = test_metrics()
    sys.exit(0 if success else 1)
