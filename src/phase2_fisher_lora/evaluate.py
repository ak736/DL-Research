"""
Evaluation Functions for Phase 1 Baseline
Evaluates LoRA model on test data and calculates metrics.

Used by: train_baseline.py and all other phases
"""

from src.phase1_baseline.model import LoRABaselineModel
from src.utils.metrics import MetricsCalculator
from src.utils.data_loader import MedQADataLoader
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def evaluate_on_dataset(model: LoRABaselineModel,
                        samples: List[Dict],
                        dataset_name: str = "test") -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: LoRA model to evaluate
        samples: List of samples with 'question', 'options', 'answer'
        dataset_name: Name for logging (e.g., "general_medicine_test")

    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*70}")
    print(f"🔍 Evaluating on {dataset_name}")
    print(f"{'='*70}")
    print(f"Samples: {len(samples)}")

    # Get predictions from model
    batch_results = model.predict_batch(samples)

    # Extract ground truth
    targets = [s['answer'] for s in samples]

    # Calculate metrics
    calc = MetricsCalculator()
    metrics = calc.calculate_all_metrics(
        predictions=batch_results['predictions'],
        targets=targets,
        confidences=batch_results['confidences'],
        probabilities=batch_results['probabilities']
    )

    # Print results
    calc.print_results(metrics, dataset_name)

    # Add dataset name to results
    metrics['dataset'] = dataset_name
    metrics['predictions'] = batch_results['predictions']
    metrics['confidences'] = batch_results['confidences']
    metrics['probabilities'] = batch_results['probabilities']

    return metrics


def evaluate_general_and_cardiology(model: LoRABaselineModel,
                                    data_dir: str = "data") -> Dict:
    """
    Evaluate model on both general medicine and cardiology test sets.

    This is the main evaluation function used in training.

    Args:
        model: LoRA model to evaluate
        data_dir: Directory with data files

    Returns:
        Dictionary with results for both domains
    """
    loader = MedQADataLoader(data_dir=data_dir)

    # Load test data
    print("\n📂 Loading test data...")
    gen_med_test = loader.load_data('general_medicine_test.json')
    cardiology_test = loader.load_data('cardiology_test.json')

    # Evaluate on both
    gen_med_results = evaluate_on_dataset(
        model,
        gen_med_test,
        "General Medicine Test"
    )

    cardiology_results = evaluate_on_dataset(
        model,
        cardiology_test,
        "Cardiology Test"
    )

    # Package results
    results = {
        'general_medicine': {
            'accuracy': gen_med_results['accuracy'],
            'ece': gen_med_results['ece'],
            'brier_score': gen_med_results['brier_score'],
            'num_samples': gen_med_results['num_samples']
        },
        'cardiology': {
            'accuracy': cardiology_results['accuracy'],
            'ece': cardiology_results['ece'],
            'brier_score': cardiology_results['brier_score'],
            'num_samples': cardiology_results['num_samples']
        }
    }

    return results


def compare_before_after(before_results: Dict,
                         after_results: Dict) -> Dict:
    """
    Compare results before and after training.

    Calculates forgetting metric.

    Args:
        before_results: Results before training
        after_results: Results after training

    Returns:
        Dictionary with comparison metrics
    """
    calc = MetricsCalculator()

    # Calculate forgetting on general medicine
    forgetting = calc.calculate_forgetting(
        before_accuracy=before_results['general_medicine']['accuracy'],
        after_accuracy=after_results['general_medicine']['accuracy']
    )

    comparison = {
        'before_training': before_results,
        'after_training': after_results,
        'forgetting': forgetting,
        'cardiology_improvement': (
            after_results['cardiology']['accuracy'] -
            before_results['cardiology']['accuracy']
        )
    }

    # Print comparison
    print("\n" + "="*70)
    print("📊 BEFORE vs AFTER TRAINING COMPARISON")
    print("="*70)
    print("\nGeneral Medicine:")
    print(f"  Before: {before_results['general_medicine']['accuracy']:.2%}")
    print(f"  After:  {after_results['general_medicine']['accuracy']:.2%}")
    print(
        f"  Forgetting: {forgetting:.2%} {'❌ HIGH' if forgetting > 0.10 else '✅ LOW'}")

    print("\nCardiology:")
    print(f"  Before: {before_results['cardiology']['accuracy']:.2%}")
    print(f"  After:  {after_results['cardiology']['accuracy']:.2%}")
    print(
        f"  Improvement: {comparison['cardiology_improvement']:.2%} {'✅' if comparison['cardiology_improvement'] > 0 else '❌'}")

    print("\n" + "="*70 + "\n")

    return comparison


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Test evaluation functions.
    """
    print("🔬 Testing Evaluation Functions\n")

    # Initialize model
    print("1️⃣ Loading model...")
    model = LoRABaselineModel(model_name="gpt2")

    # Load small test data
    print("\n2️⃣ Loading test data...")
    loader = MedQADataLoader(data_dir="data")
    gen_med_test = loader.load_data('general_medicine_test.json')[
        :5]  # Just 5 samples

    # Evaluate
    print("\n3️⃣ Running evaluation...")
    results = evaluate_on_dataset(model, gen_med_test, "Test Dataset")

    print(f"\n✅ Evaluation test complete!")
    print(f"Results: {results['accuracy']:.2%} accuracy")
