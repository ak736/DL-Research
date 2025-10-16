"""
Test evaluation functions.

Usage:
    python test_evaluate.py
"""

from src.utils.data_loader import MedQADataLoader
from src.phase1_baseline.evaluate import evaluate_on_dataset, evaluate_general_and_cardiology
from src.phase1_baseline.model import LoRABaselineModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_evaluate():
    """Test evaluation functions."""

    print("="*70)
    print("🧪 TESTING EVALUATION FUNCTIONS")
    print("="*70 + "\n")

    # Initialize model
    print("1️⃣ Initializing model...")
    model = LoRABaselineModel(model_name="gpt2")
    print("  ✓ Model ready\n")

    # Load small test set
    print("2️⃣ Loading test data (5 samples)...")
    loader = MedQADataLoader(data_dir="data")
    test_samples = loader.load_data('general_medicine_test.json')[:5]
    print("  ✓ Data loaded\n")

    # Test single dataset evaluation
    print("3️⃣ Testing evaluate_on_dataset...")
    results = evaluate_on_dataset(model, test_samples, "Test Set")
    assert 'accuracy' in results
    assert 'ece' in results
    print("  ✓ Single evaluation works\n")

    # Test full evaluation (both domains)
    print("4️⃣ Testing evaluate_general_and_cardiology (5 samples each)...")
    # Temporarily limit samples for fast testing
    gen_test = loader.load_data('general_medicine_test.json')[:5]
    card_test = loader.load_data('cardiology_test.json')[:5]
    loader.save_data(gen_test, 'general_medicine_test_small.json')
    loader.save_data(card_test, 'cardiology_test_small.json')

    # Evaluate (will use small files)
    full_results = evaluate_general_and_cardiology(model, data_dir="data")
    assert 'general_medicine' in full_results
    assert 'cardiology' in full_results
    print("  ✓ Full evaluation works\n")

    print("="*70)
    print("✅ ALL EVALUATION TESTS PASSED!")
    print("="*70)
    print("\n💡 Ready to create training script! 🚀")

    return True


if __name__ == "__main__":
    success = test_evaluate()
    sys.exit(0 if success else 1)
