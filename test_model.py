"""
Test script for LoRA baseline model.
Run this to verify model initialization and prediction works.

Usage:
    python test_model.py
"""

from src.phase1_baseline.model import LoRABaselineModel
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_model():
    """Test LoRA baseline model."""

    print("="*70)
    print("🧪 TESTING LORA BASELINE MODEL")
    print("="*70 + "\n")

    # Initialize model
    print("1️⃣ Initializing model (this may take a minute)...")
    try:
        model = LoRABaselineModel(
            model_name="gpt2",  # Small model for testing
            lora_r=8,
            lora_alpha=16
        )
        print("  ✓ Model initialized\n")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    # Test single prediction
    print("2️⃣ Testing single prediction...")
    question = "A patient with chest pain and ST elevation on ECG. Diagnosis?"
    options = {
        'A': 'STEMI',
        'B': 'NSTEMI',
        'C': 'Angina',
        'D': 'Pericarditis'
    }

    try:
        result = model.predict(question, options)
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("  ✓ Single prediction works\n")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    # Test batch prediction
    print("3️⃣ Testing batch prediction...")
    samples = [
        {
            'question': 'Patient with fever. Treatment?',
            'options': {'A': 'Antibiotics', 'B': 'Antivirals', 'C': 'Rest', 'D': 'Surgery'}
        },
        {
            'question': 'Elevated blood glucose. Diagnosis?',
            'options': {'A': 'Diabetes', 'B': 'Hypoglycemia', 'C': 'Normal', 'D': 'Prediabetes'}
        }
    ]

    try:
        batch_results = model.predict_batch(samples)
        print(f"  Predictions: {batch_results['predictions']}")
        print("  ✓ Batch prediction works\n")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    # Test save
    print("4️⃣ Testing model save...")
    try:
        model.save_model("outputs/test_model")
        print("  ✓ Model save works\n")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    print("="*70)
    print("✅ ALL MODEL TESTS PASSED!")
    print("="*70)
    print("\n💡 Model ready for Phase 1 training! 🚀")

    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
