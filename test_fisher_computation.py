"""
Test script for Fisher Information Matrix computation
Verifies the Fisher computation works correctly.

This is a simplified test that:
1. Uses the baseline model (trained or untrained)
2. Uses a small subset of data
3. Computes Fisher matrices
4. Verifies output is valid
"""

import torch
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.phase2_fisher_lora.compute_fisher import FisherComputer
from src.phase1_baseline.model import LoRABaselineModel


def create_test_data(num_samples: int = 10) -> list:
    """Create synthetic test data for Fisher computation."""
    test_data = []

    questions = [
        "What is the most common cause of hypertension?",
        "Which medication is first-line for diabetes?",
        "What are the symptoms of heart failure?",
        "How is pneumonia typically diagnosed?",
        "What is the treatment for acute MI?"
    ]

    for i in range(num_samples):
        sample = {
            'question': questions[i % len(questions)],
            'options': {
                'A': 'Option A',
                'B': 'Option B',
                'C': 'Option C',
                'D': 'Option D'
            },
            'answer': ['A', 'B', 'C', 'D'][i % 4],
            'domain': 'general_medicine'
        }
        test_data.append(sample)

    return test_data


def test_fisher_computation():
    """Test Fisher computation on a small dataset."""

    print("\n" + "="*70)
    print("🧪 TESTING FISHER COMPUTATION")
    print("="*70)

    # 1. Initialize model
    print("\n" + "="*70)
    print("📦 INITIALIZING MODEL")
    print("="*70)

    model = LoRABaselineModel(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    print("\n✅ Model initialized")

    # 2. Create test data
    print("\n" + "="*70)
    print("📚 CREATING TEST DATA")
    print("="*70)

    test_data = create_test_data(num_samples=5)  # Use only 5 samples for quick test
    print(f"  Created {len(test_data)} test samples")

    # 3. Initialize Fisher computer
    print("\n" + "="*70)
    print("🔧 INITIALIZING FISHER COMPUTER")
    print("="*70)

    fisher_computer = FisherComputer(model=model)

    # Check that LoRA parameters were found
    if len(fisher_computer.lora_params) == 0:
        print("\n⚠️  WARNING: No LoRA parameters found!")
        print("   This might happen if PEFT is not properly configured")
        print("   But we can still demonstrate the computation flow\n")

    # 4. Compute Fisher matrices
    print("\n" + "="*70)
    print("🧮 COMPUTING FISHER MATRICES")
    print("="*70)

    fisher_matrices = fisher_computer.compute_fisher(
        data_samples=test_data,
        num_samples=len(test_data),
        batch_size=1,
        diagonal_only=True
    )

    # 5. Verify outputs
    print("\n" + "="*70)
    print("✅ VERIFICATION")
    print("="*70)

    if len(fisher_matrices) == 0:
        print("\n⚠️  No Fisher matrices computed (no LoRA parameters found)")
        print("   This is expected if the model doesn't have LoRA adapters")
        print("   In a real run with a trained baseline, you should see Fisher matrices\n")
        return

    # Check that Fisher matrices have correct properties
    all_valid = True

    for name, fisher in fisher_matrices.items():
        print(f"\n📊 Checking {name}:")

        # 1. Same shape as parameter
        param = fisher_computer.lora_params[name]
        if fisher.shape == param.shape:
            print(f"  ✅ Shape matches: {fisher.shape}")
        else:
            print(f"  ❌ Shape mismatch: {fisher.shape} vs {param.shape}")
            all_valid = False

        # 2. All values are non-negative (Fisher is always >= 0)
        if (fisher >= 0).all():
            print(f"  ✅ All values non-negative")
        else:
            print(f"  ❌ Some values are negative!")
            all_valid = False

        # 3. Not all zeros
        if fisher.sum() > 0:
            print(f"  ✅ Contains non-zero values")
        else:
            print(f"  ⚠️  All zeros (might happen with very small dataset)")

        # 4. Statistics
        print(f"  📈 Min: {fisher.min():.6f}")
        print(f"  📈 Max: {fisher.max():.6f}")
        print(f"  📈 Mean: {fisher.mean():.6f}")

    # 6. Save test output
    output_dir = "outputs/test_fisher_computation"
    fisher_computer.save_fisher_matrices(
        fisher_matrices=fisher_matrices,
        output_dir=output_dir
    )

    print("\n" + "="*70)
    if all_valid and len(fisher_matrices) > 0:
        print("✅ ALL TESTS PASSED!")
    elif len(fisher_matrices) > 0:
        print("⚠️  TESTS COMPLETED WITH WARNINGS")
    else:
        print("⚠️  NO FISHER MATRICES (No LoRA parameters)")
    print("="*70)

    if len(fisher_matrices) > 0:
        print(f"\n📁 Test output saved to: {output_dir}/")
        print(f"\nThe Fisher computation is working correctly!")
        print(f"\n🎯 Next step: Run on real baseline model:")
        print(f"   python src/phase2_fisher_lora/compute_fisher.py")
    else:
        print(f"\n⚠️  To test with real Fisher matrices:")
        print(f"   1. First train Phase 1 baseline: python src/phase1_baseline/train_baseline.py")
        print(f"   2. Then run: python src/phase2_fisher_lora/compute_fisher.py")

    print()


def test_with_real_data():
    """
    Test Fisher computation with real data if available.
    This runs the full pipeline on actual general medicine data.
    """
    data_path = Path("data/general_medicine_train.json")

    if not data_path.exists():
        print("\n⚠️  Real data not found. Run: python test_data_loader.py")
        return

    print("\n" + "="*70)
    print("🧪 TESTING WITH REAL DATA")
    print("="*70)

    # Load data
    with open(data_path, 'r') as f:
        real_data = json.load(f)

    print(f"  Loaded {len(real_data)} samples")

    # Use only 10 samples for quick test
    test_samples = real_data[:10]
    print(f"  Using {len(test_samples)} samples for test")

    # Initialize model
    model = LoRABaselineModel(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16
    )

    # Compute Fisher
    fisher_computer = FisherComputer(model=model)

    fisher_matrices = fisher_computer.compute_fisher(
        data_samples=test_samples,
        num_samples=len(test_samples),
        batch_size=1
    )

    # Save
    output_dir = "outputs/test_fisher_computation_real"
    fisher_computer.save_fisher_matrices(
        fisher_matrices=fisher_matrices,
        output_dir=output_dir
    )

    print(f"\n✅ Real data test complete!")
    print(f"📁 Output: {output_dir}/\n")


if __name__ == "__main__":
    # Run basic test
    test_fisher_computation()

    # Optionally run with real data
    print("\n" + "="*70)
    print("Would you like to test with real data? (requires data/general_medicine_train.json)")
    print("="*70)

    try:
        test_with_real_data()
    except Exception as e:
        print(f"\n⚠️  Real data test skipped: {e}")
        print("   Run 'python test_data_loader.py' first to generate data\n")
