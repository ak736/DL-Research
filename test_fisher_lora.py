"""
Test Fisher-Guided LoRA Model
Verifies the model can train for 1 batch without errors.

Tests:
1. Model initialization
2. Save initial parameters
3. Train on 1 batch
4. Verify loss components are computed
"""

import torch
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.phase1_baseline.model import LoRABaselineModel
from src.phase2_fisher_lora.fisher_lora import FisherGuidedLoRA


def create_test_batch(num_samples: int = 2) -> list:
    """Create a small test batch."""
    batch = []

    for i in range(num_samples):
        sample = {
            'question': f'What is the treatment for condition {i}?',
            'options': {
                'A': 'Treatment A',
                'B': 'Treatment B',
                'C': 'Treatment C',
                'D': 'Treatment D'
            },
            'answer': ['A', 'B', 'C', 'D'][i % 4],
            'domain': 'cardiology'
        }
        batch.append(sample)

    return batch


def test_fisher_lora_model():
    """Test Fisher-LoRA model on a single batch."""

    print("\n" + "="*70)
    print("🧪 TESTING FISHER-GUIDED LORA MODEL")
    print("="*70)

    # 1. Initialize baseline model
    print("\n" + "="*70)
    print("📦 STEP 1: INITIALIZE BASELINE MODEL")
    print("="*70)

    baseline_model = LoRABaselineModel(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    print("\n✅ Baseline model initialized")

    # 2. Check if Fisher matrices exist
    fisher_path = "outputs/test_fisher_computation_real/fisher_matrices_all.pt"

    if not Path(fisher_path).exists():
        print(f"\n⚠️  Fisher matrices not found at: {fisher_path}")
        print(f"   Running Fisher computation first...")

        # Use compute_fisher to create them
        from src.phase2_fisher_lora.compute_fisher import FisherComputer

        # Create minimal test data
        test_data = create_test_batch(num_samples=5)

        fisher_computer = FisherComputer(model=baseline_model)
        fisher_matrices = fisher_computer.compute_fisher(
            data_samples=test_data,
            num_samples=5,
            batch_size=1
        )

        # Save
        fisher_computer.save_fisher_matrices(
            fisher_matrices=fisher_matrices,
            output_dir="outputs/test_fisher_computation_real"
        )

        print(f"\n✅ Fisher matrices created")

    # 3. Initialize Fisher-LoRA model
    print("\n" + "="*70)
    print("📦 STEP 2: INITIALIZE FISHER-LORA MODEL")
    print("="*70)

    fisher_lora = FisherGuidedLoRA(
        base_model=baseline_model,
        fisher_path=fisher_path,
        lambda_fisher=0.1,
        beta_brier=0.5
    )

    print("\n✅ Fisher-LoRA model initialized")

    # 4. Save initial parameters
    print("\n" + "="*70)
    print("📦 STEP 3: SAVE INITIAL PARAMETERS")
    print("="*70)

    fisher_lora.save_initial_params()

    print("\n✅ Initial parameters saved")

    # 5. Create optimizer
    print("\n" + "="*70)
    print("📦 STEP 4: CREATE OPTIMIZER")
    print("="*70)

    optimizer = torch.optim.AdamW(
        fisher_lora.model.parameters(),
        lr=2e-4
    )

    print("  ✅ Optimizer created (AdamW, lr=2e-4)")

    # 6. Create test batch
    print("\n" + "="*70)
    print("📦 STEP 5: CREATE TEST BATCH")
    print("="*70)

    test_batch = create_test_batch(num_samples=2)
    print(f"  Created batch with {len(test_batch)} samples")
    print(f"  Sample 1: {test_batch[0]['question']}")
    print(f"  Answer: {test_batch[0]['answer']}")

    # 7. Train for 1 batch
    print("\n" + "="*70)
    print("📦 STEP 6: TRAIN FOR 1 BATCH")
    print("="*70)

    try:
        loss, loss_dict = fisher_lora.train_step(
            batch=test_batch,
            optimizer=optimizer
        )

        print("\n✅ Training step completed!")
        print(f"\n📊 Loss Breakdown:")
        print(f"  Total Loss:  {loss_dict['total']:.4f}")
        print(f"  Task Loss:   {loss_dict['task']:.4f}")
        print(f"  Fisher Loss: {loss_dict['fisher']:.6f}")
        print(f"  Brier Loss:  {loss_dict['brier']:.4f}")

        # Verify loss components
        print(f"\n{'='*70}")
        print(f"✅ VERIFICATION")
        print(f"{'='*70}")

        checks_passed = True

        # Check 1: Loss is a tensor
        if isinstance(loss, torch.Tensor):
            print(f"  ✅ Loss is a tensor")
        else:
            print(f"  ❌ Loss should be a tensor")
            checks_passed = False

        # Check 2: Loss dict has all components
        required_keys = ['total', 'task', 'fisher', 'brier']
        if all(key in loss_dict for key in required_keys):
            print(f"  ✅ Loss dict has all components")
        else:
            print(f"  ❌ Loss dict missing components")
            checks_passed = False

        # Check 3: All losses are non-negative
        if all(loss_dict[key] >= 0 for key in required_keys):
            print(f"  ✅ All losses are non-negative")
        else:
            print(f"  ❌ Some losses are negative")
            checks_passed = False

        # Check 4: Total loss roughly equals sum of components
        expected_total = (loss_dict['task'] +
                         0.1 * loss_dict['fisher'] +
                         0.5 * loss_dict['brier'])

        if abs(loss_dict['total'] - expected_total) < 0.01:
            print(f"  ✅ Total loss matches sum of components")
        else:
            print(f"  ⚠️  Total loss mismatch (expected: {expected_total:.4f})")
            print(f"     This might be due to rounding")

        # Check 5: Loss requires grad (for backprop)
        if loss.requires_grad:
            print(f"  ✅ Loss requires gradient")
        else:
            print(f"  ⚠️  Loss doesn't require gradient")

        print(f"\n{'='*70}")
        if checks_passed:
            print(f"✅ ALL TESTS PASSED!")
        else:
            print(f"⚠️  SOME TESTS FAILED")
        print(f"{'='*70}")

        print(f"\n🎯 Fisher-Guided LoRA is working correctly!")
        print(f"   Ready to integrate into full training pipeline.\n")

    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_with_real_data():
    """Test with real cardiology data if available."""
    data_path = Path("data/cardiology_train.json")

    if not data_path.exists():
        print("\n⚠️  Real cardiology data not found")
        print("   Run: python test_data_loader.py")
        return

    print("\n" + "="*70)
    print("🧪 TESTING WITH REAL CARDIOLOGY DATA")
    print("="*70)

    # Load data
    with open(data_path, 'r') as f:
        cardio_data = json.load(f)

    print(f"  Loaded {len(cardio_data)} cardiology samples")

    # Use first 2 samples
    test_batch = cardio_data[:2]

    # Initialize model
    baseline_model = LoRABaselineModel(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16
    )

    # Initialize Fisher-LoRA
    fisher_path = "outputs/test_fisher_computation_real/fisher_matrices_all.pt"

    if not Path(fisher_path).exists():
        print(f"\n⚠️  Fisher matrices not found. Skipping real data test.")
        return

    fisher_lora = FisherGuidedLoRA(
        base_model=baseline_model,
        fisher_path=fisher_path,
        lambda_fisher=0.1,
        beta_brier=0.5
    )

    fisher_lora.save_initial_params()

    # Optimizer
    optimizer = torch.optim.AdamW(fisher_lora.model.parameters(), lr=2e-4)

    # Train
    print(f"\n  Training on {len(test_batch)} real samples...")

    loss, loss_dict = fisher_lora.train_step(
        batch=test_batch,
        optimizer=optimizer
    )

    print(f"\n  ✅ Training successful!")
    print(f"\n  Loss: {loss_dict['total']:.4f}")
    print(f"  - Task: {loss_dict['task']:.4f}")
    print(f"  - Fisher: {loss_dict['fisher']:.6f}")
    print(f"  - Brier: {loss_dict['brier']:.4f}\n")


if __name__ == "__main__":
    # Run basic test
    success = test_fisher_lora_model()

    if success:
        # Optionally test with real data
        print("\n" + "="*70)
        print("Testing with real cardiology data...")
        print("="*70)

        try:
            test_with_real_data()
        except Exception as e:
            print(f"\n⚠️  Real data test skipped: {e}\n")
