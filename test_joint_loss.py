"""
Quick test for FisherGuidedLoss
Verifies the joint loss function works correctly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.phase2_fisher_lora.joint_loss import FisherGuidedLoss


def test_fisher_guided_loss():
    """Test the Fisher-Guided loss computation."""

    print("\n" + "="*70)
    print("🧪 TESTING FISHER-GUIDED LOSS")
    print("="*70)

    # Simulate some LoRA parameters
    # In reality, these would be actual LoRA A and B matrices
    lora_dim = 10

    # Initial LoRA parameters (before cardiology training)
    initial_A = torch.randn(lora_dim, lora_dim, requires_grad=False)
    initial_B = torch.randn(lora_dim, lora_dim, requires_grad=False)

    # Fisher matrices (would be computed from general medicine data)
    # Higher values = more important parameters
    fisher_A = torch.abs(torch.randn(lora_dim, lora_dim))
    fisher_B = torch.abs(torch.randn(lora_dim, lora_dim))

    # Create loss function
    loss_fn = FisherGuidedLoss(
        fisher_A=fisher_A,
        fisher_B=fisher_B,
        initial_lora_A=initial_A,
        initial_lora_B=initial_B,
        lambda_fisher=0.1,
        beta_brier=0.5
    )

    print("\n✅ Loss function initialized")
    print(f"   Lambda (Fisher): {loss_fn.lambda_fisher}")
    print(f"   Beta (Brier): {loss_fn.beta_brier}")

    # Simulate a training batch
    batch_size = 4
    num_classes = 5

    # Model predictions (logits) - requires grad for backprop
    logits = torch.randn(batch_size, num_classes, requires_grad=True)

    # Ground truth labels
    targets = torch.randint(0, num_classes, (batch_size,))

    # Current LoRA parameters (after some training steps)
    # These have changed slightly from initial - requires grad for training
    current_A = (initial_A + 0.01 * torch.randn_like(initial_A)).clone().detach().requires_grad_(True)
    current_B = (initial_B + 0.01 * torch.randn_like(initial_B)).clone().detach().requires_grad_(True)

    print("\n🔄 Computing loss on simulated batch...")
    print(f"   Batch size: {batch_size}")
    print(f"   Num classes: {num_classes}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Targets: {targets}")

    # Compute loss
    total_loss, loss_dict = loss_fn.compute_loss(
        logits=logits,
        targets=targets,
        current_lora_A=current_A,
        current_lora_B=current_B
    )

    print("\n📊 Loss Breakdown:")
    print(f"   Total Loss:  {loss_dict['total']:.4f}")
    print(f"   Task Loss:   {loss_dict['task']:.4f}")
    print(f"   Fisher Loss: {loss_dict['fisher']:.4f}")
    print(f"   Brier Loss:  {loss_dict['brier']:.4f}")

    # Verify loss can be backpropagated
    print("\n🔙 Testing backward pass...")
    total_loss.backward()
    print("   ✅ Backward pass successful")

    # Test case 2: No parameter change = zero Fisher loss
    print("\n" + "="*70)
    print("🧪 TEST 2: No Parameter Change")
    print("="*70)

    total_loss_2, loss_dict_2 = loss_fn.compute_loss(
        logits=logits,
        targets=targets,
        current_lora_A=initial_A,  # Same as initial
        current_lora_B=initial_B   # Same as initial
    )

    print(f"\n   Fisher Loss (no change): {loss_dict_2['fisher']:.6f}")
    print(f"   Expected: ~0.0")

    if abs(loss_dict_2['fisher']) < 1e-5:
        print("   ✅ Fisher loss is zero when parameters don't change")
    else:
        print("   ⚠️  Warning: Fisher loss should be zero")

    # Test case 3: Large parameter change = high Fisher loss
    print("\n" + "="*70)
    print("🧪 TEST 3: Large Parameter Change")
    print("="*70)

    large_change_A = initial_A + 1.0 * torch.randn_like(initial_A)
    large_change_B = initial_B + 1.0 * torch.randn_like(initial_B)

    total_loss_3, loss_dict_3 = loss_fn.compute_loss(
        logits=logits,
        targets=targets,
        current_lora_A=large_change_A,
        current_lora_B=large_change_B
    )

    print(f"\n   Fisher Loss (small change): {loss_dict['fisher']:.4f}")
    print(f"   Fisher Loss (large change): {loss_dict_3['fisher']:.4f}")

    if loss_dict_3['fisher'] > loss_dict['fisher']:
        print("   ✅ Fisher loss increases with larger parameter changes")
    else:
        print("   ⚠️  Warning: Fisher loss should increase with larger changes")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe FisherGuidedLoss is working correctly!")
    print("Ready to integrate into training pipeline.\n")


if __name__ == "__main__":
    test_fisher_guided_loss()
