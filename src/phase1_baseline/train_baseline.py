"""
Phase 1 Baseline Training Script
Main script that demonstrates catastrophic forgetting.

This will be the baseline that Phase 2 (Fisher-LoRA) improves upon.

Usage:
    python src/phase1_baseline/train_baseline.py
"""

from src.phase1_baseline.evaluate import (
    evaluate_general_and_cardiology,
    compare_before_after
)
from src.phase1_baseline.model import LoRABaselineModel
from src.utils.metrics import MetricsCalculator
from src.utils.data_loader import MedQADataLoader
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Fix path imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now import


class BaselineTrainer:
    """
    Baseline LoRA trainer that demonstrates catastrophic forgetting.

    This is intentionally simple - no Fisher constraints, no
    uncertainty preservation. Just standard LoRA fine-tuning.
    """

    def __init__(self,
                 model_name: str = "gpt2",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 learning_rate: float = 2e-4,
                 num_epochs: int = 3,
                 batch_size: int = 4,
                 output_dir: str = "outputs/phase1_baseline"):
        """
        Initialize trainer.

        Args:
            model_name: Base model to use
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
            output_dir: Directory to save outputs
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("🎯 PHASE 1 BASELINE TRAINING")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Model: {model_name}")
        print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Output: {output_dir}\n")

        # Initialize model
        print("="*70)
        print("📦 INITIALIZING MODEL")
        print("="*70)
        self.model = LoRABaselineModel(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=learning_rate
        )

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: List of training samples

        Returns:
            Loss value
        """
        self.model.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0

        for sample in batch:
            # Format prompt
            prompt = self.model.format_qa_prompt(
                sample['question'],
                sample['options']
            )

            # Add answer to prompt for training
            full_text = prompt + " " + sample['answer']

            # Tokenize
            inputs = self.model.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            # Forward pass
            outputs = self.model.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Backward
            loss.backward()
            total_loss += loss.item()

        # Optimizer step
        self.optimizer.step()

        return total_loss / len(batch)

    def train_epoch(self, train_data):
        """
        Train for one epoch.

        Args:
            train_data: List of training samples

        Returns:
            Average loss
        """
        total_loss = 0.0
        num_batches = 0

        # Simple batching
        for i in range(0, len(train_data), self.batch_size):
            batch = train_data[i:i + self.batch_size]
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

            if (num_batches) % 10 == 0:
                print(f"  Batch {num_batches}: Loss = {loss:.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, train_data):
        """
        Full training loop.

        Args:
            train_data: List of training samples
        """
        print("\n" + "="*70)
        print("🏋️  TRAINING ON CARDIOLOGY DATA")
        print("="*70)
        print(f"Training samples: {len(train_data)}\n")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)

            avg_loss = self.train_epoch(train_data)

            print(f"\n  Average Loss: {avg_loss:.4f}")

        print("\n✅ Training complete!")

    def save_model(self, save_path: str):
        """Save trained model."""
        self.model.save_model(save_path)

    def save_results(self, results: dict, filename: str = "baseline_results.json"):
        """Save results to JSON."""
        filepath = self.output_dir / filename

        # Add metadata
        results['metadata'] = {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {filepath}")


def main():
    """
    Main training pipeline.

    This demonstrates the catastrophic forgetting problem:
    1. Evaluate BEFORE training (good on general med, bad on cardiology)
    2. Train on cardiology data
    3. Evaluate AFTER training (bad on general med, good on cardiology)
    4. Calculate forgetting metric
    """

    # Configuration
    config = {
        'model_name': 'gpt2',
        'lora_r': 8,
        'lora_alpha': 16,
        'learning_rate': 2e-4,
        'num_epochs': 3,
        'batch_size': 4,
        'output_dir': 'outputs/phase1_baseline'
    }

    # Initialize trainer
    trainer = BaselineTrainer(**config)

    # Load data
    print("\n" + "="*70)
    print("📂 LOADING DATA")
    print("="*70)
    loader = MedQADataLoader(data_dir="data")

    # Load training data (cardiology)
    cardiology_train = loader.load_data('cardiology_train.json')
    print(f"Cardiology training samples: {len(cardiology_train)}")

    # ========================================================================
    # STEP 1: EVALUATE BEFORE TRAINING
    # ========================================================================

    print("\n" + "="*70)
    print("📊 STEP 1: EVALUATE BEFORE TRAINING")
    print("="*70)
    print("This shows baseline performance - what the model knows BEFORE")
    print("fine-tuning on cardiology.\n")

    before_results = evaluate_general_and_cardiology(trainer.model)

    # ========================================================================
    # STEP 2: TRAIN ON CARDIOLOGY DATA
    # ========================================================================

    trainer.train(cardiology_train)

    # ========================================================================
    # STEP 3: EVALUATE AFTER TRAINING
    # ========================================================================

    print("\n" + "="*70)
    print("📊 STEP 2: EVALUATE AFTER TRAINING")
    print("="*70)
    print("This shows how performance changed after fine-tuning.\n")

    after_results = evaluate_general_and_cardiology(trainer.model)

    # ========================================================================
    # STEP 4: COMPARE AND CALCULATE FORGETTING
    # ========================================================================

    comparison = compare_before_after(before_results, after_results)

    # ========================================================================
    # STEP 5: SAVE EVERYTHING
    # ========================================================================

    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)

    # Save model
    model_path = str(trainer.output_dir / "models" / "baseline_lora.pt")
    trainer.save_model(model_path)

    # Save results
    trainer.save_results(comparison)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("🎯 PHASE 1 COMPLETE!")
    print("="*70)
    print("\n📊 Key Results:")
    print(f"  Forgetting: {comparison['forgetting']:.2%}")
    print(
        f"  Cardiology Improvement: {comparison['cardiology_improvement']:.2%}")

    if comparison['forgetting'] > 0.10:
        print("\n❌ HIGH FORGETTING DETECTED!")
        print("   This is the problem we're solving in Phase 2 (Fisher-LoRA)")
    else:
        print("\n✅ Low forgetting - baseline already good")

    print("\n📁 Outputs saved to:")
    print(f"  Models: {trainer.output_dir / 'models'}")
    print(f"  Results: {trainer.output_dir / 'baseline_results.json'}")

    print("\n🚀 Next Steps:")
    print("  1. Review results in baseline_results.json")
    print("  2. Hand off to Sarghi for Week 2 testing")
    print("  3. Move to Phase 2: Fisher-LoRA implementation")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
