"""
Test script for baseline training pipeline.
Runs a minimal version to verify everything works.

Usage:
    python test_train_baseline.py
"""

from src.utils.data_loader import MedQADataLoader
from src.phase1_baseline.train_baseline import BaselineTrainer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_training_pipeline():
    """Test the training pipeline with minimal data."""

    print("="*70)
    print("🧪 TESTING BASELINE TRAINING PIPELINE")
    print("="*70 + "\n")

    # Initialize trainer with small config
    print("1️⃣ Initializing trainer...")
    trainer = BaselineTrainer(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16,
        learning_rate=2e-4,
        num_epochs=1,  # Just 1 epoch for testing
        batch_size=2,  # Small batch
        output_dir="outputs/test_baseline"
    )
    print("  ✓ Trainer initialized\n")

    # Load minimal data
    print("2️⃣ Loading test data...")
    loader = MedQADataLoader(data_dir="data")
    cardiology_train = loader.load_data('cardiology_train.json')[
        :4]  # Just 4 samples
    print(f"  ✓ Loaded {len(cardiology_train)} samples\n")

    # Evaluate before
    print("3️⃣ Evaluating BEFORE training (5 samples each)...")
    gen_test = loader.load_data('general_medicine_test.json')[:5]
    card_test = loader.load_data('cardiology_test.json')[:5]

    print("  Testing on general medicine...")
    from src.phase1_baseline.evaluate import evaluate_on_dataset
    before_gen = evaluate_on_dataset(
        trainer.model, gen_test, "General Medicine")

    print("  Testing on cardiology...")
    before_card = evaluate_on_dataset(trainer.model, card_test, "Cardiology")
    print("  ✓ Before evaluation complete\n")

    # Train
    print("4️⃣ Training (1 epoch, 4 samples)...")
    trainer.train(cardiology_train)
    print("  ✓ Training complete\n")

    # Evaluate after
    print("5️⃣ Evaluating AFTER training...")
    after_gen = evaluate_on_dataset(
        trainer.model, gen_test, "General Medicine")
    after_card = evaluate_on_dataset(trainer.model, card_test, "Cardiology")
    print("  ✓ After evaluation complete\n")

    # Calculate forgetting
    print("6️⃣ Calculating forgetting...")
    from src.utils.metrics import MetricsCalculator
    calc = MetricsCalculator()
    forgetting = calc.calculate_forgetting(
        before_gen['accuracy'],
        after_gen['accuracy']
    )
    print(f"  Forgetting: {forgetting:.2%}\n")

    # Save
    print("7️⃣ Saving model and results...")
    trainer.save_model("outputs/test_baseline/models/test_model")
    results = {
        'before_training': {
            'general_medicine': {'accuracy': before_gen['accuracy']},
            'cardiology': {'accuracy': before_card['accuracy']}
        },
        'after_training': {
            'general_medicine': {'accuracy': after_gen['accuracy']},
            'cardiology': {'accuracy': after_card['accuracy']}
        },
        'forgetting': forgetting
    }
    trainer.save_results(results, "test_results.json")
    print("  ✓ Results saved\n")

    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📊 Test Results:")
    print(f"  Before - Gen Med: {before_gen['accuracy']:.2%}")
    print(f"  After  - Gen Med: {after_gen['accuracy']:.2%}")
    print(f"  Forgetting: {forgetting:.2%}")
    print(f"\n  Before - Cardiology: {before_card['accuracy']:.2%}")
    print(f"  After  - Cardiology: {after_card['accuracy']:.2%}")

    print("\n💡 Ready for full baseline training!")
    print("   Run: python src/phase1_baseline/train_baseline.py")

    return True


if __name__ == "__main__":
    success = test_training_pipeline()
    sys.exit(0 if success else 1)
