"""
Fisher-Guided LoRA Training Pipeline
Phase 2 - Step 4 (Final Implementation)

Complete training pipeline that:
1. Loads general medicine + cardiology data
2. Computes or loads Fisher matrices
3. Trains Fisher-LoRA on cardiology
4. Evaluates forgetting and compares with baseline

Usage:
    python src/phase2_fisher_lora/train_fisher_lora.py \
        --lambda_fisher 0.1 \
        --beta_brier 0.5 \
        --epochs 3 \
        --lr 2e-4
"""

import torch
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.phase1_baseline.model import LoRABaselineModel
from src.phase1_baseline.evaluate import evaluate_on_dataset
from src.phase2_fisher_lora.fisher_lora import FisherGuidedLoRA
from src.phase2_fisher_lora.compute_fisher import FisherComputer
from src.utils.data_loader import MedQADataLoader


class FisherLoRATrainer:
    """
    Complete training pipeline for Fisher-Guided LoRA.

    This is the main trainer that demonstrates the full method:
    - Fisher Information Matrix computation
    - Fisher-guided LoRA training
    - Brier score optimization
    - Forgetting analysis
    """

    def __init__(self,
                 model_name: str = "epfl-llm/meditron-7b",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lambda_fisher: float = 0.1,
                 beta_brier: float = 0.5,
                 learning_rate: float = 2e-4,
                 num_epochs: int = 3,
                 batch_size: int = 4,
                 load_in_4bit: bool = True,
                 output_dir: str = "outputs/phase2_fisher_lora"):
        """
        Initialize Fisher-LoRA trainer.

        Args:
            model_name: Base model name
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lambda_fisher: Fisher regularization weight
            beta_brier: Brier score weight
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
            output_dir: Output directory
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lambda_fisher = lambda_fisher
        self.beta_brier = beta_brier
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.load_in_4bit = load_in_4bit
        self.output_dir = Path(output_dir)

        # Create output directories
        self.models_dir = self.output_dir / "models"
        self.fisher_dir = self.output_dir / "fisher_matrices"
        self.results_dir = self.output_dir / "results"

        for dir_path in [self.models_dir, self.fisher_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("🎯 FISHER-GUIDED LORA TRAINING PIPELINE")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Model: {model_name}")
        print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
        print(f"  Lambda (Fisher): {lambda_fisher}")
        print(f"  Beta (Brier): {beta_brier}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Output: {output_dir}\n")

    def load_data(self, data_dir: str = "data"):
        """Load general medicine and cardiology datasets."""
        print("="*70)
        print("📚 LOADING DATA")
        print("="*70)

        data_loader = MedQADataLoader(data_dir=data_dir)

        # Load all datasets
        datasets = {}
        file_mapping = {
            'general_medicine_train': 'general_medicine_train.json',
            'general_medicine_test': 'general_medicine_test.json',
            'cardiology_train': 'cardiology_train.json',
            'cardiology_test': 'cardiology_test.json'
        }

        for name, filename in file_mapping.items():
            filepath = Path(data_dir) / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    datasets[name] = json.load(f)
                print(f"  ✅ {name}: {len(datasets[name])} samples")
            else:
                print(f"  ❌ {name}: File not found at {filepath}")
                raise FileNotFoundError(f"Please run: python test_data_loader.py")

        self.datasets = datasets
        return datasets

    def initialize_baseline_model(self):
        """Initialize baseline LoRA model."""
        print("\n" + "="*70)
        print("📦 INITIALIZING BASELINE MODEL")
        print("="*70)

        self.baseline_model = LoRABaselineModel(
            model_name=self.model_name,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            load_in_4bit=self.load_in_4bit
        )

        print("\n✅ Baseline model initialized")
        return self.baseline_model

    def compute_or_load_fisher(self, force_recompute: bool = False):
        """Compute or load Fisher matrices."""
        print("\n" + "="*70)
        print("🧮 FISHER INFORMATION MATRIX")
        print("="*70)

        fisher_path = self.fisher_dir / "fisher_matrices_all.pt"

        # Check if Fisher matrices already exist
        if fisher_path.exists() and not force_recompute:
            print(f"\n  Fisher matrices found at: {fisher_path}")
            print(f"  Loading existing matrices...")

            self.fisher_matrices = torch.load(fisher_path, map_location='cpu')
            print(f"  ✅ Loaded {len(self.fisher_matrices)} Fisher matrices")

        else:
            print(f"\n  Computing Fisher matrices from general medicine data...")
            print(f"  This may take a few minutes...")

            # Compute Fisher Information
            fisher_computer = FisherComputer(model=self.baseline_model)

            self.fisher_matrices = fisher_computer.compute_fisher(
                data_samples=self.datasets['general_medicine_train'],
                num_samples=800,  # Use all training samples
                batch_size=1,
                diagonal_only=True
            )

            # Save Fisher matrices
            fisher_computer.save_fisher_matrices(
                fisher_matrices=self.fisher_matrices,
                output_dir=str(self.fisher_dir)
            )

            print(f"\n  ✅ Fisher matrices computed and saved")

        return fisher_path

    def evaluate_before_training(self):
        """Evaluate model before cardiology training."""
        print("\n" + "="*70)
        print("📊 EVALUATION BEFORE TRAINING")
        print("="*70)

        results_before = {}

        # Evaluate on general medicine
        print("\n  Evaluating on general medicine test set...")
        gen_med_results = evaluate_on_dataset(
            model=self.baseline_model,
            samples=self.datasets['general_medicine_test'],
            dataset_name="general_medicine"
        )
        results_before['general_medicine'] = gen_med_results

        # Evaluate on cardiology
        print("\n  Evaluating on cardiology test set...")
        cardio_results = evaluate_on_dataset(
            model=self.baseline_model,
            samples=self.datasets['cardiology_test'],
            dataset_name="cardiology"
        )
        results_before['cardiology'] = cardio_results

        print("\n  Results BEFORE Training:")
        print(f"    General Medicine: {gen_med_results['accuracy']:.1%}")
        print(f"    Cardiology:       {cardio_results['accuracy']:.1%}")

        self.results_before = results_before
        return results_before

    def initialize_fisher_lora(self, fisher_path: Path):
        """Initialize Fisher-LoRA model."""
        print("\n" + "="*70)
        print("🔧 INITIALIZING FISHER-LORA MODEL")
        print("="*70)

        self.fisher_lora = FisherGuidedLoRA(
            base_model=self.baseline_model,
            fisher_path=str(fisher_path),
            lambda_fisher=self.lambda_fisher,
            beta_brier=self.beta_brier
        )

        # Save initial parameters (before cardiology training)
        self.fisher_lora.save_initial_params()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.fisher_lora.model.parameters(),
            lr=self.learning_rate
        )

        print(f"\n✅ Fisher-LoRA ready for training")
        return self.fisher_lora

    def train_on_cardiology(self):
        """Train Fisher-LoRA on cardiology data."""
        print("\n" + "="*70)
        print("🚀 TRAINING ON CARDIOLOGY DATA")
        print("="*70)

        cardiology_train = self.datasets['cardiology_train']
        num_samples = len(cardiology_train)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        print(f"\n  Samples: {num_samples}")
        print(f"  Batches per epoch: {num_batches}")
        print(f"  Epochs: {self.num_epochs}")

        training_log = []

        for epoch in range(self.num_epochs):
            print(f"\n{'='*70}")
            print(f"  Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*70}")

            epoch_losses = {
                'total': 0.0,
                'task': 0.0,
                'fisher': 0.0,
                'brier': 0.0
            }

            # Training loop with progress bar
            with tqdm(total=num_batches, desc=f"  Epoch {epoch+1}") as pbar:
                for batch_idx in range(num_batches):
                    # Get batch
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, num_samples)
                    batch = cardiology_train[start_idx:end_idx]

                    # Train step
                    loss, loss_dict = self.fisher_lora.train_step(
                        batch=batch,
                        optimizer=self.optimizer
                    )

                    # Accumulate losses
                    for key in epoch_losses:
                        epoch_losses[key] += loss_dict[key]

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total']:.4f}",
                        'task': f"{loss_dict['task']:.4f}",
                        'fisher': f"{loss_dict['fisher']:.6f}",
                        'brier': f"{loss_dict['brier']:.4f}"
                    })
                    pbar.update(1)

            # Average epoch losses
            for key in epoch_losses:
                epoch_losses[key] /= num_batches

            # Log epoch results
            epoch_log = {
                'epoch': epoch + 1,
                'losses': epoch_losses
            }
            training_log.append(epoch_log)

            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Total Loss:  {epoch_losses['total']:.4f}")
            print(f"    Task Loss:   {epoch_losses['task']:.4f}")
            print(f"    Fisher Loss: {epoch_losses['fisher']:.6f}")
            print(f"    Brier Loss:  {epoch_losses['brier']:.4f}")

        self.training_log = training_log
        print(f"\n✅ Training complete!")
        return training_log

    def evaluate_after_training(self):
        """Evaluate model after cardiology training."""
        print("\n" + "="*70)
        print("📊 EVALUATION AFTER TRAINING")
        print("="*70)

        results_after = {}

        # Evaluate on general medicine
        print("\n  Evaluating on general medicine test set...")
        gen_med_results = evaluate_on_dataset(
            model=self.baseline_model,  # Uses same model, now fine-tuned
            samples=self.datasets['general_medicine_test'],
            dataset_name="general_medicine"
        )
        results_after['general_medicine'] = gen_med_results

        # Evaluate on cardiology
        print("\n  Evaluating on cardiology test set...")
        cardio_results = evaluate_on_dataset(
            model=self.baseline_model,
            samples=self.datasets['cardiology_test'],
            dataset_name="cardiology"
        )
        results_after['cardiology'] = cardio_results

        print("\n  Results AFTER Training:")
        print(f"    General Medicine: {gen_med_results['accuracy']:.1%}")
        print(f"    Cardiology:       {cardio_results['accuracy']:.1%}")

        self.results_after = results_after
        return results_after

    def analyze_forgetting(self):
        """Calculate forgetting and domain transfer metrics."""
        print("\n" + "="*70)
        print("📈 FORGETTING ANALYSIS")
        print("="*70)

        # Calculate forgetting on general medicine
        acc_before = self.results_before['general_medicine']['accuracy']
        acc_after = self.results_after['general_medicine']['accuracy']
        forgetting = acc_before - acc_after

        # Calculate improvement on cardiology
        cardio_before = self.results_before['cardiology']['accuracy']
        cardio_after = self.results_after['cardiology']['accuracy']
        cardio_improvement = cardio_after - cardio_before

        # ECE comparison
        ece_before = self.results_before['general_medicine'].get('ece', 0)
        ece_after = self.results_after['general_medicine'].get('ece', 0)

        print(f"\n  📊 General Medicine:")
        print(f"    Before:     {acc_before:.1%}")
        print(f"    After:      {acc_after:.1%}")
        print(f"    Forgetting: {forgetting:.1%} {'✅' if forgetting < 0.05 else '❌'}")

        print(f"\n  📊 Cardiology:")
        print(f"    Before:      {cardio_before:.1%}")
        print(f"    After:       {cardio_after:.1%}")
        print(f"    Improvement: {cardio_improvement:.1%}")

        print(f"\n  📊 Calibration (ECE):")
        print(f"    Before: {ece_before:.4f}")
        print(f"    After:  {ece_after:.4f}")

        metrics = {
            'forgetting': float(forgetting),
            'cardiology_improvement': float(cardio_improvement),
            'ece_before': float(ece_before),
            'ece_after': float(ece_after)
        }

        self.metrics = metrics
        return metrics

    def save_results(self):
        """Save all results to JSON file."""
        print("\n" + "="*70)
        print("💾 SAVING RESULTS")
        print("="*70)

        # Compile all results
        results = {
            'metadata': {
                'model_name': self.model_name,
                'lora_r': self.lora_r,
                'lora_alpha': self.lora_alpha,
                'lambda_fisher': self.lambda_fisher,
                'beta_brier': self.beta_brier,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'timestamp': datetime.now().isoformat()
            },
            'before_training': self.results_before,
            'after_training': self.results_after,
            'training_log': self.training_log,
            'metrics': self.metrics
        }

        # Save results
        results_path = self.results_dir / "fisher_lora_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ✅ Results saved to: {results_path}")

        # Save model
        model_path = self.models_dir / "fisher_lora_model.pt"
        self.fisher_lora.save_model(str(model_path))

        print(f"  ✅ Model saved to: {model_path}")

        return results_path

    def run(self, data_dir: str = "data", force_recompute_fisher: bool = False):
        """
        Run complete training pipeline.

        Args:
            data_dir: Directory containing data
            force_recompute_fisher: Whether to recompute Fisher matrices

        Returns:
            Results dictionary
        """
        # 1. Load data
        self.load_data(data_dir)

        # 2. Initialize baseline model
        self.initialize_baseline_model()

        # 3. Compute or load Fisher matrices
        fisher_path = self.compute_or_load_fisher(force_recompute_fisher)

        # 4. Evaluate before training
        self.evaluate_before_training()

        # 5. Initialize Fisher-LoRA
        self.initialize_fisher_lora(fisher_path)

        # 6. Train on cardiology
        self.train_on_cardiology()

        # 7. Evaluate after training
        self.evaluate_after_training()

        # 8. Analyze forgetting
        self.analyze_forgetting()

        # 9. Save results
        results_path = self.save_results()

        # 10. Print summary
        self.print_summary()

        return results_path

    def print_summary(self):
        """Print final summary."""
        print("\n" + "="*70)
        print("✅ FISHER-GUIDED LORA TRAINING COMPLETE!")
        print("="*70)

        print(f"\n📊 Final Results:")
        print(f"  Forgetting:         {self.metrics['forgetting']:.1%}")
        print(f"  Cardiology Gain:    {self.metrics['cardiology_improvement']:.1%}")
        print(f"  ECE (After):        {self.metrics['ece_after']:.4f}")

        print(f"\n🎯 Target Metrics:")
        forgetting_met = self.metrics['forgetting'] < 0.05
        ece_met = self.metrics['ece_after'] < 0.08
        cardio_met = self.metrics['cardiology_improvement'] > 0.2

        print(f"  Forgetting < 5%:    {'✅' if forgetting_met else '❌'}")
        print(f"  ECE < 0.08:         {'✅' if ece_met else '❌'}")
        print(f"  Cardiology > 20%:   {'✅' if cardio_met else '❌'}")

        if forgetting_met and ece_met and cardio_met:
            print(f"\n🎉 ALL TARGETS MET! Fisher-LoRA is working as expected!")
        else:
            print(f"\n⚠️  Some targets not met. Consider tuning hyperparameters.")

        print(f"\n📁 Outputs saved to: {self.output_dir}/")
        print()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train Fisher-Guided LoRA for medical domain adaptation"
    )

    parser.add_argument('--model_name', type=str, default='epfl-llm/meditron-7b',
                       help='Base model name (default: Meditron-7B)')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lambda_fisher', type=float, default=0.1,
                       help='Fisher regularization weight')
    parser.add_argument('--beta_brier', type=float, default=0.5,
                       help='Brier score weight')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (can use 4+ with A100 GPU)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True,
                       help='Use 4-bit quantization (default: True)')
    parser.add_argument('--output_dir', type=str, default='outputs/phase2_fisher_lora',
                       help='Output directory')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--force_recompute_fisher', action='store_true',
                       help='Force recompute Fisher matrices')

    args = parser.parse_args()

    # Initialize trainer
    trainer = FisherLoRATrainer(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lambda_fisher=args.lambda_fisher,
        beta_brier=args.beta_brier,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
        output_dir=args.output_dir
    )

    # Run training pipeline
    trainer.run(
        data_dir=args.data_dir,
        force_recompute_fisher=args.force_recompute_fisher
    )


if __name__ == "__main__":
    main()
