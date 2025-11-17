"""
Fisher Information Matrix Computation for LoRA Parameters
Phase 2 - Step 2

Computes Fisher Information Matrix from the Phase 1 baseline model
on general medicine data to identify important parameters.

Fisher Information: F_i = E[(∂L/∂θ_i)²]
- High Fisher value = parameter is important for general medicine
- Low Fisher value = parameter can be changed without affecting performance

This is computed BEFORE cardiology training, using general medicine data.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.phase1_baseline.model import LoRABaselineModel
from src.utils.data_loader import MedQADataLoader


class FisherComputer:
    """
    Computes Fisher Information Matrix for LoRA parameters.

    The Fisher matrix identifies which LoRA parameters are most important
    for maintaining general medical knowledge.
    """

    def __init__(self,
                 model: LoRABaselineModel,
                 device: str = None):
        """
        Initialize Fisher computer.

        Args:
            model: Trained Phase 1 baseline model
            device: Device to use (None = auto-detect)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract LoRA parameters
        self.lora_params = self._get_lora_parameters()

        print(f"\n🔧 Fisher Computer Initialized")
        print(f"  Device: {self.device}")
        print(f"  LoRA parameters found: {len(self.lora_params)}")

    def _get_lora_parameters(self) -> Dict[str, nn.Parameter]:
        """
        Extract all LoRA A and B parameters from the model.

        Returns:
            Dictionary mapping parameter names to parameters
        """
        lora_params = {}

        for name, param in self.model.model.named_parameters():
            # Only include LoRA parameters (A and B matrices)
            if 'lora_A' in name or 'lora_B' in name:
                if param.requires_grad:
                    lora_params[name] = param
                    print(f"    Found: {name} - shape {param.shape}")

        if len(lora_params) == 0:
            print("    Warning: No LoRA parameters found!")
            print("    This might happen if the model is not a PEFT model")

        return lora_params

    def compute_fisher(self,
                      data_samples: list,
                      num_samples: int = None,
                      batch_size: int = 1,
                      diagonal_only: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix.

        Process:
        1. For each sample in general medicine data:
           - Forward pass to get predictions
           - Compute loss (cross-entropy)
           - Backward pass to get gradients
           - Accumulate squared gradients (Fisher ≈ E[grad²])
        2. Average across all samples
        3. Return Fisher matrices for each LoRA parameter

        Args:
            data_samples: List of general medicine samples
            num_samples: Number of samples to use (None = all)
            batch_size: Batch size for computation
            diagonal_only: If True, only compute diagonal of Fisher matrix
                          (faster, good approximation)

        Returns:
            Dictionary mapping parameter names to Fisher matrices
            Same shape as the LoRA parameters
        """
        if num_samples is not None:
            data_samples = data_samples[:num_samples]

        total_samples = len(data_samples)

        print(f"\n{'='*70}")
        print(f"🧮 COMPUTING FISHER INFORMATION MATRIX")
        print(f"{'='*70}")
        print(f"  Samples: {total_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Diagonal only: {diagonal_only}")
        print(f"  LoRA parameters: {len(self.lora_params)}")

        # Initialize Fisher matrices (same shape as parameters)
        fisher_matrices = {}
        for name, param in self.lora_params.items():
            fisher_matrices[name] = torch.zeros_like(param.data)

        # Set model to eval mode (no dropout, etc.)
        self.model.model.eval()

        # Compute Fisher for each sample
        num_batches = (total_samples + batch_size - 1) // batch_size

        with tqdm(total=total_samples, desc="Computing Fisher") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_samples = data_samples[start_idx:end_idx]

                for sample in batch_samples:
                    # Compute gradients for this sample
                    gradients = self._compute_sample_gradients(sample)

                    # Accumulate squared gradients (Fisher Information)
                    for name in gradients:
                        fisher_matrices[name] += gradients[name] ** 2

                    pbar.update(1)

        # Average over all samples
        for name in fisher_matrices:
            fisher_matrices[name] /= total_samples

        # Print statistics
        self._print_fisher_statistics(fisher_matrices)

        return fisher_matrices

    def _compute_sample_gradients(self, sample: dict) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for a single sample.

        Args:
            sample: Data sample with 'question', 'options', 'answer'

        Returns:
            Dictionary of gradients for each LoRA parameter
        """
        # Zero gradients
        self.model.model.zero_grad()

        # Format prompt
        prompt = self.model.format_qa_prompt(
            sample['question'],
            sample['options']
        )

        # Get correct answer
        answer_key = sample['answer']

        # Tokenize input
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Tokenize answer (target)
        # For multiple choice, we want the model to predict the answer letter
        answer_text = f" {answer_key}"
        answer_ids = self.model.tokenizer.encode(
            answer_text,
            add_special_tokens=False
        )

        if len(answer_ids) == 0:
            # Fallback: use the first token of the answer letter
            answer_ids = [self.model.tokenizer.encode(answer_key)[0]]

        # Forward pass
        outputs = self.model.model(**inputs, labels=inputs['input_ids'])

        # Get loss
        # We want the cross-entropy loss for predicting the correct answer
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Extract gradients for LoRA parameters
        gradients = {}
        for name, param in self.lora_params.items():
            if param.grad is not None:
                # Clone and detach the gradient
                gradients[name] = param.grad.data.clone().detach()
            else:
                # If no gradient, use zeros
                gradients[name] = torch.zeros_like(param.data)

        return gradients

    def _print_fisher_statistics(self, fisher_matrices: Dict[str, torch.Tensor]):
        """Print statistics about computed Fisher matrices."""
        print(f"\n{'='*70}")
        print(f"📊 FISHER MATRIX STATISTICS")
        print(f"{'='*70}")

        for name, fisher in fisher_matrices.items():
            fisher_np = fisher.cpu().numpy()

            print(f"\n{name}:")
            print(f"  Shape: {fisher.shape}")
            print(f"  Min:   {fisher_np.min():.6f}")
            print(f"  Max:   {fisher_np.max():.6f}")
            print(f"  Mean:  {fisher_np.mean():.6f}")
            print(f"  Std:   {fisher_np.std():.6f}")

            # Histogram of values
            hist, bins = np.histogram(fisher_np.flatten(), bins=5)
            print(f"  Distribution:")
            for i in range(len(hist)):
                bar = '█' * int(hist[i] / hist.max() * 20)
                print(f"    [{bins[i]:.4f}, {bins[i+1]:.4f}]: {bar} ({hist[i]})")

    def save_fisher_matrices(self,
                            fisher_matrices: Dict[str, torch.Tensor],
                            output_dir: str):
        """
        Save Fisher matrices to disk.

        Args:
            fisher_matrices: Dictionary of Fisher matrices
            output_dir: Directory to save matrices
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"💾 SAVING FISHER MATRICES")
        print(f"{'='*70}")
        print(f"  Output directory: {output_path}")

        # Save individual matrices
        for name, fisher in fisher_matrices.items():
            # Clean name for filename (remove dots, slashes)
            clean_name = name.replace('.', '_').replace('/', '_')
            filepath = output_path / f"fisher_{clean_name}.pt"

            torch.save(fisher, filepath)
            print(f"  ✅ Saved: {filepath.name}")

        # Save all matrices in one file (easier to load later)
        all_fisher_path = output_path / "fisher_matrices_all.pt"
        torch.save(fisher_matrices, all_fisher_path)
        print(f"  ✅ Saved: {all_fisher_path.name} (combined)")

        # Save metadata
        metadata = {
            'num_parameters': len(fisher_matrices),
            'parameter_names': list(fisher_matrices.keys()),
            'shapes': {name: list(fisher.shape) for name, fisher in fisher_matrices.items()},
            'statistics': {
                name: {
                    'min': float(fisher.min()),
                    'max': float(fisher.max()),
                    'mean': float(fisher.mean()),
                    'std': float(fisher.std())
                }
                for name, fisher in fisher_matrices.items()
            }
        }

        metadata_path = output_path / "fisher_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✅ Saved: {metadata_path.name} (metadata)")

        print(f"\n✅ Fisher matrices saved successfully!")


def main():
    """
    Main function to compute Fisher matrices.

    Steps:
    1. Load Phase 1 baseline model
    2. Load general medicine data
    3. Compute Fisher Information
    4. Save Fisher matrices
    """
    print("\n" + "="*70)
    print("🎯 FISHER INFORMATION COMPUTATION")
    print("   Phase 2 - Step 2")
    print("="*70)

    # Configuration
    model_path = "outputs/phase1_baseline/models/baseline_lora.pt"
    data_dir = "data"
    output_dir = "outputs/phase2_fisher_lora/fisher_matrices"
    num_samples = 800  # Use all general medicine training samples

    # 1. Load baseline model
    print("\n" + "="*70)
    print("📦 LOADING PHASE 1 BASELINE MODEL")
    print("="*70)
    print(f"  Model path: {model_path}")

    # Initialize model
    model = LoRABaselineModel(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    # Load trained weights
    if Path(model_path).exists():
        print(f"\n  Loading weights from {model_path}")
        # Note: The actual loading depends on how you saved in Phase 1
        # This might need adjustment based on your saving method
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("  Warning: Checkpoint format not recognized. Using untrained model.")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            print("  Using untrained model for demonstration.")
    else:
        print(f"  Warning: Model file not found at {model_path}")
        print(f"  Using untrained model for demonstration.")
        print(f"  To compute real Fisher matrices, first train the baseline model.")

    # 2. Load general medicine data
    print("\n" + "="*70)
    print("📚 LOADING GENERAL MEDICINE DATA")
    print("="*70)

    data_loader = MedQADataLoader(data_dir=data_dir)

    # Load general medicine training data
    general_medicine_path = Path(data_dir) / "general_medicine_train.json"

    if general_medicine_path.exists():
        print(f"  Loading from: {general_medicine_path}")
        with open(general_medicine_path, 'r') as f:
            general_medicine_data = json.load(f)
        print(f"  Loaded {len(general_medicine_data)} samples")
    else:
        print(f"  Error: Data file not found at {general_medicine_path}")
        print(f"  Please run: python test_data_loader.py")
        return

    # Limit samples if needed
    if num_samples is not None and len(general_medicine_data) > num_samples:
        general_medicine_data = general_medicine_data[:num_samples]
        print(f"  Using {num_samples} samples for Fisher computation")

    # 3. Compute Fisher Information
    fisher_computer = FisherComputer(model=model)

    fisher_matrices = fisher_computer.compute_fisher(
        data_samples=general_medicine_data,
        num_samples=num_samples,
        batch_size=1,
        diagonal_only=True
    )

    # 4. Save Fisher matrices
    fisher_computer.save_fisher_matrices(
        fisher_matrices=fisher_matrices,
        output_dir=output_dir
    )

    print("\n" + "="*70)
    print("✅ FISHER COMPUTATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output saved to: {output_dir}/")
    print(f"\n🎯 Next step: Use these Fisher matrices in Fisher-LoRA training")
    print(f"   The Fisher matrices will prevent forgetting of general medicine\n")


if __name__ == "__main__":
    main()
