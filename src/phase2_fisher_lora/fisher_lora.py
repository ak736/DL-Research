"""
Fisher-Guided LoRA Model Class
Phase 2 - Step 3

Wraps baseline LoRA model with Fisher-guided training.
Combines: LoRA + Fisher matrices + Joint loss (Task + Fisher + Brier)

This is the main model class that prevents catastrophic forgetting.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
import copy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.phase1_baseline.model import LoRABaselineModel
from src.phase2_fisher_lora.joint_loss import FisherGuidedLoss


class FisherGuidedLoRA:
    """
    Fisher-Guided LoRA model for preventing catastrophic forgetting.

    Key Innovation:
    - Uses Fisher Information to identify important parameters
    - Penalizes changes to important parameters during training
    - Maintains uncertainty calibration via Brier score

    Usage:
        # Initialize with baseline model and Fisher matrices
        model = FisherGuidedLoRA(
            base_model=baseline_model,
            fisher_path='outputs/.../fisher_matrices_all.pt',
            lambda_fisher=0.1,
            beta_brier=0.5
        )

        # Save initial parameters (before cardiology training)
        model.save_initial_params()

        # Train
        loss, loss_dict = model.train_step(batch)
    """

    def __init__(self,
                 base_model: LoRABaselineModel,
                 fisher_path: str,
                 lambda_fisher: float = 0.1,
                 beta_brier: float = 0.5,
                 device: str = None):
        """
        Initialize Fisher-Guided LoRA model.

        Args:
            base_model: Baseline LoRA model (from Phase 1)
            fisher_path: Path to Fisher matrices file
            lambda_fisher: Weight for Fisher regularization (default 0.1)
                          Higher = more protection, less learning
            beta_brier: Weight for Brier score (default 0.5)
                       Higher = better calibration
            device: Device to use (None = auto-detect)
        """
        print(f"\n{'='*70}")
        print(f"🔧 INITIALIZING FISHER-GUIDED LORA")
        print(f"{'='*70}")

        self.base_model = base_model
        self.model = base_model.model  # PEFT model
        self.tokenizer = base_model.tokenizer
        self.device = device or base_model.device

        self.lambda_fisher = lambda_fisher
        self.beta_brier = beta_brier

        print(f"  Lambda (Fisher): {lambda_fisher}")
        print(f"  Beta (Brier): {beta_brier}")
        print(f"  Device: {self.device}")

        # Load Fisher matrices
        print(f"\n📥 Loading Fisher matrices...")
        print(f"  Path: {fisher_path}")

        self.fisher_matrices = self._load_fisher_matrices(fisher_path)
        print(f"  ✅ Loaded {len(self.fisher_matrices)} Fisher matrices")

        # Extract LoRA parameters
        self.lora_params = self._get_lora_parameters()
        print(f"  ✅ Found {len(self.lora_params)} LoRA parameters")

        # Initial parameters (to be saved before training)
        self.initial_params = None

        # Loss function (will be initialized after saving initial params)
        self.loss_fn = None

        print(f"\n✅ Fisher-Guided LoRA initialized!")

    def _load_fisher_matrices(self, fisher_path: str) -> Dict[str, torch.Tensor]:
        """
        Load Fisher matrices from disk.

        Args:
            fisher_path: Path to fisher_matrices_all.pt file

        Returns:
            Dictionary of Fisher matrices
        """
        fisher_path = Path(fisher_path)

        if not fisher_path.exists():
            raise FileNotFoundError(
                f"Fisher matrices not found at: {fisher_path}\n"
                f"Please run: python src/phase2_fisher_lora/compute_fisher.py"
            )

        fisher_matrices = torch.load(fisher_path, map_location=self.device)

        return fisher_matrices

    def _get_lora_parameters(self) -> Dict[str, nn.Parameter]:
        """
        Extract all LoRA parameters from the model.

        Returns:
            Dictionary mapping parameter names to parameters
        """
        lora_params = {}

        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                if param.requires_grad:
                    lora_params[name] = param

        return lora_params

    def save_initial_params(self):
        """
        Save initial LoRA parameters before cardiology training.

        This captures the parameter values AFTER general medicine evaluation
        but BEFORE cardiology fine-tuning.

        Must be called before training starts!
        """
        print(f"\n{'='*70}")
        print(f"💾 SAVING INITIAL LORA PARAMETERS")
        print(f"{'='*70}")

        self.initial_params = {}

        for name, param in self.lora_params.items():
            # Deep copy to ensure we save the exact current values
            self.initial_params[name] = param.data.clone().detach()

        print(f"  ✅ Saved {len(self.initial_params)} initial parameters")

        # Initialize loss function now that we have initial params
        self._initialize_loss_function()

    def _initialize_loss_function(self):
        """
        Initialize the joint loss function.

        Requires:
        - Fisher matrices (loaded in __init__)
        - Initial parameters (saved in save_initial_params)
        """
        print(f"\n{'='*70}")
        print(f"🔧 INITIALIZING JOINT LOSS FUNCTION")
        print(f"{'='*70}")

        # For each LoRA parameter, get Fisher matrix and initial value
        # The loss function expects separate A and B matrices,
        # but we'll handle all parameters uniformly

        # Since we have multiple LoRA layers, we need to aggregate them
        # Option 1: Create separate loss for each layer (complex)
        # Option 2: Create single loss with all params (simpler)

        # We'll use a modified loss that handles all params at once
        self.loss_fn = FisherGuidedLossWithDict(
            fisher_dict=self.fisher_matrices,
            initial_params_dict=self.initial_params,
            lambda_fisher=self.lambda_fisher,
            beta_brier=self.beta_brier
        )

        print(f"  ✅ Loss function initialized")
        print(f"  Parameters tracked: {len(self.initial_params)}")

    def train_step(self,
                   batch: list,
                   optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, Dict]:
        """
        Perform one training step.

        Args:
            batch: List of training samples
                  Each sample: {'question', 'options', 'answer', ...}
            optimizer: PyTorch optimizer

        Returns:
            total_loss: Total loss (scalar tensor)
            loss_dict: Dictionary with loss components
                      {'total', 'task', 'fisher', 'brier'}
        """
        if self.loss_fn is None:
            raise RuntimeError(
                "Loss function not initialized. "
                "Call save_initial_params() before training!"
            )

        # Set model to training mode
        self.model.train()

        # Zero gradients
        optimizer.zero_grad()

        # Process batch
        batch_loss = 0.0
        batch_loss_dict = {'total': 0.0, 'task': 0.0, 'fisher': 0.0, 'brier': 0.0}

        for sample in batch:
            # Get logits and targets
            logits, targets = self._prepare_sample(sample)

            # Get current LoRA parameters
            current_params = {name: param for name, param in self.lora_params.items()}

            # Compute loss
            loss, loss_dict = self.loss_fn.compute_loss(
                logits=logits,
                targets=targets,
                current_params_dict=current_params
            )

            batch_loss += loss
            for key in batch_loss_dict:
                batch_loss_dict[key] += loss_dict[key]

        # Average over batch
        batch_size = len(batch)
        batch_loss /= batch_size
        for key in batch_loss_dict:
            batch_loss_dict[key] /= batch_size

        # Backward pass
        batch_loss.backward()

        # Optimizer step
        optimizer.step()

        return batch_loss, batch_loss_dict

    def _prepare_sample(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a single sample for training.

        Args:
            sample: Training sample with 'question', 'options', 'answer'

        Returns:
            logits: Model predictions [batch_size=1, num_classes]
            targets: Ground truth label [batch_size=1]
        """
        # Format prompt
        prompt = self.base_model.format_qa_prompt(
            sample['question'],
            sample['options']
        )

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get answer key (A, B, C, or D)
        answer_key = sample['answer']

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Get logits for the answer position (last token)
        # Shape: [batch_size=1, seq_len, vocab_size]
        last_token_logits = logits[:, -1, :]  # [1, vocab_size]

        # For multiple choice, we want probabilities for A, B, C, D tokens
        # Get token IDs for each option
        option_tokens = {
            'A': self.tokenizer.encode('A', add_special_tokens=False)[0],
            'B': self.tokenizer.encode('B', add_special_tokens=False)[0],
            'C': self.tokenizer.encode('C', add_special_tokens=False)[0],
            'D': self.tokenizer.encode('D', add_special_tokens=False)[0],
        }

        # Extract logits for option tokens
        option_logits = torch.stack([
            last_token_logits[0, option_tokens['A']],
            last_token_logits[0, option_tokens['B']],
            last_token_logits[0, option_tokens['C']],
            last_token_logits[0, option_tokens['D']],
        ]).unsqueeze(0)  # [1, 4]

        # Convert answer to index (A=0, B=1, C=2, D=3)
        answer_idx = ord(answer_key) - ord('A')
        targets = torch.tensor([answer_idx], device=self.device)

        return option_logits, targets

    def evaluate(self, test_data: list) -> Dict:
        """
        Evaluate model on test data.

        Reuses the evaluation logic from baseline model.

        Args:
            test_data: List of test samples

        Returns:
            Dictionary with metrics (accuracy, ECE, Brier score)
        """
        # Use baseline model's evaluation
        return self.base_model.predict_batch(test_data)

    def save_model(self, output_path: str):
        """
        Save the Fisher-LoRA model.

        Args:
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'lambda_fisher': self.lambda_fisher,
            'beta_brier': self.beta_brier,
            'initial_params': self.initial_params,
        }

        torch.save(checkpoint, output_path)
        print(f"\n✅ Model saved to: {output_path}")


class FisherGuidedLossWithDict:
    """
    Fisher-Guided loss that handles parameter dictionaries.

    This is a wrapper around the base FisherGuidedLoss that works
    with named parameter dictionaries (as returned by model.named_parameters()).
    """

    def __init__(self,
                 fisher_dict: Dict[str, torch.Tensor],
                 initial_params_dict: Dict[str, torch.Tensor],
                 lambda_fisher: float = 0.1,
                 beta_brier: float = 0.5):
        """
        Initialize loss with parameter dictionaries.

        Args:
            fisher_dict: Dictionary mapping param names to Fisher matrices
            initial_params_dict: Dictionary of initial parameter values
            lambda_fisher: Fisher regularization weight
            beta_brier: Brier score weight
        """
        self.fisher_dict = fisher_dict
        self.initial_params = initial_params_dict
        self.lambda_fisher = lambda_fisher
        self.beta_brier = beta_brier

    def compute_loss(self,
                    logits: torch.Tensor,
                    targets: torch.Tensor,
                    current_params_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute joint loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            current_params_dict: Current parameter values

        Returns:
            total_loss: Combined loss
            loss_dict: Individual components
        """
        # 1. Task loss (cross-entropy)
        task_loss = nn.functional.cross_entropy(logits, targets)

        # 2. Fisher regularization
        fisher_loss = 0.0
        num_params_with_fisher = 0

        for name, param in current_params_dict.items():
            if name in self.fisher_dict and name in self.initial_params:
                # Compute change from initial
                delta = param - self.initial_params[name]

                # Weight by Fisher importance
                fisher_penalty = torch.sum(self.fisher_dict[name] * (delta ** 2))
                fisher_loss += fisher_penalty
                num_params_with_fisher += 1

        # 3. Brier score
        probs = torch.softmax(logits, dim=-1)
        num_classes = probs.shape[-1]
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        brier_loss = torch.mean((probs - targets_onehot) ** 2)

        # 4. Combine
        total_loss = (task_loss +
                     self.lambda_fisher * fisher_loss +
                     self.beta_brier * brier_loss)

        # Loss breakdown
        loss_dict = {
            'total': total_loss.item(),
            'task': task_loss.item(),
            'fisher': fisher_loss.item() if isinstance(fisher_loss, torch.Tensor) else fisher_loss,
            'brier': brier_loss.item()
        }

        return total_loss, loss_dict
