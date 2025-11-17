"""
Joint loss function for Fisher-Guided LoRA
Combines: Task loss + Fisher regularization + Brier score

This is the core innovation of Fisher-Guided LoRA:
1. Task loss: Learn cardiology (cross-entropy)
2. Fisher loss: Prevent forgetting general medicine
3. Brier loss: Maintain uncertainty calibration
"""

import torch
import torch.nn as nn


class FisherGuidedLoss:
    """
    Joint loss function that combines three objectives:

    L_total = L_task + λ_fisher * L_fisher + β_brier * L_brier

    Where:
    - L_task: Standard cross-entropy for cardiology learning
    - L_fisher: Penalizes changes to important parameters (prevents forgetting)
    - L_brier: Maintains calibrated uncertainty estimates
    """

    def __init__(self, fisher_A, fisher_B, initial_lora_A, initial_lora_B,
                 lambda_fisher=0.1, beta_brier=0.5):
        """
        Initialize the joint loss function.

        Args:
            fisher_A: Fisher matrix for LoRA A matrices [shape: same as LoRA A params]
                     Higher values = more important for general medicine
            fisher_B: Fisher matrix for LoRA B matrices [shape: same as LoRA B params]
            initial_lora_A: Initial LoRA A parameters (before cardiology training)
                           Saved after general medicine evaluation
            initial_lora_B: Initial LoRA B parameters (before cardiology training)
            lambda_fisher: Weight for Fisher regularization (default 0.1)
                          Higher = more forgetting prevention, less cardiology learning
            beta_brier: Weight for Brier score (default 0.5)
                       Higher = better calibration, may affect accuracy
        """
        self.fisher_A = fisher_A
        self.fisher_B = fisher_B
        self.initial_A = initial_lora_A
        self.initial_B = initial_lora_B
        self.lambda_fisher = lambda_fisher
        self.beta_brier = beta_brier

        # Store these on the same device as the parameters
        if initial_lora_A is not None and hasattr(initial_lora_A, 'device'):
            self.device = initial_lora_A.device
        else:
            self.device = torch.device('cpu')

    def compute_loss(self, logits, targets, current_lora_A, current_lora_B):
        """
        Compute total loss = Task + Fisher + Brier

        This is called during each training step.

        Args:
            logits: Model predictions [batch_size, num_classes]
                   Raw logits before softmax
            targets: Ground truth labels [batch_size]
                    Integer class indices (0 to num_classes-1)
            current_lora_A: Current LoRA A parameters during training
                           Dictionary or tensor of A matrices
            current_lora_B: Current LoRA B parameters during training
                           Dictionary or tensor of B matrices

        Returns:
            total_loss: Combined loss value (single scalar tensor)
            loss_dict: Dictionary with individual loss components for logging
                      {'total', 'task', 'fisher', 'brier'}
        """
        # 1. Task loss (standard cross-entropy)
        # This encourages the model to learn cardiology
        task_loss = nn.functional.cross_entropy(logits, targets)

        # 2. Fisher regularization loss
        # This prevents forgetting of general medicine
        fisher_loss = self._compute_fisher_loss(current_lora_A, current_lora_B)

        # 3. Brier score loss
        # This maintains uncertainty calibration
        brier_loss = self._compute_brier_loss(logits, targets)

        # 4. Combine all three with weights
        total_loss = (task_loss +
                     self.lambda_fisher * fisher_loss +
                     self.beta_brier * brier_loss)

        # Return detailed breakdown for logging
        loss_dict = {
            'total': total_loss.item(),
            'task': task_loss.item(),
            'fisher': fisher_loss.item(),
            'brier': brier_loss.item()
        }

        return total_loss, loss_dict

    def _compute_fisher_loss(self, current_A, current_B):
        """
        Compute Fisher regularization: sum(Fisher_i * (theta_i - theta_initial)^2)

        Intuition:
        - If Fisher_i is high → parameter i is important for general medicine
        - If theta_i changes a lot → large penalty
        - Result: Important parameters stay close to initial values

        Args:
            current_A: Current LoRA A parameters
            current_B: Current LoRA B parameters

        Returns:
            fisher_penalty: Scalar tensor with Fisher regularization loss
        """
        # Change in LoRA A from initial values
        delta_A = current_A - self.initial_A

        # Weighted by Fisher importance
        # Element-wise: Fisher * (change)^2
        fisher_penalty_A = torch.sum(self.fisher_A * (delta_A ** 2))

        # Change in LoRA B
        delta_B = current_B - self.initial_B
        fisher_penalty_B = torch.sum(self.fisher_B * (delta_B ** 2))

        # Total Fisher penalty (sum over both A and B)
        return fisher_penalty_A + fisher_penalty_B

    def _compute_brier_loss(self, logits, targets):
        """
        Compute Brier score: mean((predicted_prob - true_label)^2)

        Intuition:
        - Brier score measures both accuracy AND calibration
        - Well-calibrated: If model says 70% confident, it's right 70% of the time
        - Lower Brier = better calibrated predictions

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            brier: Scalar tensor with Brier score
        """
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=-1)  # [batch_size, num_classes]

        # One-hot encode targets
        # E.g., if target=2 and num_classes=4 → [0, 0, 1, 0]
        num_classes = probs.shape[-1]
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Brier score: mean squared error between probabilities and one-hot labels
        # For each sample: sum over classes of (prob - true_label)^2
        brier = torch.mean((probs - targets_onehot) ** 2)

        return brier


class FisherGuidedLossV2:
    """
    Alternative implementation with support for named parameters.

    Use this if you have LoRA parameters stored as named dictionaries
    (e.g., {'layer.0.A': tensor, 'layer.1.A': tensor, ...})
    """

    def __init__(self, fisher_dict, initial_params_dict,
                 lambda_fisher=0.1, beta_brier=0.5):
        """
        Initialize with named parameter dictionaries.

        Args:
            fisher_dict: Dictionary mapping parameter names to Fisher matrices
                        e.g., {'lora_A.0': tensor, 'lora_B.0': tensor}
            initial_params_dict: Dictionary of initial LoRA parameters
            lambda_fisher: Fisher regularization weight
            beta_brier: Brier score weight
        """
        self.fisher_dict = fisher_dict
        self.initial_params = initial_params_dict
        self.lambda_fisher = lambda_fisher
        self.beta_brier = beta_brier

    def compute_loss(self, logits, targets, current_params_dict):
        """
        Compute loss with named parameters.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            current_params_dict: Current LoRA parameters (named dict)

        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # Task loss
        task_loss = nn.functional.cross_entropy(logits, targets)

        # Fisher loss over all parameters
        fisher_loss = 0.0
        for name, param in current_params_dict.items():
            if name in self.fisher_dict:
                delta = param - self.initial_params[name]
                fisher_loss += torch.sum(self.fisher_dict[name] * (delta ** 2))

        # Brier loss
        probs = torch.softmax(logits, dim=-1)
        num_classes = probs.shape[-1]
        targets_onehot = torch.zeros_like(probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        brier_loss = torch.mean((probs - targets_onehot) ** 2)

        # Combine
        total_loss = (task_loss +
                     self.lambda_fisher * fisher_loss +
                     self.beta_brier * brier_loss)

        loss_dict = {
            'total': total_loss.item(),
            'task': task_loss.item(),
            'fisher': fisher_loss.item() if isinstance(fisher_loss, torch.Tensor) else fisher_loss,
            'brier': brier_loss.item()
        }

        return total_loss, loss_dict
