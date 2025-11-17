"""
LoRA Baseline Model for Phase 1
Wrapper around base medical model + LoRA adapters

This is the BASELINE that shows the forgetting problem.
Later phases will improve upon this.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class LoRABaselineModel:
    """
    Baseline LoRA model for medical QA.

    This model will demonstrate catastrophic forgetting when
    fine-tuned on cardiology data.

    Key Components:
    - Base medical LLM (e.g., BioGPT, GPT-2)
    - LoRA adapters for efficient fine-tuning
    - Question-answering interface
    """

    def __init__(self,
                 model_name: str = "gpt2",
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 device: str = None,
                 load_in_4bit: bool = False):
        """
        Initialize LoRA baseline model.

        Args:
            model_name: HuggingFace model name
            lora_r: LoRA rank (smaller = fewer parameters)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA layers
            device: Device to use (None = auto-detect)
            load_in_4bit: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔧 Initializing LoRA Baseline Model")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Quantization: {'4-bit' if load_in_4bit else 'None'}")
        print(
            f"  LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

        # Load tokenizer
        print("\n📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with optional quantization
        print("📥 Loading base model...")

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None
            )

        # Configure LoRA
        print("🔗 Configuring LoRA adapters...")

        # Determine target modules based on model type
        if "meditron" in model_name.lower() or "llama" in model_name.lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "biogpt" in model_name.lower():
            target_modules = ["q_proj", "v_proj"]  # BioGPT uses GPT-2 architecture
        elif "gpt" in model_name.lower() and "biogpt" not in model_name.lower():
            target_modules = ["c_attn"]  # Standard GPT-2
        else:
            target_modules = ["q_proj", "v_proj"]  # Default for most models

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.to(self.device)

        # Print trainable parameters
        self._print_trainable_parameters()

        print("\n✅ Model initialized successfully!")

    def _print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"\n📊 Model Parameters:")
        print(
            f"  Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"  Total: {all_params:,}")

    def format_qa_prompt(self,
                         question: str,
                         options: Dict[str, str]) -> str:
        """
        Format medical QA as a prompt.

        Args:
            question: Medical question
            options: Dictionary of answer options

        Returns:
            Formatted prompt string
        """
        prompt = f"Question: {question}\n\n"
        prompt += "Options:\n"
        for key, value in sorted(options.items()):
            prompt += f"{key}. {value}\n"
        prompt += "\nAnswer:"

        return prompt

    def predict(self,
                question: str,
                options: Dict[str, str],
                max_new_tokens: int = 10) -> Dict:
        """
        Predict answer for a medical question.

        Args:
            question: Medical question
            options: Answer options
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Format prompt
        prompt = self.format_qa_prompt(question, options)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate with model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode prediction
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Extract answer (A, B, C, or D)
        prediction = self._extract_answer(generated_text, options)

        # Get probabilities for each option
        probabilities = self._get_option_probabilities(inputs, options)

        # Confidence = max probability
        confidence = max(probabilities.values())

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'raw_output': generated_text
        }

    def _extract_answer(self,
                        generated_text: str,
                        options: Dict[str, str]) -> str:
        """
        Extract answer letter from generated text.

        Args:
            generated_text: Generated text from model
            options: Valid options

        Returns:
            Answer letter (A, B, C, or D)
        """
        # Clean text
        text = generated_text.strip().upper()

        # Check if answer is at start
        for option in options.keys():
            if text.startswith(option):
                return option

        # Default to first option if can't parse
        return list(options.keys())[0]

    def _get_option_probabilities(self,
                                  inputs: Dict,
                                  options: Dict[str, str]) -> Dict[str, float]:
        """
        Get probability for each answer option.

        This is a simplified approach - we check the probability
        of generating each option letter.

        Args:
            inputs: Tokenized inputs
            options: Answer options

        Returns:
            Dictionary mapping option letters to probabilities
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get token IDs for A, B, C, D
        option_tokens = {
            opt: self.tokenizer.encode(f" {opt}", add_special_tokens=False)[0]
            for opt in options.keys()
        }

        # Get logits for each option
        option_logits = {
            opt: logits[token_id].item()
            for opt, token_id in option_tokens.items()
        }

        # Convert to probabilities via softmax
        logits_tensor = torch.tensor(list(option_logits.values()))
        probs = torch.softmax(logits_tensor, dim=0)

        probabilities = {
            opt: prob.item()
            for opt, prob in zip(option_logits.keys(), probs)
        }

        return probabilities

    def predict_batch(self,
                      samples: List[Dict],
                      max_new_tokens: int = 10) -> Dict:
        """
        Predict on a batch of samples.

        Args:
            samples: List of samples with 'question' and 'options'
            max_new_tokens: Max tokens to generate

        Returns:
            Dictionary with predictions, confidences, probabilities
        """
        predictions = []
        confidences = []
        probabilities = []

        print(f"🔮 Predicting on {len(samples)} samples...")

        for i, sample in enumerate(samples):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(samples)}")

            result = self.predict(
                sample['question'],
                sample['options'],
                max_new_tokens
            )

            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
            probabilities.append(result['probabilities'])

        print("  ✅ Predictions complete!")

        return {
            'predictions': predictions,
            'confidences': confidences,
            'probabilities': probabilities
        }

    def save_model(self, save_path: str):
        """Save LoRA adapters."""
        print(f"💾 Saving model to {save_path}...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print("  ✅ Model saved!")

    def load_model(self, load_path: str):
        """Load LoRA adapters."""
        print(f"📂 Loading model from {load_path}...")
        # This will be implemented when needed
        print("  ✅ Model loaded!")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Test LoRA baseline model.
    """
    print("🔬 Testing LoRA Baseline Model\n")

    # Initialize model (using GPT-2 for testing)
    model = LoRABaselineModel(
        model_name="gpt2",
        lora_r=8,
        lora_alpha=16
    )

    # Test question
    question = "A 65-year-old patient presents with chest pain. ECG shows ST elevation. What is the most likely diagnosis?"
    options = {
        'A': 'STEMI (ST-Elevation Myocardial Infarction)',
        'B': 'NSTEMI (Non-ST-Elevation Myocardial Infarction)',
        'C': 'Unstable Angina',
        'D': 'Stable Angina'
    }

    print("\n" + "="*70)
    print("📋 Test Question:")
    print(f"  {question}")
    print("\n  Options:")
    for k, v in options.items():
        print(f"    {k}. {v}")
    print("="*70)

    # Predict
    print("\n🔮 Making prediction...")
    result = model.predict(question, options)

    print(f"\n📊 Results:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for opt, prob in result['probabilities'].items():
        print(f"    {opt}: {prob:.2%}")

    print("\n✅ Model test complete!")
