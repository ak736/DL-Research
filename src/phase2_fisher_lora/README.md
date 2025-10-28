# Phase 2: Fisher-Guided LoRA

## Overview

Implementation of Fisher-Guided LoRA for preventing catastrophic forgetting in medical domain adaptation.

## Method

Fisher-Guided LoRA combines:
1. **Fisher Information**: Identifies important parameters for general medical knowledge
2. **LoRA Fine-tuning**: Efficient adaptation to cardiology domain
3. **Brier Score Optimization**: Maintains uncertainty calibration

## Files

- `fisher_computation.py` - Calculate Fisher Information Matrix
- `fisher_lora.py` - Fisher-Guided LoRA model implementation
- `train_fisher_lora.py` - Training script with Fisher constraints
- `evaluate.py` - Evaluation functions
- `config_fisher_lora.yaml` - Configuration file

## Quick Start

```bash
# Step 1: Compute Fisher Information
python src/phase2_fisher_lora/fisher_computation.py

# Step 2: Train Fisher-LoRA
python src/phase2_fisher_lora/train_fisher_lora.py

# Step 3: Check results
cat outputs/phase2_fisher_lora/results/fisher_lora_results.json
```

## Model

Using: **Meditron-7B (4-bit quantized)**
- Medical Llama 2 model
- Fits in 8GB RAM with quantization
- Strong medical QA performance

## Dataset

- General Medicine: 800 train, 200 test
- Cardiology: 400 train, 100 test
- Total: 1,500 samples

## Expected Results

**Baseline (Phase 1):**
- Forgetting: ~13%
- ECE: ~0.15

**Fisher-LoRA (Phase 2):**
- Forgetting: <5% ✅
- ECE: <0.08 ✅

## Implementation Timeline

- Week 3-4: Fisher computation
- Week 5-6: Fisher-LoRA training
- Week 7: Polish and documentation
