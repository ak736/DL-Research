# Phase 1: Baseline LoRA Implementation

**Goal:** Build baseline LoRA that demonstrates catastrophic forgetting problem.

---

## 📋 What This Phase Does

This phase creates a **standard LoRA baseline** that:

1. ✅ Trains on cardiology data
2. ❌ Forgets general medical knowledge (THE PROBLEM!)
3. 📊 Provides baseline results for comparison

**This is the problem Phase 2 (Fisher-LoRA) will solve!**

---

## 🏗️ Project Structure

```
phase1_baseline/
├── model.py              # LoRA model wrapper
├── train_baseline.py     # Main training script ⭐
├── evaluate.py           # Evaluation functions
└── README.md            # This file

outputs/
└── phase1_baseline/
    ├── models/
    │   └── baseline_lora.pt
    └── baseline_results.json
```

---

## 🚀 Quick Start

### 1. Setup (One-time)

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (if not done already)
python quick_setup.py --num-samples 100
```

### 2. Test Pipeline (Recommended First)

```bash
# Quick test with minimal data (~2-3 minutes)
python test_train_baseline.py
```

**Expected output:**

- Model loads ✅
- Training runs ✅
- Evaluation works ✅
- Results saved ✅

### 3. Run Full Baseline

```bash
# Full training (~10-15 minutes on CPU)
python src/phase1_baseline/train_baseline.py
```

---

## 📊 Expected Results

### What Success Looks Like

```
BEFORE Training:
  General Medicine: ~37% (random guessing with GPT-2)
  Cardiology:       ~10%

AFTER Training:
  General Medicine: ~30% (WORSE! ❌ Forgot!)
  Cardiology:       ~60% (Better! ✅ Learned!)

Forgetting: 7% drop
```

**This proves catastrophic forgetting exists!**

---

## 📁 Output Files

### `baseline_results.json`

```json
{
  "before_training": {
    "general_medicine": {
      "accuracy": 0.37,
      "ece": 0.5,
      "brier_score": 1.07
    },
    "cardiology": {
      "accuracy": 0.1,
      "ece": 0.73,
      "brier_score": 1.44
    }
  },
  "after_training": {
    "general_medicine": {
      "accuracy": 0.3, // DROPPED!
      "ece": 0.52,
      "brier_score": 1.15
    },
    "cardiology": {
      "accuracy": 0.6, // IMPROVED!
      "ece": 0.45,
      "brier_score": 0.82
    }
  },
  "forgetting": 0.07, // 7% forgetting
  "cardiology_improvement": 0.5
}
```

### `models/baseline_lora.pt`

- Trained LoRA adapters
- Can be loaded for testing
- Used as baseline for comparison

---

## 🔧 Configuration

Edit `train_baseline.py` to change:

```python
config = {
    'model_name': 'gpt2',        # Model to use
    'lora_r': 8,                 # LoRA rank
    'lora_alpha': 16,            # LoRA alpha
    'learning_rate': 2e-4,       # Learning rate
    'num_epochs': 3,             # Training epochs
    'batch_size': 4,             # Batch size
}
```

**For faster testing:** Reduce `num_epochs` to 1, `batch_size` to 2.

---

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size

```python
'batch_size': 2  # or 1
```

### Issue: Training too slow

**Solution:** Use fewer samples

```python
cardiology_train = cardiology_train[:50]  # Just 50 samples
```

### Issue: Model not improving

**Normal!** GPT-2 is not a medical model. Low accuracy is expected.
Switch to BioMistral in Week 1 Day 6 for real results.

---

## ✅ Phase 1 Checklist

Week 1 deliverables:

- [ ] Data loader works (`data_loader.py`)
- [ ] Metrics calculator works (`metrics.py`)
- [ ] Model initializes (`model.py`)
- [ ] Evaluation runs (`evaluate.py`)
- [ ] Training completes (`train_baseline.py`)
- [ ] Results show forgetting ❌
- [ ] Results show cardiology learning ✅
- [ ] Code documented and clean
- [ ] Ready for Sarghi testing (Week 2)

---

## 🔄 Handoff to Sarghi (Week 2)

**What Sarghi needs:**

1. All code in `src/phase1_baseline/`
2. Results in `outputs/phase1_baseline/`
3. This README

**What Sarghi will do:**

- Run `test_train_baseline.py` to verify
- Run full training 3 times (check consistency)
- Document results
- Report any bugs

---

## 📚 Code Overview

### `model.py`

- `LoRABaselineModel` class
- Wraps base model + LoRA adapters
- Methods: `predict()`, `predict_batch()`, `save_model()`

### `evaluate.py`

- `evaluate_on_dataset()` - Evaluate on any dataset
- `evaluate_general_and_cardiology()` - Both domains
- `compare_before_after()` - Calculate forgetting

### `train_baseline.py` ⭐

- `BaselineTrainer` class
- Main training loop
- Saves results and models

---

## 🎯 Next Steps

After Phase 1 complete:

**Week 2:** Sarghi tests baseline

**Week 3-7:**

- Aniket → Phase 2 (Fisher-LoRA)
- Sarghi → Phase 3 (Other baselines)

**Week 8+:** Integration and experiments

---

## 💡 Tips

1. **Start small:** Use `test_train_baseline.py` first
2. **Check outputs:** Verify `baseline_results.json` makes sense
3. **GPT-2 is fine:** Low accuracy is expected for testing
4. **Document issues:** Note any problems for team meeting

---

## 📞 Questions?

- Check team documentation in `docs/`
- Review `Plan.docx` for overall strategy
- Ask in team meeting

**Phase 1 Goal:** Prove forgetting exists. ✅
**Phase 2 Goal:** Fix it with Fisher-LoRA! 🚀
