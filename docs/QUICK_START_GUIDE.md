# Quick Start Guide for Teammates 🚀

**Read this first!** Everything you need to know in 5 minutes.

---

## 🎯 What Aniket Built (Week 1)

A complete pipeline that shows **catastrophic forgetting** in medical AI:

```
Before Training → Train on Cardiology → After Training
Gen Med: 75%        (LoRA training)      Gen Med: 62% ❌ FORGOT!
Cardiology: 40%                          Cardiology: 85% ✅ LEARNED!
```

This is the problem we're solving!

---

## 📁 What Files You'll Use

### Everyone Needs These (Don't Touch!)

```
src/utils/
├── data_loader.py    # Loads medical questions
└── metrics.py        # Calculates accuracy, ECE, Brier
```

### Phase 1 Baseline (Reference/Template)

```
src/phase1_baseline/
├── model.py              # LoRA model wrapper
├── evaluate.py           # Evaluation functions
└── train_baseline.py     # Main training script
```

### Outputs (Your Baseline to Beat)

```
outputs/phase1_baseline/
├── baseline_results.json    # Results to compare against
└── models/baseline_lora.pt  # Trained model
```

---

## 🚀 How to Run (3 Commands)

### 1. Setup (One-time)

```bash
cd DL_RESEARCH_IMPLEMENTATION
source dlenv/bin/activate
pip install -r requirements.txt
python quick_setup.py
```

### 2. Test It Works

```bash
python test_train_baseline.py
```

Should say: ✅ ALL TESTS PASSED!

### 3. Run Full Baseline

```bash
python run_baseline.py
```

Takes ~1 minute, generates results in `outputs/phase1_baseline/`

---

## 📊 Understanding the Results

Open `outputs/phase1_baseline/baseline_results.json`:

```json
{
  "before_training": {
    "general_medicine": { "accuracy": 0.75 }, // Before
    "cardiology": { "accuracy": 0.4 }
  },
  "after_training": {
    "general_medicine": { "accuracy": 0.62 }, // After: DROPPED!
    "cardiology": { "accuracy": 0.85 } // After: IMPROVED!
  },
  "forgetting": 0.13 // 13% forgetting = THE PROBLEM
}
```

**Your job:** Make YOUR method have lower forgetting!

---

## 👥 What YOU Need to Do

### Sarghi (Week 2) - Testing

**Goal:** Verify Aniket's code works on your machine

```bash
# 1. Run tests
python test_train_baseline.py

# 2. Run baseline 3 times
python run_baseline.py  # Run 1
python run_baseline.py  # Run 2
python run_baseline.py  # Run 3

# 3. Check: Are results consistent?

# 4. Document in: docs/week2_testing_report.md
```

---

### Aniket (Week 3-7) - Phase 2: Fisher-LoRA

**Goal:** Build Fisher-Guided LoRA (the main contribution)

**What to copy:**

- Copy `src/phase1_baseline/` as template
- Keep `src/utils/` exactly as-is

**What to create:**

```
src/phase2_fisher_lora/
├── fisher_computation.py  # Calculate Fisher matrix
├── fisher_lora.py         # Your Fisher-LoRA class
└── train_fisher_lora.py   # Training with Fisher
```

**Your target:**

```
Baseline:      13% forgetting
Fisher-LoRA:   <5% forgetting  ← GOAL!
```

---

### Sarghi (Week 3-7) - Phase 3: Other Baselines

**Goal:** Implement EWC-LoRA and I-LoRA for comparison

**What to copy:**

- Copy `src/phase1_baseline/` as template
- Keep `src/utils/` exactly as-is

**What to create:**

```
src/phase3_baselines/
├── standard_lora.py  # Copy from phase1_baseline
├── ewc_lora.py       # NEW
└── i_lora.py         # NEW
```

**Expected results:**

```
Standard LoRA: 13% forgetting
EWC-LoRA:      ~7% forgetting
I-LoRA:        ~9% forgetting
```

---

### Charan (Ongoing) - Documentation

**Goal:** Document everything for the paper

**Create these files:**

```
docs/
├── progress_log.md        # Weekly updates
├── phase1_report.md       # Phase 1 summary
├── phase2_report.md       # Phase 2 summary
├── phase3_report.md       # Phase 3 summary
└── meeting_notes.md       # Team meetings
```

**Track:**

- What was completed
- What worked/didn't work
- Key decisions
- Results from each phase

---

## 🎯 The Big Picture

```
Phase 1 (Week 1-2): Build baseline that shows forgetting
    ↓
Phase 2 (Week 3-7): Build Fisher-LoRA (Aniket)
Phase 3 (Week 3-7): Build other baselines (Sarghi)
    ↓
Phase 4 (Week 8-11): Compare everything
    ↓
Paper: Prove Fisher-LoRA works best!
```

---

## ⚠️ Important Notes

### Why No Results with GPT-2?

GPT-2 = Testing model (no medical knowledge)  
BioMistral = Production model (has medical knowledge)

**Current:** Using GPT-2 for code testing  
**Later:** Switch to BioMistral for real results

### All Methods Must Use Same Model!

For fair comparison in paper:

```
✅ All use: BioMistral-7B
✅ All use: Same data
✅ All use: Same evaluation
```

---

## 🐛 Quick Troubleshooting

**"ModuleNotFoundError"**  
→ Run from project root: `cd DL_RESEARCH_IMPLEMENTATION`

**"File not found"**  
→ Run: `python quick_setup.py`

**"Out of memory"**  
→ Normal on CPU, will work on GPU

---

## 📞 Need Help?

1. Read: `docs/PHASE1_PROGRESS_REPORT.md` (full details)
2. Check: `src/phase1_baseline/README.md` (Phase 1 docs)
3. Ask: Aniket (code) or team meeting (general)

---

## ✅ Week 2 Checklist (Sarghi)

- [ ] Code runs on my machine
- [ ] All tests pass
- [ ] Baseline runs 3 times successfully
- [ ] Results are consistent
- [ ] Issues documented (if any)
- [ ] Ready to approve for Phase 2/3

---

## 🚀 Ready to Start!

**Everything is set up. Just follow the steps above!**

Questions? Check the full report: `docs/PHASE1_PROGRESS_REPORT.md`
