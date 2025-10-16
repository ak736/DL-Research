# Visual Workflow Diagram

## 📊 Complete Project Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     WEEK 1-2: PHASE 1                           │
│                    (Aniket → Sarghi)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓

         ┌──────────────────────────────────┐
         │   1. Data Preparation            │
         │   python quick_setup.py          │
         │                                  │
         │   Output: data/*.json files      │
         └──────────────────────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │   2. Build Baseline LoRA         │
         │   (src/phase1_baseline/)         │
         │                                  │
         │   - model.py                     │
         │   - evaluate.py                  │
         │   - train_baseline.py            │
         └──────────────────────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │   3. Test Everything             │
         │   python test_train_baseline.py  │
         │                                  │
         │   Status: ✅ ALL TESTS PASSED    │
         └──────────────────────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │   4. Run Full Training           │
         │   python run_baseline.py         │
         │                                  │
         │   Output:                        │
         │   - baseline_results.json        │
         │   - baseline_lora.pt             │
         └──────────────────────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │   5. Sarghi Testing (Week 2)     │
         │   - Verify on own machine        │
         │   - Run 3 times                  │
         │   - Document results             │
         │                                  │
         │   Status: READY / NEEDS FIX      │
         └──────────────────────────────────┘

                         ↓
┌────────────────────────┴────────────────────────┐
│                                                  │
▼                                                  ▼

┌─────────────────────────────┐    ┌─────────────────────────────┐
│   WEEK 3-7: PHASE 2         │    │   WEEK 3-7: PHASE 3         │
│   (Aniket)                  │    │   (Sarghi)                  │
│                             │    │                             │
│   Fisher-Guided LoRA        │    │   Other Baselines           │
└─────────────────────────────┘    └─────────────────────────────┘
         │                                      │
         │  Uses:                              │  Uses:
         │  ✓ data_loader.py                  │  ✓ data_loader.py
         │  ✓ metrics.py                      │  ✓ metrics.py
         │  ✓ baseline as template            │  ✓ baseline as template
         │                                     │
         │  Creates:                           │  Creates:
         │  - fisher_computation.py            │  - ewc_lora.py
         │  - fisher_lora.py                   │  - i_lora.py
         │  - train_fisher_lora.py             │  - (cur_lora.py)
         │                                     │
         │  Goal:                              │  Goal:
         │  Forgetting < 5%                    │  Implement comparisons
         │                                     │
         └──────────────┬──────────────────────┘
                        ↓
         ┌──────────────────────────────────┐
         │   WEEK 8: INTEGRATION            │
         │   (Both)                         │
         │                                  │
         │   - Combine all methods          │
         │   - Ensure same interface        │
         │   - Test together                │
         └──────────────────────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │   WEEK 9-11: EXPERIMENTS         │
         │   (Sarghi leads)                 │
         │                                  │
         │   Run all methods:               │
         │   1. Standard LoRA               │
         │   2. EWC-LoRA                    │
         │   3. I-LoRA                      │
         │   4. Fisher-LoRA ⭐              │
         │                                  │
         │   Generate:                      │
         │   - Comparison tables            │
         │   - Plots                        │
         │   - Statistical tests            │
         └──────────────────────────────────┘
                         ↓
         ┌──────────────────────────────────┐
         │   WEEK 12-14: PAPER              │
         │   (All)                          │
         │                                  │
         │   Write paper showing:           │
         │   Fisher-LoRA WINS! 🎉          │
         └──────────────────────────────────┘
```

---

## 📂 File Dependencies

```
┌─────────────────────────┐
│  src/utils/             │  ← SHARED BY ALL PHASES
│  - data_loader.py       │     (Don't modify!)
│  - metrics.py           │
└─────────────────────────┘
          │
          ├──────────────┬──────────────┬──────────────┐
          ↓              ↓              ↓              ↓

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Phase 1     │  │ Phase 2     │  │ Phase 3     │  │ Phase 4     │
│ Baseline    │  │ Fisher-LoRA │  │ Baselines   │  │ Experiments │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
      │                │                │                │
      └────────────────┴────────────────┴────────────────┘
                           ↓
              ┌─────────────────────────┐
              │  outputs/               │
              │  - baseline_results     │
              │  - fisher_results       │
              │  - ewc_results          │
              │  - i_lora_results       │
              │  - comparison_tables    │
              │  - figures/             │
              └─────────────────────────┘
```

---

## 🔄 Data Flow Through Pipeline

```
INPUT: MedQA Dataset
         │
         ↓
┌─────────────────────┐
│  data_loader.py     │  Load & Split
└─────────────────────┘
         │
         ├─────────────────┬─────────────────┐
         ↓                 ↓                 ↓
   General Med        Cardiology      Cardiology
   Test (19)          Train (40)      Test (10)
         │                 │                 │
         │                 ↓                 │
         │         ┌──────────────┐         │
         │         │  Train Model │         │
         │         │  (LoRA)      │         │
         │         └──────────────┘         │
         │                 │                 │
         └────────┬────────┴─────────┬──────┘
                  ↓                  ↓
           ┌─────────────┐    ┌─────────────┐
           │  Evaluate   │    │  Evaluate   │
           │  (Before)   │    │  (After)    │
           └─────────────┘    └─────────────┘
                  │                  │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │  metrics.py      │  Calculate
                  │  - Accuracy      │
                  │  - ECE           │
                  │  - Brier Score   │
                  │  - Forgetting    │
                  └──────────────────┘
                           ↓
                  ┌──────────────────┐
                  │  Results JSON    │
                  └──────────────────┘
```

---

## 👥 Team Responsibilities Flow

```
ANIKET (Week 1)
    ↓ Builds baseline
    ↓ Tests everything
    ↓ Documents
    ↓
    └──→ Handoff to SARGHI

SARGHI (Week 2)
    ↓ Tests on own machine
    ↓ Validates results
    ↓ Reports issues
    ↓
    └──→ Approve / Request Fixes

TEAM MEETING (End Week 2)
    ↓ Review Phase 1
    ↓ Decision: GO / NO-GO
    ↓

┌─────────────────┴─────────────────┐
│                                    │
▼                                    ▼

ANIKET (Week 3-7)              SARGHI (Week 3-7)
Fisher-LoRA                    Other Baselines
    │                              │
    └──────────┬───────────────────┘
               ↓
        INTEGRATION (Week 8)
               ↓
     EXPERIMENTS (Week 9-11)
               ↓
         PAPER (Week 12-14)

CHARAN (Ongoing)
    ↓ Documents progress
    ↓ Takes meeting notes
    ↓ Prepares paper sections
```

---

## 🎯 Success Criteria at Each Phase

```
Phase 1 ✅
├─ Code runs without errors
├─ All tests pass
├─ Results saved correctly
├─ Demonstrates forgetting
└─ Sarghi can replicate

Phase 2 (Goal)
├─ Fisher matrix computed correctly
├─ Training with constraints works
├─ Forgetting < 5%
├─ Calibration maintained (ECE < 0.08)
└─ Code documented for Sarghi

Phase 3 (Goal)
├─ 3 baseline methods work
├─ All produce results in same format
├─ Results comparable
└─ Ready for integration

Phase 4 (Goal)
├─ All methods run on same data
├─ Fisher-LoRA outperforms baselines
├─ Statistical significance shown
└─ Paper-ready results & figures
```

---

## 📊 Expected Results Progression

```
                    Forgetting %
                         │
                    15% │  ┌─────┐
                        │  │Base │
                        │  │LoRA │
                    10% │  └─────┘
                        │     ┌──┐  ┌──┐
                        │     │I-│  │EWC
                     5% │     │LoRA │  │
                        │     └──┘  └──┘
                        │              ┌────┐
                     0% │              │Fish│
                        │              │LoRA│
                        └──────────────┴────┴─────→
                            Methods

                   ⭐ Fisher-LoRA should WIN!
```

---

## 🔧 Quick Command Reference

```bash
# Setup (one-time)
python quick_setup.py

# Test individual components
python test_data_loader.py
python test_metrics.py
python test_model.py
python test_evaluate.py
python test_train_baseline.py

# Run full pipeline
python run_baseline.py

# Check results
cat outputs/phase1_baseline/baseline_results.json
```

---

## 📍 Where You Are Now

```
✅ Week 1: COMPLETE
    ↓
→ Week 2: Testing (Sarghi) ← YOU ARE HERE
    ↓
  Week 3-7: Implementation
    ↓
  Week 8-11: Experiments
    ↓
  Week 12-14: Paper
```

**Next Action:** Sarghi runs tests and validates!
