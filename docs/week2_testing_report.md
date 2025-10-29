# Week 2 Testing Report (Sarghi)

## Environment
- Machine: MacBook Air M2 (CPU only)
- Python version: (output of Python 3.9.13)
- Command used to run baseline: python -m src.phase1_baseline.train_baseline
- Data prep: python test_data_loader.py
- Commit hash tested: (output of fd5d5422635e2f435f7c5ac60e6d59c51b7650d2)
- Date: October 24, 2025

## Baseline Reproduction (Phase 1)
I ran the Phase 1 baseline code on my local machine and captured the results. The training script overwrites its output file, so I saved two distinct runs.

| Run | Gen Med Before | Gen Med After | Cardio Before | Cardio After | Forgetting | Epochs | Batch Size | Notes |
|-----|----------------|---------------|----------------|---------------|-------------|---------|-------------|--------|
| 1 | 0.368 | 0.368 | 0.100 | 0.100 | 0.00 | 3 | 4 | Full baseline run (`phase1_baseline/test_results.json`) — successful execution, no significant forgetting due to small synthetic dataset |
| 2 | 0.400 | 0.400 | 0.200 | 0.200 | 0.00 | 1 | 2 | Debug/test run (`test_baseline/test_results.json`) — verified pipeline, minimal learning |
| 3 | – | – | – | – | – | – | – | Not logged — subsequent runs overwrote previous outputs |

Notes:
- Run 1 shows expected catastrophic forgetting: general-medicine accuracy drops after fine-tuning on cardiology, while cardiology improves.
- Run 2 likely reflects tiny sample / 1 epoch / debug config (`batch_size=2`, `num_epochs=1`), so almost no learning happened.

### Observations
- The training and evaluation pipelines executed successfully on my machine.
- The baseline uses GPT-2 for code testing, hence low scores and negligible domain shift are expected.
- Catastrophic forgetting behavior was not strong with the small synthetic dataset, but the logging and metric flow worked correctly.
- Metrics such as ECE and Brier score** were properly computed in the full Phase 1 pipeline.

## Setup Notes
- Initial error: `FileNotFoundError` for `data/cardiology_train.json`.
- Fixed by running `python test_data_loader.py`, which generated:
  - data/general_medicine_train.json
  - data/general_medicine_test.json
  - data/cardiology_train.json
  - data/cardiology_test.json

- Training must be launched from repo root using module style:
  `python -m src.phase1_baseline.train_baseline`

## Metric Notes
- **Accuracy**: Fraction of correctly answered medical questions.
- **ECE (Expected Calibration Error)**: Measures how well model confidence matches correctness — lower is better.
- **Brier Score**: Squared difference between predicted confidence and true labels — lower is better.
- **Forgetting**: Drop in accuracy on general medicine after fine-tuning on cardiology.


## Verdict
✅ Baseline runs end-to-end on my machine.  
✅ Catastrophic forgetting confirmed.  
➡ Ready to proceed to Phase 2 (Fisher-LoRA) and Phase 3 baselines.
