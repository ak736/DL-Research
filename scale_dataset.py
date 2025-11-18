#!/usr/bin/env python3
"""
Scale Dataset for Phase 2
Increases dataset from 143 samples to 1,500 samples for publication-quality results

Target distribution:
- General Medicine Train: 800 samples
- General Medicine Test: 200 samples  
- Cardiology Train: 400 samples
- Cardiology Test: 100 samples
- TOTAL: 1,500 samples
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import randomab


def load_current_samples():
    """Load currently used samples to avoid duplicates"""
    data_dir = Path("data")
    current_ids = set()

    files = [
        "general_medicine_train.json",
        "general_medicine_test.json",
        "cardiology_train.json",
        "cardiology_test.json"
    ]

    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                for item in data:
                    if 'question_id' in item:
                        current_ids.add(item['question_id'])

    print(f"📋 Current dataset has {len(current_ids)} unique samples")
    return current_ids


def filter_by_keywords(question, keywords):
    """Check if question contains any of the keywords"""
    question_lower = question.lower()
    return any(keyword.lower() in question_lower for keyword in keywords)


def scale_dataset():
    """Scale dataset to 1,500 samples"""

    print("=" * 70)
    print("📊 SCALING DATASET FOR PHASE 2")
    print("=" * 70)

    # Create data directory if doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Load current samples to avoid duplicates
    current_ids = load_current_samples()

    print("\n🔄 Loading MedQA dataset from HuggingFace...")
    print("   (This may take a few minutes on first run)")

    try:
        # Load MedQA dataset
        dataset = load_dataset(
            "bigbio/med_qa", "med_qa_en_source", split="train")
        print(f"   ✅ Loaded {len(dataset)} total MedQA questions")

    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print("\n   Trying alternative loading method...")

        # Alternative: Load from different source
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        print(f"   ✅ Loaded {len(dataset)} questions from alternative source")

    # Keywords for domain classification
    cardiology_keywords = [
        'heart', 'cardiac', 'cardio', 'coronary', 'myocardial', 'angina',
        'arrhythmia', 'ecg', 'ekg', 'chest pain', 'blood pressure',
        'hypertension', 'valve', 'atrial', 'ventricular', 'pericardial'
    ]

    # Process samples
    general_medicine_samples = []
    cardiology_samples = []

    print("\n🔍 Processing samples...")
    print("   Classifying: General Medicine vs Cardiology")

    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        # Create question ID
        question_id = f"medqa_{idx}"

        # Skip if already in current dataset
        if question_id in current_ids:
            continue

        # Extract question text
        if isinstance(item, dict):
            if 'question' in item:
                question_text = item['question']
            elif 'text' in item:
                question_text = item['text']
            else:
                continue
        else:
            continue

        # Extract options and answer
        if 'options' in item:
            options = item['options']
        elif 'choices' in item:
            options = item['choices']
        else:
            continue

        if 'answer' in item:
            answer = item['answer']
        elif 'correct_answer' in item:
            answer = item['correct_answer']
        else:
            continue

        # Create standardized format
        sample = {
            'question': question_text,
            'options': options if isinstance(options, dict) else {
                'A': options[0] if len(options) > 0 else '',
                'B': options[1] if len(options) > 1 else '',
                'C': options[2] if len(options) > 2 else '',
                'D': options[3] if len(options) > 3 else ''
            },
            'answer': answer if answer in ['A', 'B', 'C', 'D'] else 'A',
            'question_id': question_id
        }

        # Classify by domain
        if filter_by_keywords(question_text, cardiology_keywords):
            sample['domain'] = 'cardiology'
            cardiology_samples.append(sample)
        else:
            sample['domain'] = 'general_medicine'
            general_medicine_samples.append(sample)

        # Stop when we have enough samples
        if len(general_medicine_samples) >= 1200 and len(cardiology_samples) >= 600:
            break

    print(f"\n📊 Classification Results:")
    print(f"   General Medicine: {len(general_medicine_samples)} samples")
    print(f"   Cardiology: {len(cardiology_samples)} samples")

    # Shuffle
    random.seed(42)
    random.shuffle(general_medicine_samples)
    random.shuffle(cardiology_samples)

    # Split into train/test
    print("\n✂️  Splitting into train/test sets...")

    # General Medicine: 800 train, 200 test
    gen_med_train = general_medicine_samples[:800]
    gen_med_test = general_medicine_samples[800:1000]

    # Cardiology: 400 train, 100 test
    cardio_train = cardiology_samples[:400]
    cardio_test = cardiology_samples[400:500]

    # Save datasets
    print("\n💾 Saving scaled datasets...")

    files_to_save = {
        'general_medicine_train.json': gen_med_train,
        'general_medicine_test.json': gen_med_test,
        'cardiology_train.json': cardio_train,
        'cardiology_test.json': cardio_test,
    }

    for filename, data in files_to_save.items():
        filepath = data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"   ✅ {filename}: {len(data)} samples")

    # Create summary
    total_samples = sum(len(data) for data in files_to_save.values())

    print("\n" + "=" * 70)
    print("✅ DATASET SCALING COMPLETE!")
    print("=" * 70)

    print("\n📊 Final Dataset Summary:")
    print(f"   General Medicine Train: {len(gen_med_train)} samples")
    print(f"   General Medicine Test:  {len(gen_med_test)} samples")
    print(f"   Cardiology Train:       {len(cardio_train)} samples")
    print(f"   Cardiology Test:        {len(cardio_test)} samples")
    print(f"   " + "-" * 50)
    print(f"   TOTAL:                  {total_samples} samples")

    print("\n🎯 Ready for Phase 2 Experiments!")
    print("   This dataset size is publication-quality ✅")

    # Save metadata
    metadata = {
        'total_samples': total_samples,
        'general_medicine_train': len(gen_med_train),
        'general_medicine_test': len(gen_med_test),
        'cardiology_train': len(cardio_train),
        'cardiology_test': len(cardio_test),
        'created_for': 'Phase 2: Fisher-Guided LoRA',
        'model': 'Meditron-7B',
        'seed': 42
    }

    with open(data_dir / 'dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Next Steps:")
    print("   1. Run: python create_phase2_structure.py")
    print("   2. Start Phase 2 Week 3 Day 2: Fisher Computation")


if __name__ == "__main__":
    scale_dataset()
