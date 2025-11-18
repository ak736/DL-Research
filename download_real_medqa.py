"""
Download Real MedQA Dataset from HuggingFace
This gets the actual medical licensing exam questions.
"""

from datasets import load_dataset
import json
from pathlib import Path
import random

def download_medqa():
    """Download and prepare real MedQA data."""

    print("📥 Downloading Real MedQA Dataset...")
    print("This may take a few minutes...\n")

    # Download MedQA from HuggingFace
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
        print(f"✅ Downloaded {len(dataset)} samples\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Try: pip install datasets")
        return

    # Split by domain (cardiology vs general)
    general_medicine = []
    cardiology = []

    cardiology_keywords = [
        'heart', 'cardiac', 'cardio', 'ecg', 'ekg', 'myocardial',
        'arrhythmia', 'coronary', 'atrial', 'ventricular', 'valve',
        'angina', 'infarction', 'pericardial', 'endocarditis'
    ]

    print("🔍 Splitting by domain...")
    for item in dataset:
        question = item['question'].lower()

        # Check if cardiology-related
        is_cardio = any(keyword in question for keyword in cardiology_keywords)

        # Format sample
        sample = {
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer_idx'],  # Usually 'A', 'B', 'C', or 'D'
            'domain': 'cardiology' if is_cardio else 'general_medicine'
        }

        if is_cardio:
            cardiology.append(sample)
        else:
            general_medicine.append(sample)

    print(f"  General Medicine: {len(general_medicine)} samples")
    print(f"  Cardiology: {len(cardiology)} samples\n")

    # Shuffle
    random.seed(42)
    random.shuffle(general_medicine)
    random.shuffle(cardiology)

    # Split train/test
    gen_med_train = general_medicine[:800]
    gen_med_test = general_medicine[800:1000]

    cardio_train = cardiology[:400]
    cardio_test = cardiology[400:500]

    # Create data directory
    Path("data").mkdir(exist_ok=True)

    # Save files
    with open('data/general_medicine_train.json', 'w') as f:
        json.dump(gen_med_train, f, indent=2)

    with open('data/general_medicine_test.json', 'w') as f:
        json.dump(gen_med_test, f, indent=2)

    with open('data/cardiology_train.json', 'w') as f:
        json.dump(cardio_train, f, indent=2)

    with open('data/cardiology_test.json', 'w') as f:
        json.dump(cardio_test, f, indent=2)

    print("✅ Real MedQA Data Saved!")
    print("\n📁 Files created:")
    print(f"  ✅ general_medicine_train.json ({len(gen_med_train)} samples)")
    print(f"  ✅ general_medicine_test.json ({len(gen_med_test)} samples)")
    print(f"  ✅ cardiology_train.json ({len(cardio_train)} samples)")
    print(f"  ✅ cardiology_test.json ({len(cardio_test)} samples)")
    print("\n🎯 Ready for training!\n")

if __name__ == "__main__":
    download_medqa()
