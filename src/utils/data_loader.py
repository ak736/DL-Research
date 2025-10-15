"""
MedQA Dataset Loader for Fisher-Guided LoRA Project
Phase 1 Foundation - Used by ALL phases

This module loads and prepares medical QA data for:
- Phase 1: Baseline LoRA training
- Phase 2: Fisher-LoRA implementation  
- Phase 3: Baseline comparisons (EWC, I-LoRA, etc.)
- Phase 4: Full experiments

Output Format (Standardized for all phases):
{
    'question': str,
    'options': dict,  # {'A': ..., 'B': ..., 'C': ..., 'D': ...}
    'answer': str,    # 'A', 'B', 'C', or 'D'
    'domain': str,    # 'general_medicine' or 'cardiology'
    'question_id': str
}
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# For HuggingFace datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


class MedQADataLoader:
    """
    Load and prepare MedQA dataset with domain filtering.

    Key Features:
    - Loads from HuggingFace or local JSON
    - Filters by medical domain (general medicine vs cardiology)
    - Creates train/test splits
    - Saves/loads preprocessed data for reproducibility
    - Standardized output format for all phases
    """

    def __init__(self, data_dir: str = "data", seed: int = 42):
        """
        Initialize data loader.

        Args:
            data_dir: Directory to save/load data
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)

        # Cardiology keywords for filtering
        self.cardiology_keywords = [
            'heart', 'cardiac', 'cardio', 'ecg', 'ekg', 'myocardial',
            'arrhythmia', 'coronary', 'atrial', 'ventricular', 'valve',
            'angina', 'infarction', 'hypertension', 'blood pressure',
            'pericardial', 'endocarditis', 'cardiomyopathy'
        ]

    def download_medqa(self, num_samples: int = None):
        """
        Download MedQA dataset from HuggingFace.

        Args:
            num_samples: Limit number of samples (None = all)

        Returns:
            List of samples in standardized format
        """
        if not HF_AVAILABLE:
            raise ImportError("Please install: pip install datasets")

        print(f"📥 Downloading MedQA dataset...")

        # Try multiple dataset sources (in order of preference)
        datasets_to_try = [
            ("GBaker/MedQA-USMLE-4-options", None),
            ("bigbio/med_qa", "med_qa_en_bigbio_qa"),
        ]

        dataset = None
        for dataset_name, config in datasets_to_try:
            try:
                print(f"  Trying {dataset_name}...")
                if config:
                    dataset = load_dataset(
                        dataset_name, config, split="train", trust_remote_code=False)
                else:
                    dataset = load_dataset(
                        dataset_name, split="train", trust_remote_code=False)
                print(f"  ✅ Loaded from {dataset_name}")
                break
            except Exception as e:
                print(f"  ❌ Failed: {str(e)[:50]}...")
                continue

        if dataset is None:
            raise RuntimeError(
                "Could not download MedQA dataset. Check internet connection.")

        samples = []
        for idx, item in enumerate(dataset):
            if num_samples and idx >= num_samples:
                break

            # Convert to standardized format
            sample = self._format_sample(item, idx)
            if sample:
                samples.append(sample)

        print(f"✅ Loaded {len(samples)} samples")
        return samples

    def _format_sample(self, item: Dict, idx: int) -> Dict:
        """Convert raw dataset item to standardized format."""
        try:
            # Extract question text (try different field names)
            question = item.get('question', item.get(
                'QUESTION', item.get('query', '')))

            # Extract options (format varies by dataset)
            options_raw = item.get('options', item.get(
                'CONTEXTS', item.get('choices', {})))

            if isinstance(options_raw, dict):
                options = options_raw
            elif isinstance(options_raw, list):
                # Convert list to dict with A, B, C, D keys
                if len(options_raw) > 0 and isinstance(options_raw[0], dict):
                    # Format: [{'key': 'A', 'value': '...'}, ...]
                    options = {item['key']: item.get('value', item.get('text', ''))
                               for item in options_raw if 'key' in item}
                else:
                    # Format: ['option1', 'option2', ...]
                    options = {chr(65+i): opt for i,
                               opt in enumerate(options_raw[:4])}
            else:
                return None

            # Extract answer (try different formats)
            answer = item.get('answer', item.get(
                'ANSWER', item.get('answer_idx', '')))
            if isinstance(answer, int):
                answer = chr(65 + answer)  # 0->A, 1->B, etc.
            elif isinstance(answer, str) and len(answer) > 1:
                # If answer is full text, find matching option
                for key, value in options.items():
                    if value.strip().lower() == answer.strip().lower():
                        answer = key
                        break

            # Ensure answer is valid
            if answer not in options:
                answer = 'A'  # Default fallback

            # Determine domain based on keywords
            domain = self._classify_domain(question, options)

            return {
                'question': question,
                'options': options,
                'answer': answer,
                'domain': domain,
                'question_id': f"medqa_{idx}"
            }
        except Exception as e:
            print(f"⚠️  Warning: Failed to format sample {idx}: {e}")
            return None

    def _classify_domain(self, question: str, options: Dict) -> str:
        """
        Classify question as 'cardiology' or 'general_medicine'.

        Simple keyword-based classification.
        For more sophisticated classification, could use a classifier.
        """
        text = (question + ' ' + ' '.join(options.values())).lower()

        # Check for cardiology keywords
        for keyword in self.cardiology_keywords:
            if keyword in text:
                return 'cardiology'

        return 'general_medicine'

    def filter_by_domain(self, samples: List[Dict], domain: str) -> List[Dict]:
        """Filter samples by medical domain."""
        filtered = [s for s in samples if s['domain'] == domain]
        print(f"📋 Filtered {len(filtered)} {domain} samples")
        return filtered

    def create_splits(self, samples: List[Dict],
                      train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """
        Create train/test splits.

        Args:
            samples: List of samples
            train_ratio: Ratio for training set

        Returns:
            (train_samples, test_samples)
        """
        random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)

        train = samples[:split_idx]
        test = samples[split_idx:]

        print(f"✂️  Split: {len(train)} train, {len(test)} test")
        return train, test

    def save_data(self, samples: List[Dict], filename: str):
        """Save samples to JSON file."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(samples)} samples to {filepath}")

    def load_data(self, filename: str) -> List[Dict]:
        """Load samples from JSON file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        print(f"📂 Loaded {len(samples)} samples from {filepath}")
        return samples

    def prepare_phase1_data(self, num_samples: int = 1000,
                            test_ratio: float = 0.2) -> Dict:
        """
        Prepare data for Phase 1 baseline.

        This is the main function that creates ALL the data needed.

        Args:
            num_samples: Total samples to use
            test_ratio: Ratio for test set

        Returns:
            Dictionary with all data splits
        """
        print("\n" + "="*60)
        print("🚀 PHASE 1 DATA PREPARATION")
        print("="*60 + "\n")

        # Check if already prepared
        cache_file = self.data_dir / "phase1_cache.json"
        if cache_file.exists():
            print("📦 Found cached data, loading...")
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print("✅ Loaded from cache\n")
            return cache

        # Download full dataset
        all_samples = self.download_medqa(
            num_samples=num_samples*2)  # Get extra for filtering

        # Separate by domain
        general_med = self.filter_by_domain(all_samples, 'general_medicine')
        cardiology = self.filter_by_domain(all_samples, 'cardiology')

        # Limit samples
        general_med = general_med[:num_samples]
        cardiology = cardiology[:num_samples//2]  # Fewer cardiology samples

        # Create splits for general medicine
        gen_train, gen_test = self.create_splits(general_med, 1 - test_ratio)

        # Create splits for cardiology
        card_train, card_test = self.create_splits(cardiology, 1 - test_ratio)

        # Package data
        data = {
            'general_medicine_train': gen_train,
            'general_medicine_test': gen_test,
            'cardiology_train': card_train,
            'cardiology_test': card_test,
            'metadata': {
                'total_samples': len(all_samples),
                'num_general_med': len(general_med),
                'num_cardiology': len(cardiology),
                'seed': self.seed,
                'test_ratio': test_ratio
            }
        }

        # Save individual files for convenience
        self.save_data(gen_train, 'general_medicine_train.json')
        self.save_data(gen_test, 'general_medicine_test.json')
        self.save_data(card_train, 'cardiology_train.json')
        self.save_data(card_test, 'cardiology_test.json')

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(data, f)

        print("\n" + "="*60)
        print("✅ DATA PREPARATION COMPLETE")
        print("="*60)
        self._print_summary(data)

        return data

    def _print_summary(self, data: Dict):
        """Print data summary."""
        print("\n📊 DATA SUMMARY:")
        print(
            f"  General Medicine Train: {len(data['general_medicine_train'])} samples")
        print(
            f"  General Medicine Test:  {len(data['general_medicine_test'])} samples")
        print(
            f"  Cardiology Train:       {len(data['cardiology_train'])} samples")
        print(
            f"  Cardiology Test:        {len(data['cardiology_test'])} samples")
        print(
            f"\n  Total: {sum(len(v) for k, v in data.items() if k != 'metadata')} samples")
        print()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Test data loader and prepare Phase 1 data.

    Run this script to:
    1. Download MedQA data
    2. Filter by domain
    3. Create train/test splits
    4. Save for all phases
    """

    print("🔬 Testing MedQA Data Loader\n")

    # Initialize loader
    loader = MedQADataLoader(data_dir="data", seed=42)

    # Prepare data (start with small sample for testing)
    print("Starting with 100 samples for testing...")
    data = loader.prepare_phase1_data(num_samples=100, test_ratio=0.2)

    # Test loading
    print("\n🧪 Testing data loading...")
    gen_med_train = loader.load_data('general_medicine_train.json')
    print(f"Successfully loaded {len(gen_med_train)} samples")

    # Show example
    print("\n📝 Example sample:")
    example = gen_med_train[0]
    print(f"Question: {example['question'][:100]}...")
    print(f"Options: {list(example['options'].keys())}")
    print(f"Answer: {example['answer']}")
    print(f"Domain: {example['domain']}")

    print("\n✅ Data loader test complete!")
    print("\n💡 Next step: Create the training script to use this data")
