#!/usr/bin/env python3
"""
System Check for Phase 2 Implementation
Checks Mac capabilities, current data, and readiness for BioMistral-7B
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path


def check_mac_specs():
    """Check Mac specifications"""
    print("=" * 60)
    print("🖥️  MAC SYSTEM CHECK")
    print("=" * 60)

    # Basic system info
    print(f"\n📊 System Information:")
    print(f"   Platform: {platform.platform()}")
    print(f"   Python Version: {sys.version.split()[0]}")
    print(f"   Architecture: {platform.machine()}")

    # Check if Apple Silicon
    is_apple_silicon = platform.machine() == "arm64"
    print(
        f"   Apple Silicon: {'✅ YES (M1/M2/M3)' if is_apple_silicon else '❌ NO (Intel)'}")

    # Check RAM (macOS specific)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   Total RAM: {ram_gb:.1f} GB")

        if ram_gb >= 32:
            print("   ✅ Excellent for BioMistral-7B")
        elif ram_gb >= 16:
            print("   ⚠️  Adequate for BioMistral-7B (may be slow)")
        else:
            print("   ❌ Insufficient for BioMistral-7B")
    except ImportError:
        print("   ⚠️  Install psutil to check RAM: pip install psutil")

    # Check MPS (Metal Performance Shaders) availability
    try:
        import torch
        print(f"\n🔥 PyTorch Check:")
        print(f"   PyTorch Version: {torch.__version__}")
        print(
            f"   MPS Available: {'✅ YES' if torch.backends.mps.is_available() else '❌ NO'}")
        print(
            f"   MPS Built: {'✅ YES' if torch.backends.mps.is_built() else '❌ NO'}")

        if torch.backends.mps.is_available():
            print("\n   💡 You can use MPS (Metal) for acceleration!")
            print("   📝 Use device='mps' in model loading")
        else:
            print("\n   ⚠️  MPS not available, will use CPU (slower)")
    except ImportError:
        print("   ❌ PyTorch not installed")

    return is_apple_silicon


def check_current_data():
    """Check current dataset"""
    print("\n" + "=" * 60)
    print("📂 CURRENT DATASET CHECK")
    print("=" * 60)

    data_dir = Path("data")

    if not data_dir.exists():
        print("❌ Data directory not found!")
        return None

    files_to_check = [
        "general_medicine_train.json",
        "general_medicine_test.json",
        "cardiology_train.json",
        "cardiology_test.json"
    ]

    total_samples = 0
    dataset_info = {}

    print("\n📊 Current Dataset:")
    for filename in files_to_check:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                num_samples = len(data)
                total_samples += num_samples
                dataset_info[filename] = num_samples
                print(f"   ✅ {filename}: {num_samples} samples")
        else:
            print(f"   ❌ {filename}: NOT FOUND")
            dataset_info[filename] = 0

    print(f"\n📈 Total Samples: {total_samples}")

    # Check if Phase 1 baseline results exist
    baseline_results = Path("outputs/phase1_baseline/baseline_results.json")
    if baseline_results.exists():
        print("\n✅ Phase 1 baseline results found")
        with open(baseline_results, 'r') as f:
            results = json.load(f)
            print("\n📊 Phase 1 Results:")
            if 'before_training' in results:
                print(
                    f"   General Med (before): {results['before_training']['general_medicine']['accuracy']:.2%}")
                print(
                    f"   Cardiology (before): {results['before_training']['cardiology']['accuracy']:.2%}")
            if 'after_training' in results:
                print(
                    f"   General Med (after): {results['after_training']['general_medicine']['accuracy']:.2%}")
                print(
                    f"   Cardiology (after): {results['after_training']['cardiology']['accuracy']:.2%}")
            if 'forgetting' in results:
                print(f"   Forgetting: {results['forgetting']:.2%}")
    else:
        print("\n⚠️  Phase 1 baseline results not found")

    return dataset_info


def check_disk_space():
    """Check available disk space"""
    print("\n" + "=" * 60)
    print("💾 DISK SPACE CHECK")
    print("=" * 60)

    try:
        import shutil
        total, used, free = shutil.disk_usage("/")

        free_gb = free / (1024**3)
        print(f"\n   Free Space: {free_gb:.1f} GB")

        required_gb = 30  # BioMistral + checkpoints + data
        if free_gb >= required_gb:
            print(f"   ✅ Sufficient space (need ~{required_gb}GB)")
        else:
            print(f"   ⚠️  Low space (need ~{required_gb}GB)")
    except:
        print("   ⚠️  Could not check disk space")


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 60)
    print("📦 DEPENDENCIES CHECK")
    print("=" * 60)

    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'datasets': 'HuggingFace Datasets',
        'numpy': 'NumPy',
        'tqdm': 'Progress Bars',
        'scikit-learn': 'Scikit-learn',
    }

    print("\n📋 Required Packages:")
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            all_installed = False

    return all_installed


def recommendations_for_mac(is_apple_silicon, dataset_info):
    """Provide recommendations based on Mac setup"""
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS FOR YOUR MAC")
    print("=" * 60)

    if is_apple_silicon:
        print("\n✅ Apple Silicon Mac - Good for Phase 2!")
        print("\n📝 Optimizations:")
        print("   1. Use device='mps' for Metal acceleration")
        print("   2. Enable unified memory efficiently")
        print("   3. Batch size: Start with 1-2")
        print("\n⏱️  Expected Times (with MPS):")
        print("   - Fisher computation (1000 samples): ~2-3 hours")
        print("   - Training per epoch (500 samples): ~30-40 minutes")
        print("   - Total Phase 2: ~15-20 hours")
    else:
        print("\n⚠️  Intel Mac - Will be slower")
        print("\n📝 Optimizations:")
        print("   1. Use CPU only")
        print("   2. Reduce batch size to 1")
        print("   3. Consider using smaller model or cloud GPU")
        print("\n⏱️  Expected Times (CPU only):")
        print("   - Fisher computation: ~6-8 hours")
        print("   - Training per epoch: ~2-3 hours")
        print("   - Total Phase 2: ~40-50 hours (not recommended)")

    # Data recommendations
    if dataset_info:
        total_samples = sum(dataset_info.values())
        print(f"\n📊 Current Dataset: {total_samples} samples")

        if total_samples < 200:
            print("\n⚠️  Dataset is too small for publication!")
            print("📝 Recommendations:")
            print("   1. Scale up to 1,500+ samples for top-tier conference")
            print("   2. Or start with current data for testing, scale later")
            print("\n💡 Options:")
            print("   A. Test Phase 2 with 100 samples (fast, proof-of-concept)")
            print("   B. Scale to 1,500 samples now (better for final results)")


def main():
    print("🔬 PHASE 2 READINESS CHECK")
    print("Checking your Mac capabilities for BioMistral-7B implementation\n")

    # Run all checks
    is_apple_silicon = check_mac_specs()
    dataset_info = check_current_data()
    check_disk_space()
    all_deps_installed = check_dependencies()

    # Provide recommendations
    recommendations_for_mac(is_apple_silicon, dataset_info)

    # Final verdict
    print("\n" + "=" * 60)
    print("🎯 READY TO START PHASE 2?")
    print("=" * 60)

    if is_apple_silicon and all_deps_installed:
        print("\n✅ YES - You can start Phase 2!")
        print("   Your Mac should handle BioMistral-7B")
        print("\n📝 Next Steps:")
        print("   1. Decide: Test with 100 samples OR scale to 1,500?")
        print("   2. Run: python phase2_setup.py (coming next)")
        print("   3. Begin Week 3 Day 1 tasks")
    elif not is_apple_silicon:
        print("\n⚠️  CAUTION - Intel Mac may be too slow")
        print("   Consider: Cloud GPU (Colab, AWS, etc.)")
    else:
        print("\n⚠️  Install missing dependencies first")
        print("   Run: pip install -r requirements.txt")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
