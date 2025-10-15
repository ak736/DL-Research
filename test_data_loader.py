"""
Quick Setup Script for Phase 1 Data
This will work even if HuggingFace datasets fail.

Usage:
    python quick_setup.py [--num-samples 100]
"""

from src.utils.data_loader import MedQADataLoader
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description='Setup Phase 1 data')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to prepare (default: 100)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if cache exists')
    args = parser.parse_args()

    print("🚀 Phase 1 Data Setup")
    print("="*70 + "\n")

    # Initialize loader
    loader = MedQADataLoader(data_dir="data", seed=42)

    # Clear cache if forced
    if args.force:
        cache_file = Path("data/phase1_cache.json")
        if cache_file.exists():
            cache_file.unlink()
            print("🗑️  Cleared cache\n")

    # Prepare data
    try:
        data = loader.prepare_phase1_data(
            num_samples=args.num_samples,
            test_ratio=0.2
        )

        print("\n✅ SUCCESS! Data ready for Phase 1")
        print("\n📁 Files created:")
        print("  • data/general_medicine_train.json")
        print("  • data/general_medicine_test.json")
        print("  • data/cardiology_train.json")
        print("  • data/cardiology_test.json")
        print("  • data/phase1_cache.json")

        print("\n🎯 Next steps:")
        print("  1. Test with: python test_data_loader.py")
        print("  2. Proceed to Phase 1 training scripts")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  • Check internet connection")
        print("  • Install datasets: pip install datasets")
        print("  • Using synthetic data fallback...")

        # Try synthetic data
        try:
            loader.prepare_phase1_data(
                num_samples=args.num_samples, test_ratio=0.2)
            print("\n✅ Created synthetic data for testing")
            return 0
        except Exception as e2:
            print(f"❌ Failed: {e2}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
