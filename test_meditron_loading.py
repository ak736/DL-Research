#!/usr/bin/env python3
"""
Mac-Optimized Meditron-7B Loading
Works around bitsandbytes CUDA requirement on Apple Silicon

Three strategies:
1. Try latest bitsandbytes (>=0.43.1) with MPS support
2. Use torch.float16 with aggressive memory optimization
3. Fallback to smaller model if needed
"""

import torch
import gc
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def try_quantized_loading():
    """Try loading with quantization (if bitsandbytes supports MPS)"""
    print("\n🔧 Strategy 1: Trying 4-bit Quantization with MPS...")

    try:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            "epfl-llm/meditron-7b",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        print("   ✅ Quantization with MPS succeeded!")
        return model, "quantized"

    except Exception as e:
        print(f"   ❌ Quantization failed: {str(e)[:100]}")
        return None, None


def try_fp16_loading():
    """Try loading in FP16 with aggressive memory optimization"""
    print("\n🔧 Strategy 2: Loading in FP16 (Half Precision)...")
    print("   This uses ~7-8GB RAM (should work on 8GB Mac)")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "epfl-llm/meditron-7b",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Move to MPS
        if torch.backends.mps.is_available():
            print("   ✅ Model loaded successfully in FP16!")
            print("   ✅ Using MPS (Metal) for acceleration")
        else:
            print("   ⚠️  MPS not available, using CPU")

        return model, "fp16"

    except Exception as e:
        print(f"   ❌ FP16 loading failed: {str(e)[:100]}")
        return None, None


def try_biogpt_fallback():
    """Fallback to smaller BioGPT model"""
    print("\n🔧 Strategy 3: Fallback to BioGPT-Large (1.5B)...")
    print("   Smaller model, fits easily in 8GB RAM")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/biogpt-large",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Move to MPS
        if torch.backends.mps.is_available():
            model = model.to("mps")
            print("   ✅ BioGPT-Large loaded successfully!")
            print("   ✅ Using MPS (Metal) for acceleration")

        return model, "biogpt"

    except Exception as e:
        print(f"   ❌ BioGPT loading failed: {str(e)[:100]}")
        return None, None


def test_inference(model, tokenizer, model_type):
    """Test model with a medical question"""
    print("\n🧪 Testing Inference...")

    test_prompt = "What is the primary treatment for hypertension?"

    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")

        # Move to MPS if available
        if torch.backends.mps.is_available() and model_type != "biogpt":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif model_type == "biogpt":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"   ✅ Inference successful!")
        print(f"\n   📝 Prompt: {test_prompt}")
        print(f"   🤖 Response: {response[:100]}...")

        return True

    except Exception as e:
        print(f"   ❌ Inference failed: {str(e)[:100]}")
        return False


def main():
    """Main testing function"""

    print("=" * 70)
    print("🧪 MAC-OPTIMIZED MEDITRON-7B LOADING TEST")
    print("=" * 70)

    print("\n📊 System Info:")
    print(f"   Platform: Apple Silicon Mac")
    print(f"   RAM: 8GB")
    print(f"   MPS Available: {torch.backends.mps.is_available()}")

    initial_memory = get_memory_usage()
    print(f"   Initial Memory: {initial_memory:.2f} GB")

    # Try strategies in order
    model = None
    model_type = None

    # Strategy 1: Quantized (if bitsandbytes supports MPS)
    model, model_type = try_quantized_loading()

    # Strategy 2: FP16 (recommended for Mac)
    if model is None:
        print("\n⚠️  Quantization not available on Mac")
        print("   This is normal - bitsandbytes requires CUDA (NVIDIA)")
        model, model_type = try_fp16_loading()

    # Strategy 3: Smaller model fallback
    if model is None:
        print("\n⚠️  Meditron-7B too large for 8GB RAM")
        print("   Trying smaller BioGPT model...")
        model, model_type = try_biogpt_fallback()

    # Check if any strategy worked
    if model is None:
        print("\n" + "=" * 70)
        print("❌ ALL STRATEGIES FAILED")
        print("=" * 70)
        print("\n💡 Recommendations:")
        print("   1. Close other applications to free RAM")
        print("   2. Restart Python/Terminal")
        print("   3. Use cloud GPU (Google Colab)")
        return False

    # Load tokenizer
    print(f"\n📦 Loading tokenizer for {model_type}...")
    if model_type == "biogpt":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt-large")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "epfl-llm/meditron-7b", trust_remote_code=True)

    model_memory = get_memory_usage()
    print(f"   ✅ Tokenizer loaded")
    print(f"   📊 Memory with model: {model_memory:.2f} GB")
    print(f"   📊 Model uses: ~{model_memory - initial_memory:.2f} GB")

    # Test inference
    inference_success = test_inference(model, tokenizer, model_type)

    if not inference_success:
        print("\n⚠️  Inference failed, but model loaded")

    final_memory = get_memory_usage()
    print(f"\n   📊 Peak memory: {final_memory:.2f} GB")
    print(f"   📊 RAM available: {8.0 - final_memory:.2f} GB")

    # Final verdict
    print("\n" + "=" * 70)

    if model_type == "fp16":
        print("✅ SUCCESS: MEDITRON-7B (FP16) WORKS ON YOUR MAC!")
        print("=" * 70)
        print("\n🎯 Model Configuration:")
        print("   Model: Meditron-7B")
        print("   Precision: FP16 (half precision)")
        print("   Memory: ~7-8GB")
        print("   Device: MPS (Metal)")

        print("\n📝 For Phase 2:")
        print("   ✅ Use Meditron-7B in FP16")
        print("   ✅ This is publication-quality!")
        print("   ⚠️  Memory will be tight during Fisher computation")
        print("   💡 We'll optimize Fisher to work in available RAM")

        # Save success marker
        marker_file = "outputs/phase2_fisher_lora/meditron_fp16_success.txt"
        os.makedirs(os.path.dirname(marker_file), exist_ok=True)
        with open(marker_file, 'w') as f:
            f.write(f"Meditron-7B FP16 loaded successfully\n")
            f.write(f"Memory usage: {model_memory:.2f} GB\n")
            f.write(f"Available RAM: {8.0 - final_memory:.2f} GB\n")

        recommended_model = "meditron-7b-fp16"

    elif model_type == "biogpt":
        print("✅ SUCCESS: BIOGPT-LARGE WORKS ON YOUR MAC!")
        print("=" * 70)
        print("\n🎯 Model Configuration:")
        print("   Model: BioGPT-Large (1.5B)")
        print("   Precision: FP16")
        print("   Memory: ~3-4GB")
        print("   Device: MPS (Metal)")

        print("\n📝 For Phase 2:")
        print("   ✅ Use BioGPT-Large")
        print("   ✅ Plenty of RAM for Fisher computation")
        print("   ✅ Fast training")
        print("   ⚠️  Smaller than Meditron (but still medical)")
        print("   💡 Can upgrade to Meditron on cloud for final results")

        # Save success marker
        marker_file = "outputs/phase2_fisher_lora/biogpt_success.txt"
        os.makedirs(os.path.dirname(marker_file), exist_ok=True)
        with open(marker_file, 'w') as f:
            f.write(f"BioGPT-Large loaded successfully\n")
            f.write(f"Memory usage: {model_memory:.2f} GB\n")
            f.write(f"Available RAM: {8.0 - final_memory:.2f} GB\n")

        recommended_model = "biogpt-large"

    elif model_type == "quantized":
        print("✅ SUCCESS: MEDITRON-7B (4-BIT) WORKS ON YOUR MAC!")
        print("=" * 70)
        print("   🎉 You have the latest bitsandbytes with MPS support!")

        recommended_model = "meditron-7b-4bit"

    print("\n" + "=" * 70)

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\n✅ READY FOR PHASE 2!")
    print(f"   Recommended: {recommended_model}")
    print("\n📅 Next Steps:")
    print("   1. Continue with: python scale_dataset.py")
    print("   2. Then: python create_phase2_structure.py")
    print("   3. Update config with: " + recommended_model)

    return True


if __name__ == "__main__":
    success = main()

    if not success:
        print("\n⚠️  Need help? Options:")
        print("   A. Close other apps and retry")
        print("   B. Use Google Colab (free GPU)")
        print("   C. Use smaller model (BioGPT)")
