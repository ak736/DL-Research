#!/bin/bash
# Phase 2 Dependencies Installation Script
# Installs quantization and additional libraries needed for Meditron-7B

echo "🔧 Installing Phase 2 Dependencies..."
echo "=================================="
echo ""

# Install scikit-learn (missing from check)
echo "📦 Installing scikit-learn..."
pip install scikit-learn

# Install BitsAndBytes for quantization
echo "📦 Installing bitsandbytes for 4-bit quantization..."
pip install bitsandbytes

# Install accelerate for efficient loading
echo "📦 Installing accelerate..."
pip install accelerate

# Upgrade transformers to latest (for Llama 2 support)
echo "📦 Upgrading transformers..."
pip install --upgrade transformers

# Install scipy (for Fisher computation)
echo "📦 Installing scipy..."
pip install scipy

# Install matplotlib for visualizations
echo "📦 Installing matplotlib..."
pip install matplotlib

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "Next steps:"
echo "1. Test Meditron-7B loading"
echo "2. Scale dataset"
echo "3. Start Fisher computation"