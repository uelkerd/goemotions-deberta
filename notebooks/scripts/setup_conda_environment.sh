#!/bin/bash
# DeBERTa-v3-large Conda Environment Setup Script
# Run this script to create a compatible environment

set -e  # Exit on any error

echo "🚀 Setting up DeBERTa-v3-large Conda Environment"
echo "================================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Create new conda environment
echo "📦 Creating conda environment 'deberta-v3' with Python 3.9..."
conda create -n deberta-v3 python=3.9 -y

echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deberta-v3

echo "📥 Installing PyTorch with CUDA 11.8 support..."
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "📥 Installing core ML libraries..."
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install accelerate==0.24.0

echo "📥 Installing tokenizer dependencies..."
pip install sentencepiece==0.1.99
pip install tiktoken==0.5.1

echo "📥 Installing data processing libraries..."
pip install numpy==1.24.3
pip install pandas
pip install scikit-learn
pip install evaluate

echo "📥 Installing additional dependencies..."
pip install huggingface_hub
pip install safetensors

echo "🔧 Setting up environment variables..."
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0,1' >> ~/.bashrc

echo ""
echo "✅ Environment setup complete!"
echo "================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate deberta-v3"
echo ""
echo "To test the environment, run:"
echo "  python test_environment.py"
echo ""
echo "To start training, run:"
echo "  accelerate launch --num_processes=2 --mixed_precision=fp16 train_samo.py --output_dir './samo_out' --model_name 'microsoft/deberta-v3-large' --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 4 --num_train_epochs 3 --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.01 --fp16 true --tf32 true --gradient_checkpointing true"
echo ""
echo "Note: Make sure to set your HF_TOKEN environment variable:"
echo "  export HF_TOKEN='your_huggingface_token_here'"
