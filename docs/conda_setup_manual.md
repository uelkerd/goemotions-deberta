# Manual Conda Environment Setup for DeBERTa-v3-large

## Prerequisites
- Anaconda or Miniconda installed
- NVIDIA GPU with CUDA support
- At least 16GB RAM (32GB recommended)
- At least 50GB free disk space

## Step-by-Step Setup

### 1. Create the Conda Environment
```bash
# Create new environment with Python 3.9
conda create -n deberta-v3 python=3.9 -y

# Activate the environment
conda activate deberta-v3
```

### 2. Install PyTorch with CUDA Support
```bash
# Install PyTorch 2.1.0 with CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 3. Install Core ML Libraries
```bash
# Install transformers and related libraries
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install accelerate==0.24.0
```

### 4. Install Tokenizer Dependencies
```bash
# Install tokenizer libraries (critical for DeBERTa-v3-large)
pip install sentencepiece==0.1.99
pip install tiktoken==0.5.1
```

### 5. Install Data Processing Libraries
```bash
# Install data processing libraries
pip install numpy==1.24.3
pip install pandas
pip install scikit-learn
pip install evaluate
```

### 6. Install Additional Dependencies
```bash
# Install additional required libraries
pip install huggingface_hub
pip install safetensors
```

### 7. Set Environment Variables
```bash
# Add to your ~/.bashrc or ~/.zshrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0,1' >> ~/.bashrc  # Adjust based on your GPUs

# Reload your shell configuration
source ~/.bashrc
```

### 8. Set Hugging Face Token
```bash
# Set your Hugging Face token
export HF_TOKEN="your_huggingface_token_here"

# Or add to ~/.bashrc for persistence
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
```

## Verification Steps

### 1. Test the Environment
```bash
# Activate the environment
conda activate deberta-v3

# Run the compatibility test
python test_environment.py
```

### 2. Verify CUDA
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 3. Test DeBERTa-v3-large Loading
```bash
python -c "
from transformers import AutoTokenizer
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
print('✅ DeBERTa-v3-large tokenizer loaded successfully!')
"
```

## Running the Training

### 1. Activate Environment
```bash
conda activate deberta-v3
```

### 2. Set Hugging Face Token
```bash
export HF_TOKEN="your_huggingface_token_here"
```

### 3. Run Training
```bash
accelerate launch --num_processes=2 --mixed_precision=fp16 \
train_samo.py \
--output_dir "./samo_out" \
--model_name "microsoft/deberta-v3-large" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 4 \
--num_train_epochs 3 \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--fp16 true \
--tf32 true \
--gradient_checkpointing true
```

## Troubleshooting

### Common Issues:

1. **CUDA not available**
   ```bash
   # Check CUDA installation
   nvidia-smi
   # Reinstall PyTorch with correct CUDA version
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
   ```

2. **Tokenizer loading fails**
   ```bash
   # Clear Hugging Face cache
   rm -rf ~/.cache/huggingface/
   # Reinstall tokenizer dependencies
   pip install --force-reinstall sentencepiece==0.1.99 tiktoken==0.5.1
   ```

3. **Memory issues**
   ```bash
   # Reduce batch size in training command
   --per_device_train_batch_size 4
   --per_device_eval_batch_size 8
   ```

4. **Environment activation issues**
   ```bash
   # Initialize conda for your shell
   conda init bash  # or zsh
   # Restart terminal and try again
   ```

## Expected Results

With this setup, you should see:
- ✅ All compatibility tests pass
- ✅ DeBERTa-v3-large tokenizer loads without errors
- ✅ GoEmotions dataset loads successfully
- ✅ Training starts and progresses normally
- ✅ Evaluation report generated with F1 scores

## Performance Expectations

- **Training time**: ~2-4 hours for 3 epochs on 2×RTX 3090
- **Memory usage**: ~20-24GB GPU memory
- **Final F1 score**: Expected 0.85-0.90+ on GoEmotions dataset

The environment is now ready for DeBERTa-v3-large training! 🚀
