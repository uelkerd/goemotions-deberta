# DeBERTa-v3-large Compatible Environment Setup Guide

## Problem Summary
The current environment has fundamental incompatibilities with DeBERTa-v3-large's tiktoken-based tokenizer:
- Tiktoken URL resolution failures
- SentencePiece library detection issues
- NumPy compatibility problems
- Datasets library pattern matching errors

## Recommended Environment Setup

### Option 1: Google Colab (Recommended)
```python
# Install compatible versions
!pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.35.0
!pip install datasets==2.14.0
!pip install accelerate==0.24.0
!pip install sentencepiece==0.1.99
!pip install tiktoken==0.5.1
!pip install numpy==1.24.3
!pip install pandas
!pip install scikit-learn
!pip install evaluate
```

### Option 2: Local Conda Environment
```bash
# Create new conda environment
conda create -n deberta-v3 python=3.9
conda activate deberta-v3

# Install PyTorch with CUDA support
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers==4.35.0
pip install datasets==2.14.0
pip install accelerate==0.24.0
pip install sentencepiece==0.1.99
pip install tiktoken==0.5.1
pip install numpy==1.24.3
pip install pandas
pip install scikit-learn
pip install evaluate
```

### Option 3: Docker Container
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.9 python3-pip
RUN pip3 install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0
RUN pip3 install sentencepiece==0.1.99 tiktoken==0.5.1 numpy==1.24.3
RUN pip3 install pandas scikit-learn evaluate

# Copy your training script
COPY train_samo.py /workspace/
WORKDIR /workspace
```

## Key Version Compatibility Notes

### Critical Versions:
- **PyTorch**: 2.1.0 (avoid 2.3.1+ which has compatibility issues)
- **Transformers**: 4.35.0 (tested with DeBERTa-v3-large)
- **Datasets**: 2.14.0 (avoid 2.19.0+ which has pattern issues)
- **SentencePiece**: 0.1.99 (avoid 0.2.1+ which has detection issues)
- **Tiktoken**: 0.5.1 (avoid 0.11.0+ which has URL resolution issues)
- **NumPy**: 1.24.3 (avoid 1.23.5 which has `np.int` issues)

### Environment Variables:
```bash
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="your_huggingface_token_here"
```

## Testing the Environment

Run this test script to verify compatibility:

```python
import torch
import transformers
import datasets
import sentencepiece
import tiktoken
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"SentencePiece: {sentencepiece.__version__}")
print(f"Tiktoken: {tiktoken.__version__}")
print(f"NumPy: {np.__version__}")

# Test DeBERTa-v3-large loading
from transformers import AutoTokenizer, AutoConfig
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    print("✅ DeBERTa-v3-large tokenizer loaded successfully!")
    
    # Test tokenization
    test_text = "I love this movie! It's amazing."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"✅ Tokenization test passed: {tokens['input_ids'].shape}")
    
except Exception as e:
    print(f"❌ DeBERTa-v3-large test failed: {e}")

# Test GoEmotions dataset loading
try:
    from datasets import load_dataset
    dataset = load_dataset("go_emotions")
    print(f"✅ GoEmotions dataset loaded: {len(dataset['train'])} examples")
except Exception as e:
    print(f"❌ GoEmotions dataset test failed: {e}")
```

## Migration Steps

1. **Set up the new environment** using one of the options above
2. **Copy your training script** (`train_samo.py`) to the new environment
3. **Run the test script** to verify compatibility
4. **Execute training** with the same command:
   ```bash
   accelerate launch --num_processes=2 --mixed_precision=fp16 train_samo.py --output_dir "./samo_out" --model_name "microsoft/deberta-v3-large" --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 4 --num_train_epochs 3 --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 --weight_decay 0.01 --fp16 true --tf32 true --gradient_checkpointing true
   ```

## Expected Results

With the compatible environment, you should see:
- ✅ DeBERTa-v3-large tokenizer loads successfully
- ✅ GoEmotions dataset loads without pattern errors
- ✅ Training starts and progresses normally
- ✅ Evaluation report generated with F1 scores

## Troubleshooting

If you still encounter issues:
1. **Restart the runtime/kernel** after installing dependencies
2. **Clear cache**: `rm -rf ~/.cache/huggingface/`
3. **Verify CUDA**: `torch.cuda.is_available()` should return `True`
4. **Check GPU memory**: `nvidia-smi` should show available GPUs

The training script is already optimized and ready - it just needs a compatible environment to run in!
