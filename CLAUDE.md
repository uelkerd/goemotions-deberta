# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project focused on **multi-label emotion classification** using the **GoEmotions dataset** with the **DeBERTa-v3-large** model. The project implements advanced techniques including asymmetric loss functions, combined loss strategies, and SMOTE augmentation for handling class imbalance.

## Project Architecture

### Core Structure
```
goemotions-deberta/
├── notebooks/                    # Jupyter notebooks for experimentation
│   ├── scripts/                  # Training and utility scripts
│   └── outputs/                  # Training outputs and checkpoints
├── data/                         # Dataset files (GoEmotions)
├── models/                       # Saved model checkpoints
├── logs/                         # Training logs and metrics
└── docs/                         # Environment setup guides
```

### Key Scripts and Their Purpose

**Main Training Scripts:**
- `notebooks/scripts/train_deberta_local.py` - Primary training script with all loss functions
- `notebooks/scripts/train_samo.py` - SAMO (Stochastic Adaptive Momentum Optimization) implementation
- `temp_parallel_training.py` - Multi-GPU parallel training orchestrator

**Testing and Validation Scripts:**
- `quick_integration_test.py` - Fast 50-step validation tests for loss functions
- `notebooks/scripts/test_environment.py` - Environment validation script
- `notebooks/scripts/simple_test.py` - Basic model functionality tests
- `final_scientific_validation.py` - Comprehensive validation pipeline

**Loss Function Research:**
- `debug_asymmetric_loss.py` - Asymmetric Loss debugging and analysis
- `asymmetric_loss_analysis.py` - Mathematical analysis of loss gradients
- `notebooks/scripts/test_asymmetric_loss.py` - Unit tests for AsymmetricLoss

## Common Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n deberta-v3 python=3.9 -y
conda activate deberta-v3

# Install dependencies
pip install -r requirements.txt

# Test environment
python notebooks/scripts/test_environment.py
```

### Training Commands
```bash
# Basic DeBERTa training
python notebooks/scripts/train_deberta_local.py \
  --output_dir ./outputs/basic_training \
  --model_type deberta-v3-large \
  --per_device_train_batch_size 4 \
  --num_train_epochs 2

# Training with Asymmetric Loss
python notebooks/scripts/train_deberta_local.py \
  --output_dir ./outputs/asymmetric_training \
  --use_asymmetric_loss \
  --model_type deberta-v3-large

# Training with Combined Loss
python notebooks/scripts/train_deberta_local.py \
  --output_dir ./outputs/combined_training \
  --use_combined_loss \
  --loss_combination_ratio 0.7

# Multi-GPU parallel training
python temp_parallel_training.py
```

### Testing and Validation
```bash
# Quick integration test (5 minutes)
python quick_integration_test.py

# Full scientific validation
python final_scientific_validation.py

# Asymmetric loss gradient analysis
python debug_asymmetric_loss.py
```

### DeepSpeed Integration
```bash
# Use DeepSpeed with config
deepspeed notebooks/scripts/train_deberta_local.py \
  --deepspeed deepspeed_config.json \
  --output_dir ./outputs/deepspeed_training
```

## Key Technical Components

### Loss Functions
- **AsymmetricLoss**: Custom implementation with gamma_neg=2.0 for hard negative mining
- **Combined Loss**: Weighted combination of BCE and Asymmetric Loss
- **Standard BCE**: Baseline multi-label classification loss

### Data Augmentation
- **SMOTE**: Synthetic oversampling for minority emotion classes
- **NLP Augmentation**: Text-based augmentation using nlpaug library

### Training Infrastructure
- **Mixed Precision (FP16)**: Memory-efficient training
- **Gradient Checkpointing**: Memory optimization for large models
- **DeepSpeed ZeRO Stage 2**: Distributed training optimization
- **Multi-GPU Support**: Parallel training across multiple GPUs

## Critical Architecture Notes

### Loss Function Integration
The training scripts support multiple loss functions through command-line arguments:
- `--use_asymmetric_loss`: Enables AsymmetricLoss with gamma_neg=2.0
- `--use_combined_loss` + `--loss_combination_ratio X`: Weighted BCE + AsymmetricLoss
- Default: Standard BCEWithLogitsLoss

### Model Architecture
- **Base Model**: microsoft/deberta-v3-large (900M parameters)
- **Task**: Multi-label classification (28 emotion classes)
- **Output**: Sigmoid activation for multi-label predictions
- **Tokenization**: DeBERTa-v3 tokenizer with max_length=256

### Data Pipeline
- **Dataset**: GoEmotions (58k Reddit comments, 28 emotion labels)
- **Preprocessing**: Automatic tokenization with padding/truncation
- **Class Imbalance**: Handled via SMOTE augmentation and asymmetric loss
- **Train/Val Split**: 80/20 with stratification

## Development Workflow

### For Training Experiments
1. Use `notebooks/scripts/test_environment.py` to verify setup
2. Run `quick_integration_test.py` for 5-minute validation
3. Execute full training with appropriate loss function flags
4. Monitor training with logs in `logs/` directory
5. Evaluate models using `notebooks/scripts/simple_eval.py`

### For Research and Analysis
1. Use Jupyter notebooks in `notebooks/` for experimentation
2. Key notebooks: `GoEmotions_DeBERTa_BULLETPROOF.ipynb` for latest implementation
3. Run gradient analysis with `debug_asymmetric_loss.py`
4. Use visualization scripts in root directory for loss curves

### Backup and Deployment
```bash
# Backup training artifacts to Google Drive
./backup_to_gdrive.sh

# Setup local model cache
python notebooks/scripts/setup_local_cache.py
```

## Critical Configuration Files

- `requirements.txt`: Python dependencies with specific versions
- `deepspeed_config.json`: DeepSpeed configuration (ZeRO stage 2, FP16)
- `notebooks/scripts/setup_conda_environment.sh`: Complete environment setup

## Important Implementation Details

### Gradient Health Monitoring
The codebase includes extensive gradient monitoring to prevent vanishing gradients, particularly important for AsymmetricLoss. Check `debug_asymmetric_loss.py:122-130` for gradient health validation.

### Memory Management
- Use gradient_accumulation_steps=4 for effective batch size control
- Enable gradient checkpointing for large models
- Monitor GPU memory usage during training

### Error Handling
Recent commits focus on fixing critical issues:
- Data collator recursion errors
- AsymmetricLoss gradient vanishing 
- Multi-label classification edge cases
- Combined loss AttributeError fixes