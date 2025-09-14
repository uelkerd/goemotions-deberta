# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project focused on **multi-label emotion classification** using **multiple datasets** (GoEmotions, SemEval, ISEAR, MELD) with the **DeBERTa-v3-large** model. The project implements advanced techniques including asymmetric loss functions, combined loss strategies, multi-dataset training, and comprehensive evaluation pipelines.

## Project Architecture

### Core Structure
```
goemotions-deberta/
├── notebooks/                    # Jupyter notebooks and training scripts
│   ├── scripts/                  # Core training and utility scripts (29 files)
│   ├── SAMo_MultiDataset_Streamlined_CLEAN.ipynb  # Main training notebook
│   └── prepare_all_datasets.py   # Multi-dataset preparation
├── data/                         # Multi-dataset storage
│   ├── goemotions/              # GoEmotions dataset cache
│   ├── semeval/                 # SemEval-2018 dataset
│   ├── meld/                    # MELD emotion dataset
│   └── combined_all_datasets/   # Unified training data
├── src/                         # Core implementation modules
│   ├── evaluation/              # Testing and validation scripts
│   ├── models/                  # Model implementations
│   └── utils/                   # Utility functions
├── configs/                     # Training configurations
├── checkpoints_*/              # Model outputs and checkpoints
├── logs/                       # Training logs and metrics
└── scripts/                    # Training orchestration scripts
```

### Key Scripts and Their Purpose

**Main Training Scripts:**
- `notebooks/scripts/train_deberta_local.py` - Primary training script with comprehensive loss functions and progress monitoring
- `notebooks/scripts/train_samo.py` - SAMO (Stochastic Adaptive Momentum Optimization) implementation
- `src/models/temp_parallel_training.py` - Multi-GPU parallel training orchestrator
- `notebooks/prepare_all_datasets.py` - Multi-dataset preparation (GoEmotions + SemEval + ISEAR + MELD)

**Key Notebooks:**
- `notebooks/SAMo_MultiDataset_Streamlined_CLEAN.ipynb` - Main training interface with proven BCE configuration
- Interactive workflow: data prep (Cell 2) → training (Cell 4) → analysis (Cell 6)

**Testing and Validation Scripts:**
- `src/evaluation/quick_integration_test.py` - Fast 50-step validation tests
- `notebooks/scripts/test_environment.py` - Environment validation script
- `src/evaluation/final_scientific_validation.py` - Comprehensive validation pipeline
- `notebooks/scripts/simple_test.py` - Basic model functionality tests

**Loss Function Research:**
- `src/evaluation/debug_asymmetric_loss.py` - Asymmetric Loss debugging and gradient analysis
- `src/evaluation/asymmetric_loss_analysis.py` - Mathematical analysis of gradient vanishing
- `notebooks/scripts/test_asymmetric_loss.py` - Unit tests for AsymmetricLoss implementation

## Common Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n samo-dl-stable python=3.9 -y
conda activate samo-dl-stable

# Install core dependencies (no requirements.txt in project)
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn
pip install accelerate deepspeed wandb sentencepiece tiktoken

# Test environment
python notebooks/scripts/test_environment.py
```

### Training Commands
```bash
# Multi-dataset preparation (required first)
python notebooks/prepare_all_datasets.py

# Basic DeBERTa training with BCE (proven configuration)
python notebooks/scripts/train_deberta_local.py \
  --output_dir ./checkpoints_basic_training \
  --model_type deberta-v3-large \
  --per_device_train_batch_size 4 \
  --num_train_epochs 5

# Training with Asymmetric Loss (experimental - gradient issues known)
python notebooks/scripts/train_deberta_local.py \
  --output_dir ./checkpoints_asymmetric_training \
  --use_asymmetric_loss \
  --model_type deberta-v3-large

# Training with Combined Loss
python notebooks/scripts/train_deberta_local.py \
  --output_dir ./checkpoints_combined_training \
  --use_combined_loss \
  --loss_combination_ratio 0.7

# Multi-dataset training (preferred approach)
bash scripts/train_comprehensive_multidataset.sh

# Multi-GPU parallel training
python src/models/temp_parallel_training.py
```

### Testing and Validation
```bash
# Quick integration test (5 minutes)
python src/evaluation/quick_integration_test.py

# Full scientific validation
python src/evaluation/final_scientific_validation.py

# Asymmetric loss gradient analysis
python src/evaluation/debug_asymmetric_loss.py

# Simple model functionality test
python notebooks/scripts/simple_test.py

# Environment and dependency validation
python notebooks/scripts/test_environment.py
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
- **Primary Dataset**: GoEmotions (58k Reddit comments, 28 emotion labels)
- **Additional Datasets**: SemEval-2018, ISEAR, MELD (combined ~38k+ total samples)
- **Preprocessing**: Automatic tokenization with padding/truncation (max_length=256)
- **Class Imbalance**: Handled via weighted sampling and loss function modifications
- **Train/Val Split**: 80/20 with stratification across all datasets
- **Multi-Dataset Integration**: Unified format via `notebooks/prepare_all_datasets.py`

## Development Workflow

### For Training Experiments
1. Use `notebooks/scripts/test_environment.py` to verify setup
2. Run `src/evaluation/quick_integration_test.py` for 5-minute validation
3. Prepare multi-dataset: `python notebooks/prepare_all_datasets.py`
4. Execute training via main notebook: `notebooks/SAMo_MultiDataset_Streamlined_CLEAN.ipynb`
5. Monitor training with logs: `tail -f logs/train_comprehensive_multidataset.log`
6. Check results: `checkpoints_*/eval_report.json`

### For Research and Analysis
1. Use main notebook: `notebooks/SAMo_MultiDataset_Streamlined_CLEAN.ipynb` (Cell 2 → Cell 4 → Cell 6)
2. Run gradient analysis: `python src/evaluation/debug_asymmetric_loss.py`
3. Comprehensive validation: `python src/evaluation/final_scientific_validation.py`
4. Loss comparison: `python notebooks/scripts/rigorous_loss_comparison.py`

### Backup and Deployment
```bash
# Backup training artifacts to Google Drive
./backup_to_gdrive.sh

# Setup local model cache
python notebooks/scripts/setup_local_cache.py
```

## Critical Configuration Files

- `configs/deepspeed_config.json`: DeepSpeed configuration (ZeRO stage 2, FP16, batch_size=4)
- `configs/per_emotion_thresholds.json`: Optimized thresholds for each emotion class
- `.cursorrules`: Terminal operation guidelines for reliable execution
- `notebooks/scripts/setup_conda_environment.sh`: Complete environment setup
- Environment: Use `samo-dl-stable` conda environment (not `deberta-v3`)

## Important Implementation Details

### Gradient Health Monitoring
The codebase includes extensive gradient monitoring to prevent vanishing gradients. AsymmetricLoss has known gradient vanishing issues (gradients ~1.5e-04 vs BCE ~0.3-1.2). Check `src/evaluation/debug_asymmetric_loss.py:42-50` for root cause analysis of gamma_neg=4.0 causing extreme gradient suppression.

### Memory Management
- Use gradient_accumulation_steps=4 for effective batch size control
- Enable gradient checkpointing for large models (900M parameters)
- Monitor GPU memory usage during training
- Automatic disk quota management with cleanup of old checkpoints
- Progress monitoring with automatic restart on training stalls (600s timeout)

### Multi-Dataset Training Architecture
- **Unified Format**: All datasets converted to consistent JSON structure
- **Weighted Sampling**: Balances contribution from each dataset
- **28 Emotion Classes**: Standardized GoEmotions emotion taxonomy
- **Performance Target**: >60% F1-macro (baseline: 51.79% on GoEmotions only)

### Known Issues and Fixes
Recent commits address:
- AsymmetricLoss gradient vanishing (torch.pow(1-pt, 4.0) creates ~1e-8 gradients)
- Multi-dataset loading errors in prepare_all_datasets.py
- NCCL timeout configuration for distributed training
- Disk quota exceeded errors with automatic cleanup
- Data collator recursion in multi-label classification