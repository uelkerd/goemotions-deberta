# GoEmotions DeBERTa Multi-Label Classification

A machine learning project for multi-label emotion classification using the DeBERTa-v3-large model on the GoEmotions dataset.

## Project Structure

```
goemotions-deberta/
├── notebooks/          # Jupyter notebooks for experimentation and analysis
├── scripts/           # Python training scripts and shell scripts
├── models/            # Saved model checkpoints and outputs
├── data/              # Dataset files and processed data
├── docs/              # Documentation and setup guides
├── logs/              # Training logs and outputs
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Features

- **Multi-label emotion classification** using the GoEmotions dataset
- **DeBERTa-v3-large model** for state-of-the-art performance
- **Distributed training** with Accelerate framework
- **Mixed precision training** (FP16) for efficiency
- **Comprehensive logging** and monitoring

## Notebooks

- `GoEmotions_DeBERTa_MultiLabel_Classification.ipynb` - Main DeBERTa implementation
- `GoEmotions_RoBERTa_MultiLabel_Classification.ipynb` - RoBERTa baseline comparison
- `SAMO GoEmotions - DeBERTa-v3-large--P6000.ipynb` - SAMO optimization experiments
- `SAMO--deberta-v3-large.ipynb` - SAMO implementation
- `SAMO_2x3090_Vast.ipynb` - Multi-GPU training setup
- `SAMO_deberta_v3_optimized_plus60F1_v3.ipynb` - Optimized SAMO version

## Scripts

- `train_samo.py` - Main training script
- `train_samo_backup.py` - Backup training script
- `test_environment.py` - Environment testing script
- `setup_conda_environment.sh` - Environment setup script

## Documentation

- `conda_setup_manual.md` - Manual conda environment setup
- `environment_setup_guide.md` - Comprehensive setup guide

## Getting Started

1. **Environment Setup**: Follow the guides in `docs/` directory
2. **Data Preparation**: Place your dataset in the `data/` directory
3. **Training**: Use the scripts in `scripts/` directory
4. **Experimentation**: Work with notebooks in `notebooks/` directory

## Model Information

- **Base Model**: microsoft/deberta-v3-large
- **Task**: Multi-label emotion classification
- **Dataset**: GoEmotions
- **Framework**: PyTorch with Transformers
- **Training**: Accelerate with mixed precision

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Accelerate
- CUDA-compatible GPU (recommended)

## License

This project is for research and educational purposes.

## Author

Deniz Uelker (uelkerd)
