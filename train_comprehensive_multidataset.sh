#!/bin/bash

# 🚀 Comprehensive Multi-Dataset Training Script
# Trains DeBERTa-v3-large on combined GoEmotions, SemEval, ISEAR, and MELD datasets

set -e  # Exit on any error

echo "🎯 Multi-Dataset Emotion Classification Training"
echo "=============================================="
echo "📅 Date: $(date)"
echo "🖥️  Host: $(hostname)"
echo "👤 User: $(whoami)"
echo

# Configuration
EXPERIMENT_NAME="MultiDataset_BCE_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="data/combined"
OUTPUT_DIR="outputs/multidataset"
MODEL_NAME="microsoft/deberta-v3-large"

# Training parameters
NUM_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5
MAX_LENGTH=512
WARMUP_STEPS=500
LOGGING_STEPS=100
EVAL_STEPS=500
SAVE_STEPS=1000

echo "🔧 Configuration:"
echo "   📊 Experiment: $EXPERIMENT_NAME"
echo "   📁 Data: $DATA_DIR"
echo "   📤 Output: $OUTPUT_DIR"
echo "   🤖 Model: $MODEL_NAME"
echo "   📈 Epochs: $NUM_EPOCHS"
echo "   📦 Batch Size: $BATCH_SIZE"
echo "   🎯 Learning Rate: $LEARNING_RATE"
echo

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory $DATA_DIR not found!"
    echo "🔄 Running data preparation..."
    python3 prepare_all_datasets.py
    echo "✅ Data preparation completed"
fi

# Check if training data exists
if [ ! -f "$DATA_DIR/train.jsonl" ] || [ ! -f "$DATA_DIR/val.jsonl" ]; then
    echo "❌ Training data files not found!"
    echo "🔄 Running data preparation..."
    python3 prepare_all_datasets.py
    echo "✅ Data preparation completed"
fi

# Display data statistics
echo "📊 Dataset Statistics:"
TRAIN_SAMPLES=$(wc -l < "$DATA_DIR/train.jsonl")
VAL_SAMPLES=$(wc -l < "$DATA_DIR/val.jsonl")
TOTAL_SAMPLES=$((TRAIN_SAMPLES + VAL_SAMPLES))

echo "   📈 Training samples: $TRAIN_SAMPLES"
echo "   📈 Validation samples: $VAL_SAMPLES"
echo "   📈 Total samples: $TOTAL_SAMPLES"
echo

# Check GPU availability
echo "🖥️  GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "   🎮 Available GPUs: $GPU_COUNT"
else
    echo "   ⚠️  nvidia-smi not found - GPU status unknown"
fi
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=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
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# Start training
echo "🏃 Starting Multi-Dataset Training..."
echo "⏰ Start time: $(date)"
echo "📝 Logs will be saved to: logs/train_comprehensive_multidataset.log"
echo

# Run training with visible progress
python3 train_multidataset_deberta.py \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH" \
    --warmup_steps "$WARMUP_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    2>&1 | tee "logs/train_comprehensive_multidataset.log"

# Check training exit status
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo
echo "⏰ End time: $(date)"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Display final results
    echo
    echo "📊 Final Results:"
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        cat "$OUTPUT_DIR/eval_results.json"
    else
        echo "   📁 Check logs for detailed results"
    fi
    
    # Backup to Google Drive
    echo
    echo "📤 Backing up to Google Drive..."
    if [ -f "backup_to_gdrive.sh" ]; then
        bash backup_to_gdrive.sh
        echo "✅ Google Drive backup completed"
    else
        echo "⚠️  Google Drive backup script not found"
    fi
    
    echo
    echo "🎉 Multi-Dataset Training Complete!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Logs saved to: logs/"
    echo "☁️  Backup available in Google Drive"
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "📝 Check logs/train_comprehensive_multidataset.log for details"
    
    # Still backup logs for debugging
    echo "📤 Backing up logs for debugging..."
    if [ -f "backup_to_gdrive.sh" ]; then
        bash backup_to_gdrive.sh
    fi
    
    exit $TRAINING_EXIT_CODE
fi