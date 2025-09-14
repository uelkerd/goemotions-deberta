#!/bin/bash

# 🚀 COMPREHENSIVE MULTI-DATASET TRAINING
# ========================================
# 🎯 TARGET: >60% F1-macro with all datasets combined
# 📊 Datasets: GoEmotions + SemEval + ISEAR + MELD
# ⚡ Configuration: BCE Extended (your proven 51.79% winner)
# ==========================================

echo "🚀 COMPREHENSIVE MULTI-DATASET TRAINING"
echo "========================================"
echo "🎯 TARGET: >60% F1-macro with all datasets combined"
echo "📊 Datasets: GoEmotions + SemEval + ISEAR + MELD"
echo "⚡ Configuration: BCE Extended (your proven 51.79% winner)"
echo "=========================================="

# Set working directory
cd /home/user/goemotions-deberta 2>/dev/null || cd $(pwd)

# Create output and log directories
mkdir -p checkpoints_comprehensive_multidataset
mkdir -p logs

# Set log file
LOGFILE="logs/train_comprehensive_multidataset.log"
echo "📊 Logging to: $LOGFILE"

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

# Start training with comprehensive logging
{
    log_with_timestamp "🚀 Starting comprehensive multi-dataset training..."
    log_with_timestamp "📊 Configuration: BCE (proven winner from 51.79% baseline)"
    log_with_timestamp "🎯 Target performance: >60% F1-macro"

    # Check prerequisites
    log_with_timestamp "🔍 Checking prerequisites..."

    if [ ! -f "data/combined_all_datasets/train.jsonl" ]; then
        log_with_timestamp "❌ Combined dataset not found! Run data preparation first."
        exit 1
    fi

    if [ ! -f "notebooks/scripts/train_deberta_local.py" ]; then
        log_with_timestamp "❌ Training script not found!"
        exit 1
    fi

    # Count samples
    TRAIN_COUNT=$(wc -l < data/combined_all_datasets/train.jsonl)
    VAL_COUNT=$(wc -l < data/combined_all_datasets/val.jsonl)
    log_with_timestamp "✅ Dataset ready: $TRAIN_COUNT train, $VAL_COUNT val samples"

    # Set GPU
    export CUDA_VISIBLE_DEVICES=0
    log_with_timestamp "🎮 Using GPU: $CUDA_VISIBLE_DEVICES"

    # Training parameters (using proven BCE configuration)
    OUTPUT_DIR="checkpoints_comprehensive_multidataset"
    MODEL_TYPE="deberta-v3-large"
    BATCH_SIZE=4
    EVAL_BATCH_SIZE=8
    GRAD_ACCUM=4
    EPOCHS=3  # Extended training for better convergence
    LR="3e-5"  # Proven optimal learning rate

    log_with_timestamp "⚙️ Training parameters:"
    log_with_timestamp "   Model: $MODEL_TYPE"
    log_with_timestamp "   Epochs: $EPOCHS"
    log_with_timestamp "   Batch size: $BATCH_SIZE"
    log_with_timestamp "   Learning rate: $LR"
    log_with_timestamp "   Output: $OUTPUT_DIR"

    # Start training with Combined Loss (better for multi-dataset)
    log_with_timestamp "🚀 Starting training with Combined Loss (optimized for multi-dataset)..."

    # Build the command as a single line to avoid bash line continuation issues
    CMD="python3 notebooks/scripts/train_deberta_local.py \
        --output_dir \"$OUTPUT_DIR\" \
        --model_type \"$MODEL_TYPE\" \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $EVAL_BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_train_epochs $EPOCHS \
        --learning_rate $LR \
        --lr_scheduler_type \"cosine\" \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --fp16 \
        --max_length 256 \
        --use_combined_loss \
        --loss_combination_ratio 0.7 \
        --gamma 2.0 \
        --label_smoothing 0.1 \
        --augment_prob 0.0 \
        --freeze_layers 0 \
        --early_stopping_patience 3"

    # Execute the command with logging
    eval "$CMD" 2>&1 | tee -a "$LOGFILE"

    TRAINING_EXIT_CODE=${PIPESTATUS[0]}

    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        log_with_timestamp "✅ Training completed successfully!"

        # Check if results exist
        if [ -f "$OUTPUT_DIR/eval_report.json" ]; then
            log_with_timestamp "📊 Results found: $OUTPUT_DIR/eval_report.json"

            # Extract F1 scores using Python
            RESULTS=$(python3 -c "
import json, os
try:
    with open('$OUTPUT_DIR/eval_report.json', 'r') as f:
        data = json.load(f)
    f1_macro = data.get('f1_macro', 'N/A')
    f1_micro = data.get('f1_micro', 'N/A')
    print(f'F1_MACRO:{f1_macro}|F1_MICRO:{f1_micro}')
except:
    print('ERROR:Could not read results')
")

            if [[ $RESULTS == F1_MACRO:* ]]; then
                F1_MACRO=$(echo $RESULTS | cut -d'|' -f1 | cut -d':' -f2)
                F1_MICRO=$(echo $RESULTS | cut -d'|' -f2 | cut -d':' -f2)

                log_with_timestamp "📈 FINAL RESULTS:"
                log_with_timestamp "   F1 Macro: $F1_MACRO"
                log_with_timestamp "   F1 Micro: $F1_MICRO"

                # Success evaluation
                THRESHOLD_CHECK=$(python3 -c "
try:
    f1 = float('$F1_MACRO')
    if f1 >= 0.60:
        print('EXCELLENT')
    elif f1 >= 0.55:
        print('GOOD')
    elif f1 > 0.5179:
        print('IMPROVEMENT')
    else:
        print('BELOW_BASELINE')
except:
    print('ERROR')
")

                case $THRESHOLD_CHECK in
                    "EXCELLENT")
                        log_with_timestamp "🎉 EXCELLENT: Achieved >60% F1-macro target!"
                        log_with_timestamp "🚀 Multi-dataset training SUCCESSFUL!"
                        ;;
                    "GOOD")
                        log_with_timestamp "✅ GOOD: Achieved >55% F1-macro!"
                        log_with_timestamp "📈 Significant improvement with multi-dataset approach"
                        ;;
                    "IMPROVEMENT")
                        log_with_timestamp "👍 IMPROVEMENT: Better than 51.79% baseline"
                        log_with_timestamp "🔧 Consider extended training or hyperparameter tuning"
                        ;;
                    "BELOW_BASELINE")
                        log_with_timestamp "⚠️ BELOW BASELINE: F1-macro < 51.79%"
                        log_with_timestamp "🔍 Check data quality or training configuration"
                        ;;
                    *)
                        log_with_timestamp "⚠️ Could not evaluate performance"
                        ;;
                esac
            else
                log_with_timestamp "⚠️ Could not extract F1 scores from results"
            fi

            log_with_timestamp "🎯 TARGET COMPARISON:"
            log_with_timestamp "   Target: >60% F1-macro (multi-dataset goal)"
            log_with_timestamp "   Baseline: 51.79% F1-macro (GoEmotions BCE)"

        else
            log_with_timestamp "⚠️ No evaluation results found at $OUTPUT_DIR/eval_report.json"
        fi

        # Final backup and cleanup
        log_with_timestamp "💾 Training artifacts saved to: $OUTPUT_DIR"
        log_with_timestamp "📊 Full logs available at: $LOGFILE"

    else
        log_with_timestamp "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
        log_with_timestamp "🔍 Check logs above for error details"
        exit $TRAINING_EXIT_CODE
    fi

    log_with_timestamp "🏁 Comprehensive multi-dataset training complete!"

} 2>&1 | tee -a "$LOGFILE"

echo ""
echo "🎉 TRAINING COMPLETE!"
echo "📊 Check results: checkpoints_comprehensive_multidataset/eval_report.json"
echo "📝 Full logs: logs/train_comprehensive_multidataset.log"
echo ""