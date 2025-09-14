#!/bin/bash
# Terminal setup aliases for GoEmotions DeBERTa project

# Training aliases
alias train-basic="python3 notebooks/scripts/train_deberta_local.py --output_dir ./outputs/basic_training --model_type deberta-v3-large --per_device_train_batch_size 4 --num_train_epochs 2"
alias train-asymmetric="python3 notebooks/scripts/train_deberta_local.py --output_dir ./outputs/asymmetric_training --use_asymmetric_loss --model_type deberta-v3-large"
alias train-combined="python3 notebooks/scripts/train_deberta_local.py --output_dir ./outputs/combined_training --use_combined_loss --loss_combination_ratio 0.7"
alias train-parallel="python3 temp_parallel_training.py"

# Testing aliases
alias test-quick="python3 quick_integration_test.py"
alias test-env="python3 notebooks/scripts/test_environment.py"
alias test-full="python3 final_scientific_validation.py"
alias test-loss="python3 debug_asymmetric_loss.py"

# Utility aliases
alias gemo-logs="tail -f logs/*.log"
alias gemo-status="ls -la outputs/ && echo '---' && ls -la models/"
alias gemo-clean="rm -rf outputs/* logs/*.log"

echo "GoEmotions DeBERTa aliases loaded!"
echo ""
echo "Training commands:"
echo "  train-basic      - Basic DeBERTa training"
echo "  train-asymmetric - Training with Asymmetric Loss"
echo "  train-combined   - Training with Combined Loss"
echo "  train-parallel   - Multi-GPU parallel training"
echo ""
echo "Testing commands:"
echo "  test-quick       - Quick 5-minute integration test"
echo "  test-env         - Environment validation"
echo "  test-full        - Full scientific validation"
echo "  test-loss        - Asymmetric loss gradient analysis"
echo ""
echo "Utility commands:"
echo "  gemo-logs        - Watch training logs"
echo "  gemo-status      - Check outputs and models"
echo "  gemo-clean       - Clean outputs and logs"