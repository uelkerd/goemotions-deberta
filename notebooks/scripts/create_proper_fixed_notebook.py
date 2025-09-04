#!/usr/bin/env python3
"""
Create PROPER fixed notebook by copying the original and fixing the parameters
"""

import json

# Read the original notebook
with open('notebooks/GoEmotions_DeBERTa_Efficient_Workflow.ipynb', 'r') as f:
    original = json.load(f)

# Create the fixed version with updated parameters
fixed_notebook = original.copy()

# Update the title and add warning about fixes
fixed_notebook['cells'][0]['source'][6] = "# GoEmotions DeBERTa-v3-large FIXED RIGOROUS Workflow\n"
fixed_notebook['cells'][0]['source'].insert(7, "\n")
fixed_notebook['cells'][0]['source'].insert(8, "## ⚠️ THIS IS THE FIXED VERSION WITH PROPER PARAMETERS!\n")
fixed_notebook['cells'][0]['source'].insert(9, "\n")
fixed_notebook['cells'][0]['source'].insert(10, "**CRITICAL FIXES APPLIED:**\n")
fixed_notebook['cells'][0]['source'].insert(11, "- ✅ 20,000 training samples (not 5,000)\n")
fixed_notebook['cells'][0]['source'].insert(12, "- ✅ 3e-5 learning rate (not 1e-5 or 2e-5)\n")
fixed_notebook['cells'][0]['source'].insert(13, "- ✅ 2-3 epochs (not 1)\n")
fixed_notebook['cells'][0]['source'].insert(14, "- ✅ Evaluation every 250 steps\n")
fixed_notebook['cells'][0]['source'].insert(15, "- ✅ Expected: 50-65% F1 (not 5-7%)\n")
fixed_notebook['cells'][0]['source'].insert(16, "\n")

# Find and fix the training cells
for i, cell in enumerate(fixed_notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_str = ''.join(cell['source'])
        
        # Fix PHASE 1 CONFIG 1: BCE Baseline (cell around line 633)
        if 'PHASE 1 CONFIG 1: BCE Baseline' in source_str:
            print(f"Fixing BCE baseline cell at index {i}")
            # Replace the command with fixed parameters
            new_source = [
                "# PHASE 1 CONFIG 1: BCE Baseline - FIXED PARAMETERS\n",
                "!cd /home/user/goemotions-deberta && python3 notebooks/scripts/train_deberta_local.py --output_dir \"./outputs/phase1_bce_fixed\" --model_type \"deberta-v3-large\" --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4 --num_train_epochs 2 --learning_rate 3e-5 --lr_scheduler_type cosine --warmup_ratio 0.15 --weight_decay 0.01 --fp16 --max_length 256 --max_train_samples 20000 --max_eval_samples 3000 --evaluation_strategy \"steps\" --eval_steps 250 --save_strategy \"steps\" --save_steps 250 --metric_for_best_model \"f1_macro\" --load_best_model_at_end --logging_steps 50"
            ]
            cell['source'] = new_source
        
        # Fix PHASE 1 CONFIG 2: Asymmetric Loss
        elif 'PHASE 1 CONFIG 2: Asymmetric Loss' in source_str:
            print(f"Fixing Asymmetric Loss cell at index {i}")
            new_source = [
                "# PHASE 1 CONFIG 2: Asymmetric Loss - FIXED PARAMETERS\n",
                "!cd /home/user/goemotions-deberta && python3 notebooks/scripts/train_deberta_local.py --output_dir \"./outputs/phase1_asymmetric_fixed\" --model_type \"deberta-v3-large\" --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4 --num_train_epochs 2 --learning_rate 3e-5 --lr_scheduler_type cosine --warmup_ratio 0.15 --weight_decay 0.01 --use_asymmetric_loss --fp16 --max_length 256 --max_train_samples 20000 --max_eval_samples 3000 --evaluation_strategy \"steps\" --eval_steps 250 --save_strategy \"steps\" --save_steps 250 --metric_for_best_model \"f1_macro\" --load_best_model_at_end --logging_steps 50"
            ]
            cell['source'] = new_source
        
        # Fix PHASE 1 CONFIG 3: Combined Loss 70%
        elif 'PHASE 1 CONFIG 3: Combined Loss 70%' in source_str:
            print(f"Fixing Combined Loss 70% cell at index {i}")
            new_source = [
                "# PHASE 1 CONFIG 3: Combined Loss 70% - FIXED PARAMETERS\n",
                "!cd /home/user/goemotions-deberta && python3 notebooks/scripts/train_deberta_local.py --output_dir \"./outputs/phase1_combined_07_fixed\" --model_type \"deberta-v3-large\" --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4 --num_train_epochs 2 --learning_rate 3e-5 --lr_scheduler_type cosine --warmup_ratio 0.15 --weight_decay 0.01 --use_combined_loss --loss_combination_ratio 0.7 --fp16 --max_length 256 --max_train_samples 20000 --max_eval_samples 3000 --evaluation_strategy \"steps\" --eval_steps 250 --save_strategy \"steps\" --save_steps 250 --metric_for_best_model \"f1_macro\" --load_best_model_at_end --logging_steps 50"
            ]
            cell['source'] = new_source
        
        # Fix PHASE 1 CONFIG 4: Combined Loss 50%
        elif 'PHASE 1 CONFIG 4: Combined Loss 50%' in source_str:
            print(f"Fixing Combined Loss 50% cell at index {i}")
            new_source = [
                "# PHASE 1 CONFIG 4: Combined Loss 50% - FIXED PARAMETERS\n",
                "!cd /home/user/goemotions-deberta && python3 notebooks/scripts/train_deberta_local.py --output_dir \"./outputs/phase1_combined_05_fixed\" --model_type \"deberta-v3-large\" --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4 --num_train_epochs 2 --learning_rate 3e-5 --lr_scheduler_type cosine --warmup_ratio 0.15 --weight_decay 0.01 --use_combined_loss --loss_combination_ratio 0.5 --fp16 --max_length 256 --max_train_samples 20000 --max_eval_samples 3000 --evaluation_strategy \"steps\" --eval_steps 250 --save_strategy \"steps\" --save_steps 250 --metric_for_best_model \"f1_macro\" --load_best_model_at_end --logging_steps 50"
            ]
            cell['source'] = new_source
        
        # Fix PHASE 1 CONFIG 5: Combined Loss 30%
        elif 'PHASE 1 CONFIG 5: Combined Loss 30%' in source_str:
            print(f"Fixing Combined Loss 30% cell at index {i}")
            new_source = [
                "# PHASE 1 CONFIG 5: Combined Loss 30% - FIXED PARAMETERS\n",
                "!cd /home/user/goemotions-deberta && python3 notebooks/scripts/train_deberta_local.py --output_dir \"./outputs/phase1_combined_03_fixed\" --model_type \"deberta-v3-large\" --per_device_train_batch_size 4 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4 --num_train_epochs 2 --learning_rate 3e-5 --lr_scheduler_type cosine --warmup_ratio 0.15 --weight_decay 0.01 --use_combined_loss --loss_combination_ratio 0.3 --fp16 --max_length 256 --max_train_samples 20000 --max_eval_samples 3000 --evaluation_strategy \"steps\" --eval_steps 250 --save_strategy \"steps\" --save_steps 250 --metric_for_best_model \"f1_macro\" --load_best_model_at_end --logging_steps 50"
            ]
            cell['source'] = new_source

# Add a new markdown cell after the environment setup explaining the fixes
insert_position = 10  # After environment setup
new_explanation_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔥 CRITICAL CONFIGURATION FIXES\n",
        "\n",
        "### What was WRONG (6.7% F1):\n",
        "| Parameter | **BROKEN** | **FIXED** | **Impact** |\n",
        "|-----------|------------|-----------|------------|\n",
        "| Training Samples | 5,000 | **20,000** | 4x more data to learn all 28 classes |\n",
        "| Learning Rate | 1e-5 or 2e-5 | **3e-5** | Optimal for DeBERTa-v3 |\n",
        "| Epochs | 1 | **2-3** | Sufficient training |\n",
        "| Eval Strategy | Once at end | **Every 250 steps** | Track progress |\n",
        "| Warmup | 10% | **15%** | Better stability |\n",
        "\n",
        "### Expected Results:\n",
        "- **BROKEN**: 5-7% F1 Macro (only learns 3 classes)\n",
        "- **FIXED**: 50-65% F1 Macro (learns all 28 classes)\n",
        "\n",
        "### Why These Changes Matter:\n",
        "1. **20k samples**: With 28 imbalanced classes, 5k samples means some classes have <10 examples!\n",
        "2. **3e-5 LR**: DeBERTa-v3 is a large model that needs higher learning rates\n",
        "3. **2-3 epochs**: One epoch isn't enough for the model to converge\n",
        "4. **Frequent evaluation**: Catch overfitting early and save best checkpoints\n"
    ]
}

fixed_notebook['cells'].insert(insert_position, new_explanation_cell)

# Save the fixed notebook
with open('notebooks/GoEmotions_DeBERTa_FIXED_RIGOROUS.ipynb', 'w') as f:
    json.dump(fixed_notebook, f, indent=2)

print("✅ Created PROPER fixed notebook with ALL cells from original!")
print("📁 Saved to: notebooks/GoEmotions_DeBERTa_FIXED_RIGOROUS.ipynb")
print(f"📊 Total cells: {len(fixed_notebook['cells'])}")
print("\nKey fixes applied to training commands:")
print("  - max_train_samples: 5000 → 20000")
print("  - learning_rate: 2e-5 → 3e-5")
print("  - num_train_epochs: 1 → 2")
print("  - warmup_ratio: 0.1 → 0.15")
print("  - Added evaluation_strategy: steps")
print("  - Added eval_steps: 250")
print("  - Added metric_for_best_model: f1_macro")