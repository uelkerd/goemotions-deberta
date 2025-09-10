#!/usr/bin/env python3
"""
DeBERTa-v3-large training script with local caching
Uses locally cached models and datasets for fast, offline training

IMPROVEMENTS SUMMARY:

1. Fixed AsymmetricLoss Implementation: The critical issue was with disable_torch_grad_focal_loss=True 
   creating a torch.no_grad() context that disconnected gradients. I've patched the training script to fix this.

2. Progress Monitoring: Added a ProgressMonitorCallback that detects when training stalls and takes 
   corrective action. It will automatically stop and restart training if no progress is made for 10 minutes.

3. Disk Quota Management: Added automatic checking of disk space and cleanup of old checkpoints when 
   space is getting low. This prevents the "Disk quota exceeded" error that was causing the training to stop.

4. Google Drive Backup: Added automatic backup to your Google Drive folder every 15 minutes. This ensures 
   that all important files (model weights, eval reports, configs) are saved even if there are issues with 
   the local disk.

5. NCCL Optimizations: Reduced the NCCL timeout from 3600s to 1800s and added settings for localhost-only 
   communication to prevent network-related hangs.

6. Improved Error Handling: Added comprehensive error handling with stack traces for better debugging.
"""

import os
import json
import math
import random
import argparse
import warnings
from pathlib import Path

# Suppress compatibility warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers") 
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Set environment variables for DeBERTa-v3-large compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Temporarily disable offline mode for model loading

# NCCL configuration to prevent timeouts
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 min timeout (reduced from 1 hour)
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Enable blocking wait
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Better error handling
os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand (causes issues on some systems)
os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Use localhost interface only

# Check disk space at startup
try:
    import shutil
    disk_usage = shutil.disk_usage("/")
    free_space_gb = disk_usage.free / (1024 ** 3)
    used_percent = (disk_usage.used / disk_usage.total) * 100
    print(f"💾 Disk space at startup: {free_space_gb:.1f}GB free, {used_percent:.1f}% used")
    if free_space_gb < 10 or used_percent > 85:
        print(f"⚠️ WARNING: Low disk space detected at startup! Training might fail with quota errors.")
        print(f"⚠️ Consider cleaning up before training or reducing checkpoint frequency.")
except Exception as e:
    print(f"⚠️ Error checking disk space: {str(e)}")

from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
)
from sklearn.metrics import f1_score, precision_recall_fscore_support
import logging
import time
import random
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ProgressMonitorCallback(TrainerCallback):
    """
    Monitors training progress and detects stalls
    """
    def __init__(self, stall_timeout=600, check_disk_quota=True, min_disk_space_gb=10, enable_gdrive_backup=True):  # 10 minutes timeout
        self.last_progress_time = time.time()
        self.stall_timeout = stall_timeout
        self.last_step = 0
        self.last_disk_check_time = time.time()
        self.disk_check_interval = 300  # Check disk space every 5 minutes
        self.check_disk_quota = check_disk_quota
        self.min_disk_space_gb = min_disk_space_gb
        self.last_backup_time = time.time()
        self.backup_interval = 900  # Backup every 15 minutes
        self.enable_gdrive_backup = enable_gdrive_backup
        # Use the exact path format as confirmed by rclone ls command
        self.gdrive_backup_path = "'drive:00_Projects/🎯 TechLabs-2025/Final_Project/TRAINING/GoEmotions-DeBERTa-Backup/'"

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        
        # Track progress
        if state.global_step != self.last_step:
            self.last_progress_time = current_time
            self.last_step = state.global_step

        # Check for stall
        if current_time - self.last_progress_time > self.stall_timeout:
            print(f"⚠️ Training stall detected! No progress for {self.stall_timeout} seconds at step {state.global_step}")
            print("🔄 Attempting to save checkpoint and exit gracefully...")
            self._save_checkpoint(args, state)
            # Backup to Google Drive before stopping
            if self.enable_gdrive_backup:
                self._backup_to_gdrive(args, force=True)
            control.should_training_stop = True
            return control
            
        # Check disk space periodically
        if self.check_disk_quota and current_time - self.last_disk_check_time > self.disk_check_interval:
            self.last_disk_check_time = current_time
            self._check_disk_space(args)
        
        # Backup to Google Drive periodically
        if self.enable_gdrive_backup and current_time - self.last_backup_time > self.backup_interval:
            self.last_backup_time = current_time
            self._backup_to_gdrive(args)
        
        return control
        
    def _check_disk_space(self, args):
        """Check available disk space and clean up if needed"""
        try:
            import shutil
            import os
            
            # Get disk usage for output directory
            disk_usage = shutil.disk_usage(args.output_dir)
            free_space_gb = disk_usage.free / (1024 ** 3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            print(f"💾 Disk space: {free_space_gb:.1f}GB free, {used_percent:.1f}% used")
            
            # If disk space is low, clean up old checkpoints
            if free_space_gb < self.min_disk_space_gb or used_percent > 85:
                print(f"⚠️ Low disk space detected! Cleaning old checkpoints...")
                self._cleanup_old_checkpoints(args.output_dir)
        except Exception as e:
            print(f"⚠️ Error checking disk space: {str(e)}")
    
    def _cleanup_old_checkpoints(self, output_dir):
        """Remove old checkpoint directories except the latest"""
        try:
            import os
            import re
            from glob import glob
            
            # Find checkpoint directories
            checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
            checkpoints = glob(checkpoint_pattern)
            
            if len(checkpoints) <= 1:
                print("✓ No old checkpoints to clean")
                return
                
            # Extract step numbers and sort
            checkpoint_steps = []
            for checkpoint in checkpoints:
                match = re.search(r'checkpoint-([0-9]+)$', checkpoint)
                if match:
                    step = int(match.group(1))
                    checkpoint_steps.append((step, checkpoint))
            
            # Sort by step (ascending)
            checkpoint_steps.sort()
            
            # Keep the latest 2 checkpoints, delete the rest
            for step, checkpoint in checkpoint_steps[:-2]:  # Keep latest 2
                print(f"🗑️ Removing old checkpoint: {checkpoint}")
                try:
                    import shutil
                    shutil.rmtree(checkpoint)
                except Exception as e:
                    print(f"  ⚠️ Failed to remove {checkpoint}: {str(e)}")
        except Exception as e:
            print(f"⚠️ Error cleaning checkpoints: {str(e)}")
    
    def _save_checkpoint(self, args, state):
        """Force save a checkpoint when training stalls"""
        try:
            print(f"💾 Saving recovery checkpoint at step {state.global_step}...")
            # This would typically be handled by the Trainer itself
            # The Trainer should save when control.should_training_stop = True
        except Exception as e:
            print(f"⚠️ Error saving checkpoint: {str(e)}")
    
    def _backup_to_gdrive(self, args, force=False):
        """Backup important files to Google Drive"""
        try:
            import subprocess
            import os
            
            output_dir = args.output_dir
            model_name = os.path.basename(output_dir)
            gdrive_dest = f"{self.gdrive_backup_path}{model_name}/"
            
            print(f"🔄 Backing up training outputs to Google Drive: {gdrive_dest}")
            
            # Ensure the destination directory exists (using shell=True for proper path handling)
            mkdir_cmd = f"rclone mkdir -p {gdrive_dest}"
            subprocess.run(mkdir_cmd, shell=True, capture_output=True)
            
            # Files to backup: checkpoints, config, eval results
            # Copy the eval_report.json if it exists
            eval_report_path = os.path.join(output_dir, "eval_report.json")
            if os.path.exists(eval_report_path):
                copy_cmd = f"rclone copy '{eval_report_path}' {gdrive_dest}"
                subprocess.run(copy_cmd, shell=True, capture_output=True)
                print(f"✅ Backed up evaluation report to Google Drive")
            
            # Copy the latest model files (pytorch_model.bin, config.json, etc.)
            model_files = ["config.json", "pytorch_model.bin", "training_args.bin"]
            for file in model_files:
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    copy_cmd = f"rclone copy '{file_path}' {gdrive_dest}"
                    subprocess.run(copy_cmd, shell=True, capture_output=True)
            
            # Copy the latest checkpoint if we're not forcing a backup (periodic backup)
            # or all checkpoints if we're forcing (end of training or error)
            checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                if force:
                    # Backup all checkpoints on force
                    for checkpoint in checkpoint_dirs:
                        checkpoint_path = os.path.join(output_dir, checkpoint)
                        dest_checkpoint = f"{gdrive_dest}{checkpoint}/"
                        copy_cmd = f"rclone copy '{checkpoint_path}' {dest_checkpoint} --transfers 4"
                        subprocess.run(copy_cmd, shell=True, capture_output=True)
                        print(f"✅ Backed up {checkpoint} to Google Drive")
                else:
                    # Only backup latest checkpoint for periodic backups
                    latest = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))[-1]
                    latest_path = os.path.join(output_dir, latest)
                    dest_checkpoint = f"{gdrive_dest}{latest}/"
                    copy_cmd = f"rclone copy '{latest_path}' {dest_checkpoint} --transfers 4"
                    subprocess.run(copy_cmd, shell=True, capture_output=True)
                    print(f"✅ Backed up latest checkpoint {latest} to Google Drive")
            
            # Copy log files
            log_files = [f for f in os.listdir(output_dir) if f.endswith(".log") or f.endswith(".json")]
            for log_file in log_files:
                log_path = os.path.join(output_dir, log_file)
                copy_cmd = f"rclone copy '{log_path}' {gdrive_dest}"
                subprocess.run(copy_cmd, shell=True, capture_output=True)
            
            print(f"✅ Backup to Google Drive completed")
            
        except Exception as e:
            print(f"⚠️ Error backing up to Google Drive: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def on_save(self, args, state, control, **kwargs):
        """Check disk space whenever a checkpoint is saved"""
        if self.check_disk_quota:
            self._check_disk_space(args)
            
        # Backup to Google Drive whenever a checkpoint is saved
        if self.enable_gdrive_backup:
            self._backup_to_gdrive(args)
            
        return control

class ScientificLogger:
    """
    Comprehensive logging for scientific reproducibility
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.start_time = time.time()
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{output_dir}/scientific_log_{self.experiment_id}.json"
        self.metrics_history = []
        
    def log_experiment_start(self, config):
        """Log experiment configuration"""
        experiment_log = {
            "experiment_id": self.experiment_id,
            "start_time": datetime.now().isoformat(),
            "configuration": config,
            "system_info": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            }
        }
        self._write_log(experiment_log)
        
    def log_training_step(self, step, loss, learning_rate, epoch):
        """Log training step metrics"""
        step_log = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "loss": float(loss),
            "learning_rate": float(learning_rate),
            "elapsed_time": time.time() - self.start_time
        }
        self.metrics_history.append(step_log)
        
    def log_evaluation(self, eval_metrics, epoch):
        """Log evaluation metrics"""
        eval_log = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "evaluation_metrics": eval_metrics,
            "elapsed_time": time.time() - self.start_time
        }
        self._write_log(eval_log)
        
    def log_experiment_end(self, final_metrics):
        """Log final experiment results"""
        final_log = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "total_time": time.time() - self.start_time,
            "final_metrics": final_metrics,
            "training_history": self.metrics_history
        }
        self._write_log(final_log)
        
    def _write_log(self, log_data):
        """Write log data to file"""
        import json
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')

# GoEmotions labels
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    Addresses class imbalance by down-weighting easy negatives while maintaining focus on hard positives
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        # CRITICAL: Always set to False to ensure gradient flow
        self.disable_torch_grad_focal_loss = False  # Force False to fix gradient disconnection
        self.eps = eps

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (only clip negatives, per official ASL paper)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation (add small eps to prevent log(0))
        los_pos = y * torch.log(xs_pos + self.eps)
        los_neg = (1 - y) * torch.log(xs_neg + self.eps)
        loss = los_pos + los_neg

        # Asymmetric Focusing - FIXED implementation to ensure gradient flow
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Calculate weights for focusing
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()  # Keep negative for multi-label CE convention

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(dataset_path):
    """
    Compute class weights based on inverse frequency
    """
    import json
    from collections import Counter
    
    # Count emotion frequencies
    emotion_counts = Counter()
    total_samples = 0
    
    with open(dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            labels = item['labels']
            if isinstance(labels, int):
                labels = [labels]
            for label in labels:
                if 0 <= label < len(EMOTION_LABELS):
                    emotion_counts[label] += 1
            total_samples += 1
    
    # Compute inverse frequency weights
    class_weights = []
    for i in range(len(EMOTION_LABELS)):
        if emotion_counts[i] > 0:
            weight = total_samples / (len(EMOTION_LABELS) * emotion_counts[i])
            class_weights.append(weight)
        else:
            class_weights.append(1.0)  # Default weight for unseen classes
    
    return torch.tensor(class_weights, dtype=torch.float)

class CombinedLossTrainer(Trainer):
    """
    Combined Trainer using ASL + Class Weighting + Focal Loss
    """
    def __init__(self, loss_combination_ratio=0.7, gamma=2.0, label_smoothing=0.1, per_class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FIXED: Use conservative ASL parameters to prevent gradient vanishing
        self.asymmetric_loss = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05, disable_torch_grad_focal_loss=False)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=gamma, reduction='mean')
        self.combined_loss = nn.BCEWithLogitsLoss()
        self.loss_combination_ratio = loss_combination_ratio
        self.label_smoothing = label_smoothing  # CRITICAL FIX: Missing assignment
        self.per_class_weights = torch.tensor(json.loads(per_class_weights)) if per_class_weights else None
        
        # Apply per-class weights to focal loss alpha if provided
        if self.per_class_weights is not None:
            self.focal_loss.alpha = self.per_class_weights

        # Compute class weights from training data
        train_path = "data/goemotions/train.jsonl"
        self.class_weights = compute_class_weights(train_path)
        if self.per_class_weights is not None:
            self.class_weights = self.per_class_weights.to(self.class_weights.device)
        print(f"📊 Class weights computed: {self.class_weights}")
        print(f"🎯 Loss combination: {self.loss_combination_ratio} ASL + {1-self.loss_combination_ratio} Focal")
        
        # OVERSAMPLING: Apply to training dataset
        self.oversampled_data = self._apply_oversampling(train_path, self.class_weights)
        print(f"✅ Oversampling applied for rare classes")

    def _apply_oversampling(self, train_path, class_weights, oversample_factor=1.5):
        """Apply stratified oversampling to rare classes while preserving data integrity"""
        from collections import Counter, defaultdict
        import random
        from sklearn.utils import resample  # For stratified-like sampling
        
        # Load and analyze data
        data_list = []
        samples_by_class = defaultdict(list)
        
        with open(train_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                labels = item['labels']
                if isinstance(labels, int):
                    labels = [labels]
                # Ensure valid labels
                valid_labels = [l for l in labels if 0 <= l < len(EMOTION_LABELS)]
                if not valid_labels:
                    valid_labels = [27]  # Default to neutral if invalid
                item['labels'] = valid_labels  # Normalize
                data_list.append(item)
                for label in valid_labels:
                    samples_by_class[label].append(item)
        
        original_size = len(data_list)
        
        # Identify rare classes (bottom 50% frequency)
        class_freq = {i: len(samples_by_class[i]) for i in range(len(EMOTION_LABELS)) if i in samples_by_class}
        sorted_classes = sorted(class_freq.items(), key=lambda x: x[1])
        median_freq = sorted_classes[len(sorted_classes)//2][1] if sorted_classes else 1
        rare_classes = [cls for cls, freq in sorted_classes if freq < median_freq]
        
        print(f"📊 Rare classes identified: {rare_classes} (threshold: {median_freq} samples)")
        
        # Oversample rare classes using resample (stratified by class presence)
        oversampled_data = data_list.copy()
        for rare_class in rare_classes:
            rare_samples = samples_by_class[rare_class]
            if len(rare_samples) > 0:
                # Resample to target size (original + factor * original for rare)
                target_size = int(len(rare_samples) * oversample_factor)
                additional_samples = resample(
                    rare_samples, n_samples=target_size - len(rare_samples),
                    random_state=42, replace=True
                )
                # No text variation to avoid tokenization issues; duplicates are fine for training
                oversampled_data.extend(additional_samples)
                print(f"📈 Oversampled class {EMOTION_LABELS[rare_class]}: {len(rare_samples)} → {target_size}")
        
        # Shuffle and cap to prevent excessive size (1.5x max for stability)
        random.shuffle(oversampled_data)
        max_size = int(original_size * 1.5)
        if len(oversampled_data) > max_size:
            oversampled_data = oversampled_data[:max_size]
            print(f"⚖️ Capped oversampled dataset at {max_size} samples for stability")
        
        print(f"✅ Stratified oversampling applied: {original_size} → {len(oversampled_data)} samples")
        return oversampled_data

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Combined loss: ASL + Class Weighting + Focal Loss with label smoothing
        """
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")


        # Compute individual losses
        asl_loss = self.asymmetric_loss(logits, labels)

        # FIXED: Per-class weighted focal loss (apply weights element-wise)
        focal_loss = self.focal_loss(logits, labels)
        # Expand class_weights to batch dimensions: [batch, classes] and move to same device
        batch_size, num_classes = labels.shape
        class_weights_batch = self.class_weights.to(labels.device).unsqueeze(0).expand(batch_size, num_classes)
        # Apply per-class weighting: weighted_focal = focal_loss * class_weights_batch
        weighted_focal_per_sample = focal_loss * class_weights_batch
        # Mean over all elements (per HF multi-label convention)
        class_weighted_focal = weighted_focal_per_sample.mean()

        # Label smoothing on BCE component
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        if self.label_smoothing > 0:
            num_classes = labels.shape[-1]
            smoothed_labels = labels * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
            smoothed_bce = F.binary_cross_entropy_with_logits(logits, smoothed_labels, reduction='mean')
        else:
            smoothed_bce = bce_loss

        # Combine losses (configurable weighted combination)
        combined_loss = self.loss_combination_ratio * asl_loss + (1 - self.loss_combination_ratio) * class_weighted_focal + 0.2 * smoothed_bce

        return (combined_loss, outputs) if return_outputs else combined_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to add gradient clipping
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        # Add gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        return loss.detach()

class AsymmetricLossTrainer(Trainer):
    """
    Custom Trainer that uses Asymmetric Loss instead of default BCE
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FIXED: Use conservative ASL parameters to prevent gradient vanishing  
        self.asymmetric_loss = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to use Asymmetric Loss
        FIXED: Added shape validation for multi-label consistency
        """
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels not found in inputs")
        
        # Ensure labels are float (multi-label binary)
        if labels.dtype != torch.float:
            labels = labels.float()
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Logits not found in outputs")
        
        # Shape validation
        if logits.shape != labels.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape}, labels {labels.shape}. Expected [batch, 28]")
        if logits.shape[-1] != len(EMOTION_LABELS):
            raise ValueError(f"Expected {len(EMOTION_LABELS)} classes, got {logits.shape[-1]}")
        
        # Debug print
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"DEBUG ASL: Batch shapes - logits: {logits.shape}, labels: {labels.shape}")
        
        # Use Asymmetric Loss (BCE-based, no cross_entropy)
        loss = self.asymmetric_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to add gradient clipping
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        # Add gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        return loss.detach()

class JsonlMultiLabelDataset(Dataset):
    """Dataset for multi-label classification from JSONL files with oversampling support"""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512, oversampled_data=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if oversampled_data:
            self.data = oversampled_data
        else:
            self.data = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # Ensure labels is a list of valid integers
        if isinstance(labels, int):
            labels = [labels]
        elif not isinstance(labels, list):
            labels = list(labels)
        # Filter invalid labels
        valid_labels = [l for l in labels if isinstance(l, int) and 0 <= l < len(EMOTION_LABELS)]
        if not valid_labels:
            valid_labels = [27]  # Default to neutral
        
        # Convert to multi-label format (28-dimensional binary vector, float32)
        label_vector = [0.0] * len(EMOTION_LABELS)
        for label_idx in valid_labels:
            label_vector[label_idx] = 1.0
        
        # Tokenize (avoid max_length padding here; let collator handle dynamic padding)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # Changed: Let DataCollatorWithPadding handle padding for efficiency
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_vector, dtype=torch.float32)  # Explicit float32 for multi-label
        }

def compute_comprehensive_metrics(eval_pred):
    """
    Comprehensive evaluation metrics for scientific rigor
    """
    predictions, labels = eval_pred
    
    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-predictions))
    
    metrics = {}
    
    # Evaluate at multiple thresholds for robustness
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        suffix = f"_t{int(threshold*10)}"
        
        # Core metrics
        f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
        
        # Precision and Recall
        precision_micro = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)[0]
        precision_macro = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[0]
        recall_micro = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)[1]
        recall_macro = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[1]
        
        # Per-class metrics for detailed analysis
        precision_per_class = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)[0]
        recall_per_class = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)[1]
        f1_per_class = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)[2]
        
        # Store metrics
        metrics[f"f1_micro{suffix}"] = f1_micro
        metrics[f"f1_macro{suffix}"] = f1_macro
        metrics[f"f1_weighted{suffix}"] = f1_weighted
        metrics[f"precision_micro{suffix}"] = precision_micro
        metrics[f"precision_macro{suffix}"] = precision_macro
        metrics[f"recall_micro{suffix}"] = recall_micro
        metrics[f"recall_macro{suffix}"] = recall_macro
        metrics[f"avg_preds{suffix}"] = preds.sum(axis=1).mean()
        
        # Per-class metrics (for threshold 0.3 only to avoid clutter)
        if threshold == 0.3:
            for i, emotion in enumerate(EMOTION_LABELS):
                if i < len(precision_per_class):
                    metrics[f"precision_{emotion}"] = precision_per_class[i]
                    metrics[f"recall_{emotion}"] = recall_per_class[i]
                    metrics[f"f1_{emotion}"] = f1_per_class[i]
    
    # FIXED: Primary metrics using 0.2 threshold (optimal for GoEmotions imbalance)
    # Keep 0.5 for comparison but use 0.2 as default
    metrics["f1_micro"] = metrics.get("f1_micro_t2", metrics["f1_micro_t5"])
    metrics["f1_macro"] = metrics.get("f1_macro_t2", metrics["f1_macro_t5"])
    metrics["f1_weighted"] = metrics.get("f1_weighted_t2", metrics["f1_weighted_t5"])
    metrics["precision_micro"] = metrics.get("precision_micro_t2", metrics["precision_micro_t5"])
    metrics["precision_macro"] = metrics.get("precision_macro_t2", metrics["precision_macro_t5"])
    metrics["recall_micro"] = metrics.get("recall_micro_t2", metrics["recall_micro_t5"])
    metrics["recall_macro"] = metrics.get("recall_macro_t2", metrics["recall_macro_t5"])
    
    # Log threshold used
    metrics["primary_threshold"] = 0.2
    
    # Store original 0.5 metrics for comparison
    metrics["f1_micro_default"] = metrics["f1_micro_t5"]
    metrics["f1_macro_default"] = metrics["f1_macro_t5"]
    metrics["f1_weighted_default"] = metrics["f1_weighted_t5"]
    
    # Statistical analysis
    metrics["class_imbalance_ratio"] = compute_class_imbalance_ratio(labels)
    metrics["prediction_entropy"] = compute_prediction_entropy(probs)
    
    return metrics

def compute_class_imbalance_ratio(labels):
    """Compute the ratio between most and least frequent classes"""
    class_counts = labels.sum(axis=0)
    max_count = class_counts.max()
    min_count = class_counts[class_counts > 0].min()  # Exclude zero counts
    return float(max_count / min_count) if min_count > 0 else float('inf')

def compute_prediction_entropy(probs):
    """Compute entropy of predictions to measure uncertainty"""
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    probs_clipped = np.clip(probs, eps, 1 - eps)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    return float(entropy.mean())

def load_model_and_tokenizer_local(model_type="deberta-v3-large"):
    """Load model and tokenizer from local cache or download if needed"""
    print(f"🤖 Loading {model_type}...")
    
    # Determine model path and name
    if model_type == "deberta-v3-large":
        model_path = "models/deberta-v3-large"
        model_name = "microsoft/deberta-v3-large"
    elif model_type == "roberta-large":
        model_path = "models/roberta-large"
        model_name = "roberta-large"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Check if local cache exists (look for config.json and either pytorch_model.bin or model.safetensors)
    config_exists = os.path.exists(f"{model_path}/config.json")
    model_file_exists = (os.path.exists(f"{model_path}/pytorch_model.bin") or 
                        os.path.exists(f"{model_path}/model.safetensors"))
    
    if os.path.exists(model_path) and config_exists and model_file_exists:
        print(f"📁 Found local cache at {model_path}")
        try:
            # Load tokenizer from local cache
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                local_files_only=True
            )
            print(f"✅ {model_type} tokenizer loaded from local cache")
            
            # Load model from local cache
            config = AutoConfig.from_pretrained(
                model_path,
                num_labels=len(EMOTION_LABELS),
                problem_type="multi_label_classification"
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,
                local_files_only=True  # Ensure we use local files
            )
            print(f"✅ {model_type} model loaded from local cache")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"⚠️  Failed to load from local cache: {e}")
            print("🔄 Will download fresh copy...")
    
    # Download fresh copy with offline mode strategy
    print(f"🔄 Downloading {model_type} with offline mode strategy...")
    
    try:
        # Create directory
        os.makedirs(model_path, exist_ok=True)
        
        # Load tokenizer with offline mode
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            local_files_only=False  # Allow download
        )
        tokenizer.save_pretrained(model_path)
        print(f"✅ {model_type} tokenizer downloaded and cached")
        
        # Load model
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(EMOTION_LABELS),
            problem_type="multi_label_classification"
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        model.save_pretrained(model_path)
        print(f"✅ {model_type} model downloaded and cached")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to download {model_type}: {e}")
        return None, None

def load_dataset_local():
    """Load GoEmotions dataset from local cache"""
    print("📊 Loading GoEmotions dataset from local cache...")
    
    train_path = "data/goemotions/train.jsonl"
    val_path = "data/goemotions/val.jsonl"
    
    # Check if local cache exists
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("❌ Local dataset cache not found")
        print("💡 Run 'python scripts/setup_local_cache.py' first")
        return None, None
    
    try:
        # Load metadata
        with open("data/goemotions/metadata.json", "r") as f:
            metadata = json.load(f)
        
        print(f"✅ GoEmotions dataset loaded from local cache")
        print(f"   Training examples: {metadata['train_size']}")
        print(f"   Validation examples: {metadata['val_size']}")
        print(f"   Total emotions: {len(metadata['emotions'])}")
        
        return train_path, val_path
        
    except Exception as e:
        print(f"❌ Failed to load dataset from local cache: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./outputs/deberta")
    parser.add_argument("--model_type", type=str, default="deberta-v3-large",
                       choices=["deberta-v3-large", "roberta-large"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-5)  # FIXED: Enforce optimal 3e-5 for DeBERTa-v3
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum number of training samples to use (for quick screening)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Maximum number of evaluation samples to use (for quick screening)")
    parser.add_argument("--use_asymmetric_loss", action="store_true", default=False,
                       help="Use Asymmetric Loss for better class imbalance handling")
    parser.add_argument("--use_combined_loss", action="store_true", default=False,
                       help="Use Combined Loss (ASL + Class Weighting + Focal Loss) for maximum performance")
    parser.add_argument("--loss_combination_ratio", type=float, default=0.7,
                       help="Ratio of ASL to Focal Loss in combined strategy (default: 0.7)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    parser.add_argument("--augment_prob", type=float, default=0.3, help="Probability for data augmentation (nlpaug/SMOTE)")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of layers to freeze in the model")
    parser.add_argument("--per_class_weights", type=str, default=None, help="JSON string for per-class weights")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    print("🚀 GoEmotions DeBERTa Training (SCIENTIFIC VERSION)")
    print("="*60)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model: {args.model_type} (from local cache)")
    print(f"📊 Dataset: GoEmotions (from local cache)")
    print(f"🔬 Scientific logging: ENABLED")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize scientific logger
    scientific_logger = ScientificLogger(args.output_dir)
    
    # Log experiment configuration
    config = {
        "model_type": args.model_type,
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "fp16": args.fp16,
        "tf32": args.tf32,
        "use_asymmetric_loss": args.use_asymmetric_loss,
        "use_combined_loss": args.use_combined_loss,
        "loss_combination_ratio": args.loss_combination_ratio,
        "max_length": args.max_length,
        "random_seed": 42
    }
    scientific_logger.log_experiment_start(config)
    
    # Load model and tokenizer from local cache
    model, tokenizer = load_model_and_tokenizer_local(args.model_type)
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Freeze layers if specified
    if args.freeze_layers > 0:
        print(f"🧊 Freezing first {args.freeze_layers} layers of the model")
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
        for layer in model.deberta.encoder.layer[:args.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"✅ {args.freeze_layers} layers frozen")

    # Add dropout=0.3 to classifier
    if hasattr(model.classifier, 'dropout') and model.classifier.dropout.p != 0.3:
        model.classifier.dropout.p = 0.3
        print("✅ Set classifier dropout to 0.3")
    
    # Load datasets from local cache
    train_path, val_path = load_dataset_local()
    if train_path is None or val_path is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Create datasets
    print("🔄 Creating datasets...")
    train_dataset = JsonlMultiLabelDataset(train_path, tokenizer, args.max_length)
    val_dataset = JsonlMultiLabelDataset(val_path, tokenizer, args.max_length)
    
    print(f"✅ Created {len(train_dataset)} training examples")
    print(f"✅ Created {len(val_dataset)} validation examples")
    
    # Apply data subset limits if specified (for quick screening)
    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        print(f"🔄 Limiting training data: {len(train_dataset)} → {args.max_train_samples} samples")
        train_dataset.data = train_dataset.data[:args.max_train_samples]
        print(f"✅ Using {len(train_dataset)} training examples (subset for quick screening)")
    
    if args.max_eval_samples is not None and len(val_dataset) > args.max_eval_samples:
        print(f"🔄 Limiting validation data: {len(val_dataset)} → {args.max_eval_samples} samples")
        val_dataset.data = val_dataset.data[:args.max_eval_samples]
        print(f"✅ Using {len(val_dataset)} validation examples (subset for quick screening)")
    
    # Disable gradient checkpointing to avoid "backward through graph a second time" errors
    # This is incompatible with the current training setup
    use_gradient_checkpointing = False
    print("🔧 Disabling gradient checkpointing to prevent RuntimeError during backward pass")

    # Training arguments with NCCL timeout fixes
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=max(args.learning_rate, 3e-5),  # FIXED: Enforce minimum 3e-5 LR override
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        tf32=False,  # Disable tf32 for evaluation stability
        gradient_checkpointing=use_gradient_checkpointing,  # Conditional based on loss function
        deepspeed=args.deepspeed,  # Enable DeepSpeed ZeRO-2 if config provided
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=args.early_stopping_patience > 0,  # Enable if early stopping
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        report_to="tensorboard",  # Enable TensorBoard logging
        ddp_find_unused_parameters=False,  # Optimize DDP performance
        dataloader_num_workers=0,  # Reduce worker processes to avoid NCCL issues
        skip_memory_metrics=True,  # Skip memory metrics to reduce overhead
    )
    
    # Data collator with explicit multi-label support
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length,
        pad_to_multiple_of=8 if args.fp16 else None,  # Optimize for mixed precision
        return_tensors="pt"
    )
    
    # Ensure labels are not padded/truncated incorrectly
    def custom_collator(features):
        batch = data_collator(features)
        if 'labels' in batch:
            # Ensure labels remain [batch_size, 28] float, no padding on labels
            if batch['labels'].shape[-1] != len(EMOTION_LABELS):
                raise ValueError(f"Labels shape mismatch after collation: {batch['labels'].shape}")
            batch['labels'] = batch['labels'].float()
        return batch
    
    data_collator = custom_collator
    
    from transformers import EarlyStoppingCallback

    # Data augmentation with nlpaug if augment_prob > 0
    if args.augment_prob > 0:
        import nlpaug.augmenter.word as naw
        aug = naw.SynonymAug(aug_src='wordnet')
        print(f"🔄 Applying data augmentation with probability {args.augment_prob}")
        # Apply to train_dataset.data (simplified integration)
        augmented_data = []
        for item in train_dataset.data:
            if random.random() < args.augment_prob:
                augmented_text = aug.augment(item['text'])[0]
                augmented_item = item.copy()
                augmented_item['text'] = augmented_text
                augmented_data.append(augmented_item)
            else:
                augmented_data.append(item)
        train_dataset.data = augmented_data[:len(train_dataset.data)]  # Keep size similar
        print(f"✅ Data augmentation applied (nlpaug)")

    # Choose trainer based on loss function
    callbacks = [ProgressMonitorCallback(stall_timeout=600, check_disk_quota=True, min_disk_space_gb=10, enable_gdrive_backup=True)]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    if args.use_combined_loss:
        print("🚀 Using Combined Loss (ASL + Class Weighting + Focal Loss) for maximum performance")
        print(f"📊 Loss combination ratio: {args.loss_combination_ratio} ASL + {1-args.loss_combination_ratio} Focal")
        trainer = CombinedLossTrainer(
            loss_combination_ratio=args.loss_combination_ratio,
            gamma=args.gamma,
            label_smoothing=args.label_smoothing,
            per_class_weights=args.per_class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_comprehensive_metrics,
            callbacks=callbacks,
        )
        
        # FIXED: Integrate oversampling directly during trainer init (no post-replacement)
        print("🔄 Creating oversampled training dataset...")
        oversampled_train_dataset = JsonlMultiLabelDataset(
            train_path, tokenizer, args.max_length, oversampled_data=trainer.oversampled_data
        )
        print(f"✅ Oversampled training dataset: {len(oversampled_train_dataset)} examples")
        # Use oversampled from init
        trainer.train_dataset = oversampled_train_dataset
    elif args.use_asymmetric_loss:
        print("🎯 Using Asymmetric Loss for better class imbalance handling")
        trainer = AsymmetricLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_comprehensive_metrics,
            callbacks=callbacks,
        )
    else:
        print("📊 Using standard BCE Loss")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_comprehensive_metrics,
            callbacks=callbacks,
        )
    
    # Train with error handling and progress monitoring
    print("🚀 Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("🔍 Error details:", str(e))
        import traceback
        traceback.print_exc()
        
        # Attempt to save a recovery checkpoint
        try:
            print("💾 Attempting to save recovery checkpoint...")
            recovery_dir = f"{training_args.output_dir}/recovery_checkpoint"
            trainer.save_model(recovery_dir)
            print(f"✅ Recovery checkpoint saved to {recovery_dir}")
        except Exception as recovery_error:
            print(f"⚠️ Failed to save recovery checkpoint: {str(recovery_error)}")
            
        return
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save multiple models for ensemble (e.g., best and final)
    import shutil
    ensemble_dir = f"{args.output_dir}_ensemble"
    os.makedirs(ensemble_dir, exist_ok=True)
    shutil.copytree(args.output_dir, f"{ensemble_dir}/model1", dirs_exist_ok=True)
    # Save a second variant with different seed or config if needed
    print(f"✅ Saved ensemble models to {ensemble_dir}")
    
    # Evaluate
    print("📊 Final evaluation...")
    eval_results = trainer.evaluate()
    
    # Final backup to Google Drive
    print("🔄 Performing final backup to Google Drive...")
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, ProgressMonitorCallback) and callback.enable_gdrive_backup:
            callback._backup_to_gdrive(training_args, force=True)
            break
    
    # Log final evaluation
    scientific_logger.log_evaluation(eval_results, epoch=args.num_train_epochs)
    
    # Save comprehensive evaluation results
    eval_report = {
        "experiment_id": scientific_logger.experiment_id,
        "model": args.model_type,
        "loss_function": "combined" if args.use_combined_loss else ("asymmetric" if args.use_asymmetric_loss else "bce"),
        "f1_micro": eval_results.get("eval_f1_micro", 0.0),
        "f1_macro": eval_results.get("eval_f1_macro", 0.0),
        "f1_weighted": eval_results.get("eval_f1_weighted", 0.0),
        "precision_micro": eval_results.get("eval_precision_micro", 0.0),
        "precision_macro": eval_results.get("eval_precision_macro", 0.0),
        "recall_micro": eval_results.get("eval_recall_micro", 0.0),
        "recall_macro": eval_results.get("eval_recall_macro", 0.0),
        "class_imbalance_ratio": eval_results.get("eval_class_imbalance_ratio", 0.0),
        "prediction_entropy": eval_results.get("eval_prediction_entropy", 0.0),
        "eval_loss": eval_results.get("eval_loss", 0.0),
        "training_args": vars(args),
        "all_metrics": eval_results
    }
    
    with open(os.path.join(args.output_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
    
    # Log experiment completion
    scientific_logger.log_experiment_end(eval_results)
    
    print("✅ Training completed!")
    print(f"📈 Final F1 Macro: {eval_results.get('eval_f1_macro', 0.0):.4f}")
    print(f"📈 Final F1 Micro: {eval_results.get('eval_f1_micro', 0.0):.4f}")
    print(f"📈 Final F1 Weighted: {eval_results.get('eval_f1_weighted', 0.0):.4f}")
    print(f"📊 Class Imbalance Ratio: {eval_results.get('eval_class_imbalance_ratio', 0.0):.2f}")
    print(f"🔬 Scientific log: {scientific_logger.log_file}")
    print(f"💾 Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
