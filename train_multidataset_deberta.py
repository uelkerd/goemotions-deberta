#!/usr/bin/env python3
"""
Multi-Dataset DeBERTa-v3-large Training Script
Trains on combined GoEmotions, SemEval, ISEAR, and MELD datasets

Author: SAMo Multi-Dataset Pipeline
Date: 2025-01-03
"""

import os
import json
import math
import random
import argparse
import warnings
from pathlib import Path
import subprocess
import time
from datetime import datetime

# Suppress compatibility warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers") 
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Set environment variables for DeBERTa-v3-large compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"

# NCCL configuration to prevent timeouts
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 min timeout
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

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
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class MultiDatasetEmotionDataset(Dataset):
    """Dataset class for multi-dataset emotion classification"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        logger.info(f"📊 Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['labels'], dtype=torch.float)
        }

class ScientificLogger:
    """Comprehensive scientific logging system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"MultiDataset_BCE_{self.timestamp}"
        
        # Create experiment directory
        self.experiment_dir = self.log_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logs
        self.training_log = []
        self.evaluation_log = []
        self.system_log = []
        
        logger.info(f"🔬 Scientific Logger initialized: {self.experiment_id}")
    
    def log_training_step(self, step: int, loss: float, learning_rate: float, **kwargs):
        """Log training step information"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            **kwargs
        }
        self.training_log.append(log_entry)
    
    def log_evaluation(self, metrics: Dict[str, float], epoch: int, **kwargs):
        """Log evaluation metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics,
            **kwargs
        }
        self.evaluation_log.append(log_entry)
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'system_info': info
        }
        self.system_log.append(log_entry)
    
    def save_logs(self):
        """Save all logs to files"""
        # Save training log
        with open(self.experiment_dir / 'training_log.json', 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # Save evaluation log
        with open(self.experiment_dir / 'evaluation_log.json', 'w') as f:
            json.dump(self.evaluation_log, f, indent=2)
        
        # Save system log
        with open(self.experiment_dir / 'system_log.json', 'w') as f:
            json.dump(self.system_log, f, indent=2)
        
        # Save experiment summary
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp,
            'total_training_steps': len(self.training_log),
            'total_evaluations': len(self.evaluation_log),
            'final_metrics': self.evaluation_log[-1]['metrics'] if self.evaluation_log else None
        }
        
        with open(self.experiment_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Scientific logs saved to {self.experiment_dir}")

class GoogleDriveBackup:
    """Google Drive backup system"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.backup_path = f"drive:00_Projects/🎯 TechLabs-2025/Final_Project/TRAINING/MultiDataset_BCE_{experiment_id}/"
        
    def backup_experiment(self, source_dir: str):
        """Backup experiment to Google Drive"""
        try:
            logger.info(f"📤 Backing up experiment to Google Drive...")
            
            # Create backup directory
            subprocess.run(['rclone', 'mkdir', self.backup_path], check=True)
            
            # Backup files
            subprocess.run([
                'rclone', 'copy', source_dir, self.backup_path,
                '--drive-pacer-min-sleep=1s',
                '--drive-pacer-burst=10'
            ], check=True)
            
            logger.info(f"✅ Experiment backed up to {self.backup_path}")
            
        except Exception as e:
            logger.error(f"❌ Google Drive backup failed: {e}")

class ProgressMonitorCallback(TrainerCallback):
    """Enhanced progress monitoring with scientific logging"""
    
    def __init__(self, scientific_logger: ScientificLogger, gdrive_backup: GoogleDriveBackup):
        self.scientific_logger = scientific_logger
        self.gdrive_backup = gdrive_backup
        self.last_backup_time = time.time()
        self.backup_interval = 1800  # 30 minutes
    
    def on_step_end(self, args, state, control, **kwargs):
        # Log training step
        if 'log' in kwargs:
            log_dict = kwargs['log']
            self.scientific_logger.log_training_step(
                step=state.global_step,
                loss=log_dict.get('loss', 0.0),
                learning_rate=log_dict.get('learning_rate', 0.0),
                epoch=state.epoch
            )
        
        # Periodic backup
        current_time = time.time()
        if current_time - self.last_backup_time > self.backup_interval:
            self.last_backup_time = current_time
            self.gdrive_backup.backup_experiment(str(self.scientific_logger.experiment_dir))
        
        return control
    
    def on_evaluate(self, args, state, control, **kwargs):
        # Log evaluation metrics
        if 'metrics' in kwargs:
            self.scientific_logger.log_evaluation(
                metrics=kwargs['metrics'],
                epoch=state.epoch,
                step=state.global_step
            )
        
        return control

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions))
    
    # Convert to binary predictions
    binary_predictions = (predictions > 0.2).float()
    
    # Compute metrics
    f1_macro = f1_score(labels, binary_predictions, average='macro', zero_division=0)
    f1_micro = f1_score(labels, binary_predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, binary_predictions, average='weighted', zero_division=0)
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Multi-Dataset DeBERTa Training')
    parser.add_argument('--data_dir', type=str, default='data/combined', help='Combined dataset directory')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large', help='Model name')
    parser.add_argument('--output_dir', type=str, default='outputs/multidataset', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save steps')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_seeds(42)
    
    # Initialize scientific logger
    scientific_logger = ScientificLogger()
    
    # Initialize Google Drive backup
    gdrive_backup = GoogleDriveBackup(scientific_logger.timestamp)
    
    # Log system information
    system_info = {
        'gpu_count': torch.cuda.device_count(),
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'training_args': vars(args)
    }
    scientific_logger.log_system_info(system_info)
    
    logger.info("🚀 Starting Multi-Dataset DeBERTa Training")
    logger.info("=" * 60)
    logger.info(f"📊 Experiment ID: {scientific_logger.experiment_id}")
    logger.info(f"📁 Data directory: {args.data_dir}")
    logger.info(f"🤖 Model: {args.model_name}")
    logger.info(f"📈 Epochs: {args.num_epochs}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    logger.info(f"🎯 Learning rate: {args.learning_rate}")
    
    try:
        # Load tokenizer and model
        logger.info("📥 Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Load model configuration
        config = AutoConfig.from_pretrained(args.model_name)
        config.num_labels = 28  # 28 emotion classes
        config.problem_type = "multi_label_classification"
        
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            config=config
        )
        
        # Load datasets
        logger.info("📊 Loading datasets...")
        train_dataset = MultiDatasetEmotionDataset(
            os.path.join(args.data_dir, 'train.jsonl'),
            tokenizer,
            args.max_length
        )
        
        val_dataset = MultiDatasetEmotionDataset(
            os.path.join(args.data_dir, 'val.jsonl'),
            tokenizer,
            args.max_length
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to="tensorboard",
            logging_dir=f"{args.output_dir}/logs",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=4,
            fp16=True,
            gradient_accumulation_steps=2,
            save_total_limit=3,
            seed=42
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[ProgressMonitorCallback(scientific_logger, gdrive_backup)]
        )
        
        # Start training
        logger.info("🏃 Starting training...")
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"⏱️ Training completed in {training_time/3600:.2f} hours")
        
        # Final evaluation
        logger.info("📊 Running final evaluation...")
        eval_results = trainer.evaluate()
        
        logger.info("📈 Final Results:")
        for metric, value in eval_results.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        # Save scientific logs
        scientific_logger.save_logs()
        
        # Final backup to Google Drive
        gdrive_backup.backup_experiment(str(scientific_logger.experiment_dir))
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Model saved to: {args.output_dir}")
        logger.info(f"📊 Logs saved to: {scientific_logger.experiment_dir}")
        logger.info(f"☁️ Backup available at: {gdrive_backup.backup_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        # Save logs even on failure
        scientific_logger.save_logs()
        gdrive_backup.backup_experiment(str(scientific_logger.experiment_dir))
        raise

if __name__ == "__main__":
    main()