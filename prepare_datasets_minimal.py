#!/usr/bin/env python3
"""
Minimal Multi-Dataset Preparation Script for Emotion Classification
Combines GoEmotions with synthetic data for other datasets

Author: SAMo Multi-Dataset Pipeline
Date: 2025-01-03
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalMultiDatasetPreparator:
    """Prepares and combines multiple emotion datasets for training"""
    
    def __init__(self, output_dir: str = "data/combined"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Unified emotion mapping (28 emotions from GoEmotions as base)
        self.emotion_mapping = {
            'goemotions': {
                'admiration': 0, 'amusement': 1, 'anger': 2, 'annoyance': 3, 'approval': 4,
                'caring': 5, 'confusion': 6, 'curiosity': 7, 'desire': 8, 'disappointment': 9,
                'disapproval': 10, 'disgust': 11, 'embarrassment': 12, 'excitement': 13,
                'fear': 14, 'gratitude': 15, 'grief': 16, 'joy': 17, 'love': 18,
                'nervousness': 19, 'optimism': 20, 'pride': 21, 'realization': 22,
                'relief': 23, 'remorse': 24, 'sadness': 25, 'surprise': 26, 'neutral': 27
            },
            'semeval': {
                'anger': 2, 'fear': 14, 'joy': 17, 'love': 18, 'sadness': 25,
                'surprise': 26, 'disgust': 11, 'shame': 12, 'guilt': 24, 'contempt': 10,
                'neutral': 27
            },
            'isear': {
                'joy': 17, 'fear': 14, 'anger': 2, 'sadness': 25, 'disgust': 11,
                'shame': 12, 'guilt': 24
            },
            'meld': {
                'joy': 17, 'sadness': 25, 'anger': 2, 'fear': 14, 'surprise': 26,
                'disgust': 11, 'neutral': 27
            }
        }
    
    def load_goemotions(self) -> Tuple[List[Dict], List[Dict]]:
        """Load GoEmotions dataset"""
        logger.info("📊 Loading GoEmotions dataset...")
        
        train_data = []
        val_data = []
        
        # Load training data
        train_path = Path('data/goemotions/train.jsonl')
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    train_data.append({
                        'text': data['text'],
                        'labels': data['labels'],
                        'dataset': 'goemotions'
                    })
        
        # Load validation data
        val_path = Path('data/goemotions/val.jsonl')
        if val_path.exists():
            with open(val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    val_data.append({
                        'text': data['text'],
                        'labels': data['labels'],
                        'dataset': 'goemotions'
                    })
        
        logger.info(f"✅ GoEmotions loaded: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    def create_synthetic_semeval(self) -> Tuple[List[Dict], List[Dict]]:
        """Create synthetic SemEval data"""
        logger.info("🔄 Creating synthetic SemEval data...")
        
        synthetic_texts = [
            "I am so angry about this situation!",
            "This makes me really happy and excited!",
            "I feel sad and disappointed about the outcome.",
            "I'm surprised by this unexpected turn of events.",
            "This is disgusting and I can't stand it.",
            "I feel afraid and nervous about what might happen.",
            "I love this so much, it brings me joy!",
            "I feel guilty about what I did wrong.",
            "I'm ashamed of my behavior yesterday.",
            "I have contempt for this kind of behavior.",
            "I feel neutral about this situation."
        ]
        
        emotions = ['anger', 'joy', 'sadness', 'surprise', 'disgust', 'fear', 'love', 'guilt', 'shame', 'contempt', 'neutral']
        
        train_data = []
        val_data = []
        
        # Create multiple variations of each emotion
        for emotion in emotions:
            if emotion in self.emotion_mapping['semeval']:
                label_idx = self.emotion_mapping['semeval'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                # Create 10 variations per emotion
                for i in range(10):
                    text = f"{synthetic_texts[emotions.index(emotion)]} (variation {i+1})"
                    
                    if i < 8:  # 80% for training
                        train_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'semeval_synthetic'
                        })
                    else:  # 20% for validation
                        val_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'semeval_synthetic'
                        })
        
        logger.info(f"✅ Synthetic SemEval created: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    def create_synthetic_isear(self) -> Tuple[List[Dict], List[Dict]]:
        """Create synthetic ISEAR data"""
        logger.info("🔄 Creating synthetic ISEAR data...")
        
        synthetic_texts = [
            "I feel joyful and happy about this achievement!",
            "I'm scared and afraid of what might happen next.",
            "I'm angry and frustrated with this situation.",
            "I feel sad and depressed about the loss.",
            "This is disgusting and I can't stand it.",
            "I feel ashamed of my past mistakes.",
            "I feel guilty about hurting someone's feelings."
        ]
        
        emotions = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']
        
        train_data = []
        val_data = []
        
        # Create multiple variations of each emotion
        for emotion in emotions:
            if emotion in self.emotion_mapping['isear']:
                label_idx = self.emotion_mapping['isear'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                # Create 20 variations per emotion
                for i in range(20):
                    text = f"{synthetic_texts[emotions.index(emotion)]} (variation {i+1})"
                    
                    if i < 16:  # 80% for training
                        train_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'isear_synthetic'
                        })
                    else:  # 20% for validation
                        val_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'isear_synthetic'
                        })
        
        logger.info(f"✅ Synthetic ISEAR created: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    def create_synthetic_meld(self) -> Tuple[List[Dict], List[Dict]]:
        """Create synthetic MELD data"""
        logger.info("🔄 Creating synthetic MELD data...")
        
        synthetic_texts = [
            "I'm so happy and joyful about this news!",
            "I feel sad and disappointed about the outcome.",
            "I'm angry and frustrated with this situation.",
            "I'm scared and afraid of what might happen.",
            "I'm surprised by this unexpected development.",
            "This is disgusting and I can't stand it.",
            "I feel neutral about this situation."
        ]
        
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        
        train_data = []
        val_data = []
        
        # Create multiple variations of each emotion
        for emotion in emotions:
            if emotion in self.emotion_mapping['meld']:
                label_idx = self.emotion_mapping['meld'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                # Create 30 variations per emotion
                for i in range(30):
                    text = f"{synthetic_texts[emotions.index(emotion)]} (variation {i+1})"
                    
                    if i < 24:  # 80% for training
                        train_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'meld_synthetic'
                        })
                    else:  # 20% for validation
                        val_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'meld_synthetic'
                        })
        
        logger.info(f"✅ Synthetic MELD created: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    def combine_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """Combine all datasets into unified format"""
        logger.info("🔄 Combining all datasets...")
        
        # Load all datasets
        goemotions_train, goemotions_val = self.load_goemotions()
        semeval_train, semeval_val = self.create_synthetic_semeval()
        isear_train, isear_val = self.create_synthetic_isear()
        meld_train, meld_val = self.create_synthetic_meld()
        
        # Combine all training data
        combined_train = goemotions_train + semeval_train + isear_train + meld_train
        
        # Combine all validation data
        combined_val = goemotions_val + semeval_val + isear_val + meld_val
        
        # Shuffle the combined datasets
        random.shuffle(combined_train)
        random.shuffle(combined_val)
        
        logger.info(f"✅ Combined datasets:")
        logger.info(f"   📊 Total training samples: {len(combined_train)}")
        logger.info(f"   📊 Total validation samples: {len(combined_val)}")
        logger.info(f"   📊 GoEmotions: {len(goemotions_train)} train, {len(goemotions_val)} val")
        logger.info(f"   📊 SemEval: {len(semeval_train)} train, {len(semeval_val)} val")
        logger.info(f"   📊 ISEAR: {len(isear_train)} train, {len(isear_val)} val")
        logger.info(f"   📊 MELD: {len(meld_train)} train, {len(meld_val)} val")
        
        return combined_train, combined_val
    
    def save_combined_datasets(self, train_data: List[Dict], val_data: List[Dict]):
        """Save combined datasets to files"""
        logger.info("💾 Saving combined datasets...")
        
        # Save training data
        train_path = self.output_dir / 'train.jsonl'
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save validation data
        val_path = self.output_dir / 'val.jsonl'
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save metadata
        metadata = {
            'dataset': 'combined_emotion_classification',
            'train_size': len(train_data),
            'val_size': len(val_data),
            'total_size': len(train_data) + len(val_data),
            'emotions': list(self.emotion_mapping['goemotions'].keys()),
            'emotion_count': 28,
            'datasets_included': ['goemotions', 'semeval_synthetic', 'isear_synthetic', 'meld_synthetic'],
            'created_at': datetime.now().isoformat(),
            'emotion_mapping': self.emotion_mapping
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Datasets saved to {self.output_dir}")
        logger.info(f"   📁 Training data: {train_path}")
        logger.info(f"   📁 Validation data: {val_path}")
        logger.info(f"   📁 Metadata: {metadata_path}")
    
    def run(self):
        """Run the complete dataset preparation pipeline"""
        logger.info("🚀 Starting Minimal Multi-Dataset Preparation Pipeline")
        logger.info("=" * 60)
        
        try:
            # Combine all datasets
            train_data, val_data = self.combine_datasets()
            
            # Save combined datasets
            self.save_combined_datasets(train_data, val_data)
            
            logger.info("🎉 Multi-Dataset Preparation Complete!")
            logger.info("=" * 60)
            logger.info(f"📊 Final Statistics:")
            logger.info(f"   • Training samples: {len(train_data):,}")
            logger.info(f"   • Validation samples: {len(val_data):,}")
            logger.info(f"   • Total samples: {len(train_data) + len(val_data):,}")
            logger.info(f"   • Emotion classes: 28")
            logger.info(f"   • Output directory: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error in dataset preparation: {e}")
            raise

def main():
    """Main function to run dataset preparation"""
    preparator = MinimalMultiDatasetPreparator()
    preparator.run()

if __name__ == "__main__":
    main()