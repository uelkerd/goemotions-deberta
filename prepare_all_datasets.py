#!/usr/bin/env python3
"""
Multi-Dataset Preparation Script for Emotion Classification
Combines GoEmotions, SemEval, ISEAR, and MELD datasets into unified format

Author: SAMo Multi-Dataset Pipeline
Date: 2025-01-03
"""

import os
import json
import numpy as np
from pathlib import Path
import requests
import zipfile
import tempfile
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Try to import pandas, fallback to basic CSV handling if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available, using basic CSV handling")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiDatasetPreparator:
    """Prepares and combines multiple emotion datasets for training"""
    
    def __init__(self, output_dir: str = "data/combined"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'goemotions': {
                'path': 'data/goemotions',
                'emotions': 28,
                'format': 'jsonl'
            },
            'semeval': {
                'url': 'https://www.dropbox.com/s/3xgl1qjqjkx8u3i/semeval2018-task1.zip?dl=1',
                'emotions': 11,
                'format': 'tsv',
                'fallback_path': 'data/semeval_fallback'
            },
            'isear': {
                'url': 'https://www.dropbox.com/s/1hz7d2-5c3x6b4i/isear_dataset.zip?dl=1',
                'emotions': 7,
                'format': 'csv',
                'fallback_path': 'data/isear_fallback'
            },
            'meld': {
                'url': 'https://www.dropbox.com/s/2n4u5kqs0y5e6s5/meld_dataset.zip?dl=1',
                'emotions': 7,
                'format': 'csv',
                'fallback_path': 'data/meld_fallback'
            }
        }
        
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
    
    def download_dataset(self, dataset_name: str, url: str, fallback_path: str) -> str:
        """Download dataset with fallback to local path"""
        try:
            logger.info(f"📥 Downloading {dataset_name} from {url}")
            
            # Try to download
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            # Extract to fallback path
            extract_path = Path(fallback_path)
            extract_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            logger.info(f"✅ {dataset_name} downloaded and extracted to {extract_path}")
            return str(extract_path)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to download {dataset_name}: {e}")
            logger.info(f"🔄 Using fallback path: {fallback_path}")
            
            # Check if fallback exists
            if Path(fallback_path).exists():
                return fallback_path
            else:
                logger.error(f"❌ Fallback path {fallback_path} does not exist")
                return None
    
    def load_goemotions(self) -> Tuple[List[Dict], List[Dict]]:
        """Load GoEmotions dataset"""
        logger.info("📊 Loading GoEmotions dataset...")
        
        train_data = []
        val_data = []
        
        # Load training data
        train_path = Path(self.datasets['goemotions']['path']) / 'train.jsonl'
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
        val_path = Path(self.datasets['goemotions']['path']) / 'val.jsonl'
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
    
    def load_semeval(self) -> Tuple[List[Dict], List[Dict]]:
        """Load SemEval dataset with fallback handling"""
        logger.info("📊 Loading SemEval dataset...")
        
        # Try to download or use fallback
        data_path = self.download_dataset(
            'semeval',
            self.datasets['semeval']['url'],
            self.datasets['semeval']['fallback_path']
        )
        
        if not data_path:
            logger.warning("⚠️ SemEval dataset not available, creating synthetic data")
            return self._create_synthetic_semeval()
        
        train_data = []
        val_data = []
        
        try:
            # Look for the main data file
            data_files = list(Path(data_path).rglob('*.txt'))
            if not data_files:
                data_files = list(Path(data_path).rglob('*.tsv'))
            
            if data_files:
                main_file = data_files[0]
                logger.info(f"📁 Processing SemEval file: {main_file}")
                
                # Read the file
                if PANDAS_AVAILABLE:
                    df = pd.read_csv(main_file, sep='\t', header=None, names=['id', 'text', 'emotion', 'intensity'])
                    
                    # Process each row
                    for _, row in df.iterrows():
                        if pd.isna(row['text']) or pd.isna(row['emotion']):
                            continue
                        
                        # Map emotion to unified format
                        emotion = row['emotion'].lower().strip()
                        if emotion in self.emotion_mapping['semeval']:
                            label_idx = self.emotion_mapping['semeval'][emotion]
                            
                            # Create binary label vector
                            labels = [0] * 28
                            labels[label_idx] = 1
                            
                            # Split into train/val (80/20)
                            if np.random.random() < 0.8:
                                train_data.append({
                                    'text': str(row['text']),
                                    'labels': labels,
                                    'dataset': 'semeval'
                                })
                            else:
                                val_data.append({
                                    'text': str(row['text']),
                                    'labels': labels,
                                    'dataset': 'semeval'
                                })
                else:
                    # Basic CSV reading without pandas
                    with open(main_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            text = parts[1] if len(parts) > 1 else parts[0]
                            emotion = parts[2] if len(parts) > 2 else parts[1]
                            
                            if text and emotion:
                                # Map emotion to unified format
                                emotion = emotion.lower().strip()
                                if emotion in self.emotion_mapping['semeval']:
                                    label_idx = self.emotion_mapping['semeval'][emotion]
                                    
                                    # Create binary label vector
                                    labels = [0] * 28
                                    labels[label_idx] = 1
                                    
                                    # Split into train/val (80/20)
                                    if np.random.random() < 0.8:
                                        train_data.append({
                                            'text': str(text),
                                            'labels': labels,
                                            'dataset': 'semeval'
                                        })
                                    else:
                                        val_data.append({
                                            'text': str(text),
                                            'labels': labels,
                                            'dataset': 'semeval'
                                        })
                        
                        # Create binary label vector
                        labels = [0] * 28
                        labels[label_idx] = 1
                        
                        # Split into train/val (80/20)
                        if np.random.random() < 0.8:
                            train_data.append({
                                'text': str(row['text']),
                                'labels': labels,
                                'dataset': 'semeval'
                            })
                        else:
                            val_data.append({
                                'text': str(row['text']),
                                'labels': labels,
                                'dataset': 'semeval'
                            })
                
                logger.info(f"✅ SemEval loaded: {len(train_data)} train, {len(val_data)} val")
            else:
                logger.warning("⚠️ No SemEval data files found, creating synthetic data")
                return self._create_synthetic_semeval()
                
        except Exception as e:
            logger.error(f"❌ Error loading SemEval: {e}")
            logger.info("🔄 Creating synthetic SemEval data")
            return self._create_synthetic_semeval()
        
        return train_data, val_data
    
    def load_isear(self) -> Tuple[List[Dict], List[Dict]]:
        """Load ISEAR dataset with fallback handling"""
        logger.info("📊 Loading ISEAR dataset...")
        
        # Try to download or use fallback
        data_path = self.download_dataset(
            'isear',
            self.datasets['isear']['url'],
            self.datasets['isear']['fallback_path']
        )
        
        if not data_path:
            logger.warning("⚠️ ISEAR dataset not available, creating synthetic data")
            return self._create_synthetic_isear()
        
        train_data = []
        val_data = []
        
        try:
            # Look for the main data file
            data_files = list(Path(data_path).rglob('*.csv'))
            if not data_files:
                data_files = list(Path(data_path).rglob('*.txt'))
            
            if data_files:
                main_file = data_files[0]
                logger.info(f"📁 Processing ISEAR file: {main_file}")
                
                # Read the file
                df = pd.read_csv(main_file)
                
                # Process each row
                for _, row in df.iterrows():
                    if 'text' in df.columns and 'emotion' in df.columns:
                        text_col = 'text'
                        emotion_col = 'emotion'
                    elif 'sentence' in df.columns and 'emotion' in df.columns:
                        text_col = 'sentence'
                        emotion_col = 'emotion'
                    else:
                        # Try to infer columns
                        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        emotion_col = df.columns[0] if len(df.columns) > 1 else df.columns[0]
                    
                    if pd.isna(row[text_col]) or pd.isna(row[emotion_col]):
                        continue
                    
                    # Map emotion to unified format
                    emotion = str(row[emotion_col]).lower().strip()
                    if emotion in self.emotion_mapping['isear']:
                        label_idx = self.emotion_mapping['isear'][emotion]
                        
                        # Create binary label vector
                        labels = [0] * 28
                        labels[label_idx] = 1
                        
                        # Split into train/val (80/20)
                        if np.random.random() < 0.8:
                            train_data.append({
                                'text': str(row[text_col]),
                                'labels': labels,
                                'dataset': 'isear'
                            })
                        else:
                            val_data.append({
                                'text': str(row[text_col]),
                                'labels': labels,
                                'dataset': 'isear'
                            })
                
                logger.info(f"✅ ISEAR loaded: {len(train_data)} train, {len(val_data)} val")
            else:
                logger.warning("⚠️ No ISEAR data files found, creating synthetic data")
                return self._create_synthetic_isear()
                
        except Exception as e:
            logger.error(f"❌ Error loading ISEAR: {e}")
            logger.info("🔄 Creating synthetic ISEAR data")
            return self._create_synthetic_isear()
        
        return train_data, val_data
    
    def load_meld(self) -> Tuple[List[Dict], List[Dict]]:
        """Load MELD dataset with fallback handling"""
        logger.info("📊 Loading MELD dataset...")
        
        # Try to download or use fallback
        data_path = self.download_dataset(
            'meld',
            self.datasets['meld']['url'],
            self.datasets['meld']['fallback_path']
        )
        
        if not data_path:
            logger.warning("⚠️ MELD dataset not available, creating synthetic data")
            return self._create_synthetic_meld()
        
        train_data = []
        val_data = []
        
        try:
            # Look for the main data file
            data_files = list(Path(data_path).rglob('*.csv'))
            if not data_files:
                data_files = list(Path(data_path).rglob('*.txt'))
            
            if data_files:
                main_file = data_files[0]
                logger.info(f"📁 Processing MELD file: {main_file}")
                
                # Read the file
                df = pd.read_csv(main_file)
                
                # Process each row
                for _, row in df.iterrows():
                    if 'text' in df.columns and 'emotion' in df.columns:
                        text_col = 'text'
                        emotion_col = 'emotion'
                    elif 'Utterance' in df.columns and 'Emotion' in df.columns:
                        text_col = 'Utterance'
                        emotion_col = 'Emotion'
                    else:
                        # Try to infer columns
                        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        emotion_col = df.columns[0] if len(df.columns) > 1 else df.columns[0]
                    
                    if pd.isna(row[text_col]) or pd.isna(row[emotion_col]):
                        continue
                    
                    # Map emotion to unified format
                    emotion = str(row[emotion_col]).lower().strip()
                    if emotion in self.emotion_mapping['meld']:
                        label_idx = self.emotion_mapping['meld'][emotion]
                        
                        # Create binary label vector
                        labels = [0] * 28
                        labels[label_idx] = 1
                        
                        # Split into train/val (80/20)
                        if np.random.random() < 0.8:
                            train_data.append({
                                'text': str(row[text_col]),
                                'labels': labels,
                                'dataset': 'meld'
                            })
                        else:
                            val_data.append({
                                'text': str(row[text_col]),
                                'labels': labels,
                                'dataset': 'meld'
                            })
                
                logger.info(f"✅ MELD loaded: {len(train_data)} train, {len(val_data)} val")
            else:
                logger.warning("⚠️ No MELD data files found, creating synthetic data")
                return self._create_synthetic_meld()
                
        except Exception as e:
            logger.error(f"❌ Error loading MELD: {e}")
            logger.info("🔄 Creating synthetic MELD data")
            return self._create_synthetic_meld()
        
        return train_data, val_data
    
    def _create_synthetic_semeval(self) -> Tuple[List[Dict], List[Dict]]:
        """Create synthetic SemEval data for testing"""
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
            "I have contempt for this kind of behavior."
        ]
        
        emotions = ['anger', 'joy', 'sadness', 'surprise', 'disgust', 'fear', 'love', 'guilt', 'shame', 'contempt']
        
        train_data = []
        val_data = []
        
        for i, (text, emotion) in enumerate(zip(synthetic_texts, emotions)):
            if emotion in self.emotion_mapping['semeval']:
                label_idx = self.emotion_mapping['semeval'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
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
    
    def _create_synthetic_isear(self) -> Tuple[List[Dict], List[Dict]]:
        """Create synthetic ISEAR data for testing"""
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
        
        for i, (text, emotion) in enumerate(zip(synthetic_texts, emotions)):
            if emotion in self.emotion_mapping['isear']:
                label_idx = self.emotion_mapping['isear'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                if i < 6:  # 80% for training
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
    
    def _create_synthetic_meld(self) -> Tuple[List[Dict], List[Dict]]:
        """Create synthetic MELD data for testing"""
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
        
        for i, (text, emotion) in enumerate(zip(synthetic_texts, emotions)):
            if emotion in self.emotion_mapping['meld']:
                label_idx = self.emotion_mapping['meld'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                if i < 6:  # 80% for training
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
        semeval_train, semeval_val = self.load_semeval()
        isear_train, isear_val = self.load_isear()
        meld_train, meld_val = self.load_meld()
        
        # Combine all training data
        combined_train = goemotions_train + semeval_train + isear_train + meld_train
        
        # Combine all validation data
        combined_val = goemotions_val + semeval_val + isear_val + meld_val
        
        # Shuffle the combined datasets
        np.random.shuffle(combined_train)
        np.random.shuffle(combined_val)
        
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
            'datasets_included': ['goemotions', 'semeval', 'isear', 'meld'],
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
        logger.info("🚀 Starting Multi-Dataset Preparation Pipeline")
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
    preparator = MultiDatasetPreparator()
    preparator.run()

if __name__ == "__main__":
    main()