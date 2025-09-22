#!/usr/bin/env python3
"""
Realistic Multi-Dataset Preparation Script for Emotion Classification
Creates realistic fallback data for SemEval, ISEAR, and MELD datasets

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

class RealisticMultiDatasetPreparator:
    """Prepares and combines realistic emotion datasets for training"""
    
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
    
    def create_realistic_semeval(self) -> Tuple[List[Dict], List[Dict]]:
        """Create realistic SemEval-style data"""
        logger.info("🔄 Creating realistic SemEval data...")
        
        # More realistic and diverse SemEval-style data
        semeval_templates = {
            'anger': [
                "I'm absolutely furious about this situation!",
                "This makes me so angry I can't even think straight!",
                "I'm boiling with rage right now!",
                "This is completely unacceptable and infuriating!",
                "I'm so mad I could scream!",
                "This injustice is making me furious!",
                "I'm absolutely livid about this!",
                "This is the most frustrating thing ever!",
                "I'm so angry I can't contain myself!",
                "This situation is absolutely maddening!"
            ],
            'fear': [
                "I'm terrified of what might happen next!",
                "This situation is really scary and unsettling!",
                "I'm afraid this won't end well for anyone!",
                "This is absolutely frightening!",
                "I'm so scared I can't sleep!",
                "This fear is overwhelming me!",
                "I'm terrified of the consequences!",
                "This is the most frightening thing I've ever seen!",
                "I'm so afraid I can't think clearly!",
                "This situation fills me with dread!"
            ],
            'joy': [
                "I'm so happy and excited about this news!",
                "This brings me incredible joy and happiness!",
                "I'm over the moon with excitement!",
                "This is the best news I've heard in ages!",
                "I'm absolutely thrilled about this!",
                "This joy is overwhelming and wonderful!",
                "I'm so happy I could dance!",
                "This is the most joyful moment ever!",
                "I'm filled with pure happiness!",
                "This brings me so much joy!"
            ],
            'love': [
                "I love this so much it makes my heart sing!",
                "This fills my heart with so much love!",
                "I'm deeply in love with everything about this!",
                "This is absolutely lovely and wonderful!",
                "I adore this with all my heart!",
                "This love is pure and beautiful!",
                "I'm so in love I can't contain it!",
                "This is the most loving thing ever!",
                "I'm filled with love and warmth!",
                "This brings me so much love!"
            ],
            'sadness': [
                "I'm so sad about this outcome!",
                "This breaks my heart completely!",
                "I'm feeling really depressed and down!",
                "This sadness is overwhelming me!",
                "I'm so sad I can't stop crying!",
                "This is the saddest thing I've ever experienced!",
                "I'm consumed by this deep sadness!",
                "This grief is unbearable!",
                "I'm drowning in sorrow!",
                "This brings me so much sadness!"
            ],
            'surprise': [
                "I'm completely surprised by this unexpected turn!",
                "This is so surprising I can't believe it!",
                "I'm shocked by this development!",
                "This is absolutely shocking and unexpected!",
                "I'm so surprised I'm speechless!",
                "This surprise is overwhelming!",
                "I'm stunned by this news!",
                "This is the most surprising thing ever!",
                "I'm filled with amazement!",
                "This brings me so much surprise!"
            ],
            'disgust': [
                "This is absolutely disgusting and repulsive!",
                "I'm repelled by this completely!",
                "This makes me feel sick to my stomach!",
                "I can't stand this at all!",
                "This is absolutely revolting!",
                "I'm disgusted beyond words!",
                "This is the most disgusting thing ever!",
                "I'm repulsed by this!",
                "This fills me with disgust!",
                "This brings me so much disgust!"
            ],
            'shame': [
                "I'm so ashamed of myself right now!",
                "I feel completely embarrassed about this!",
                "I'm humiliated by this situation!",
                "This is so shameful I can't believe it!",
                "I'm mortified by this!",
                "This shame is overwhelming!",
                "I'm so embarrassed I want to hide!",
                "This is the most shameful thing ever!",
                "I'm filled with shame!",
                "This brings me so much shame!"
            ],
            'guilt': [
                "I feel so guilty about what I did!",
                "I'm consumed by this overwhelming guilt!",
                "I can't stop feeling guilty about this!",
                "This guilt is eating me alive!",
                "I'm drowning in guilt!",
                "I'm so guilty I can't sleep!",
                "This guilt is unbearable!",
                "I'm filled with remorse!",
                "This brings me so much guilt!",
                "I'm overwhelmed by this guilt!"
            ],
            'contempt': [
                "I have nothing but contempt for this!",
                "This is beneath me completely!",
                "I despise this with every fiber of my being!",
                "This is absolutely contemptible!",
                "I look down on this with disdain!",
                "This contempt is overwhelming!",
                "I'm filled with scorn!",
                "This is the most contemptible thing ever!",
                "I'm disgusted by this!",
                "This brings me so much contempt!"
            ],
            'neutral': [
                "This is okay, I guess.",
                "I don't have strong feelings about this either way.",
                "This is neither good nor bad to me.",
                "I'm pretty indifferent to this situation.",
                "This is just normal and ordinary.",
                "I don't feel strongly about this.",
                "This is average and unremarkable.",
                "I'm neutral about this.",
                "This doesn't affect me much.",
                "This is just standard and typical."
            ]
        }
        
        train_data = []
        val_data = []
        
        # Create realistic data for each emotion
        for emotion, templates in semeval_templates.items():
            if emotion in self.emotion_mapping['semeval']:
                label_idx = self.emotion_mapping['semeval'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                # Create 20 variations per emotion (16 train + 4 val)
                for i in range(20):
                    base_text = random.choice(templates)
                    
                    # Add variations
                    variations = [
                        f"{base_text}",
                        f"{base_text} Really!",
                        f"{base_text} I mean it!",
                        f"{base_text} This is serious!",
                        f"{base_text} I can't believe it!",
                        f"{base_text} It's incredible!",
                        f"{base_text} I'm telling you!",
                        f"{base_text} Believe me!",
                        f"{base_text} I swear!",
                        f"{base_text} Honestly!"
                    ]
                    
                    text = random.choice(variations)
                    
                    if i < 16:  # 80% for training
                        train_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'semeval_realistic'
                        })
                    else:  # 20% for validation
                        val_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'semeval_realistic'
                        })
        
        logger.info(f"✅ Realistic SemEval created: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    def create_realistic_isear(self) -> Tuple[List[Dict], List[Dict]]:
        """Create realistic ISEAR-style data"""
        logger.info("🔄 Creating realistic ISEAR data...")
        
        # More realistic and diverse ISEAR-style data
        isear_templates = {
            'joy': [
                "I feel joyful and happy about this wonderful achievement!",
                "This success brings me incredible joy and satisfaction!",
                "I'm filled with happiness and pure joy!",
                "This moment of joy is absolutely precious to me!",
                "I'm experiencing pure joy and contentment!",
                "This joy is overwhelming and beautiful!",
                "I'm so joyful I can't contain my happiness!",
                "This brings me such deep joy!",
                "I'm overflowing with joy and gratitude!",
                "This joy fills my entire being!"
            ],
            'fear': [
                "I'm scared and afraid of what might happen next!",
                "This situation fills me with deep fear and anxiety!",
                "I'm terrified of the potential consequences!",
                "This fear is overwhelming and paralyzing!",
                "I'm so afraid I can't think clearly!",
                "This fear consumes me completely!",
                "I'm paralyzed with fear and dread!",
                "This is absolutely terrifying!",
                "I'm overwhelmed by this fear!",
                "This fear is unbearable!"
            ],
            'anger': [
                "I'm angry and frustrated with this unfair situation!",
                "This injustice makes me absolutely furious!",
                "I'm boiling with anger and rage!",
                "This situation is completely infuriating!",
                "I'm consumed by this overwhelming anger!",
                "This anger is burning inside me!",
                "I'm so angry I can't control myself!",
                "This makes me absolutely livid!",
                "I'm filled with rage and fury!",
                "This anger is consuming me!"
            ],
            'sadness': [
                "I feel sad and depressed about this terrible loss!",
                "This sadness is overwhelming and crushing!",
                "I'm drowning in sorrow and grief!",
                "This grief is unbearable and painful!",
                "I'm consumed by this deep sadness!",
                "This sadness fills my entire being!",
                "I'm so sad I can't stop crying!",
                "This brings me such profound sadness!",
                "I'm overwhelmed by this sorrow!",
                "This sadness is all-consuming!"
            ],
            'disgust': [
                "This is absolutely disgusting and repulsive to me!",
                "I'm completely repelled by this!",
                "This makes me feel physically sick!",
                "I can't stand this at all!",
                "This is absolutely revolting!",
                "I'm disgusted beyond words!",
                "This fills me with complete disgust!",
                "I'm repulsed by this completely!",
                "This disgust is overwhelming!",
                "This brings me such deep disgust!"
            ],
            'shame': [
                "I feel deeply ashamed of my behavior!",
                "I'm completely embarrassed about this!",
                "I'm humiliated by this situation!",
                "This shame is overwhelming and crushing!",
                "I'm so ashamed I want to disappear!",
                "This brings me such deep shame!",
                "I'm mortified by this completely!",
                "This shame consumes me!",
                "I'm filled with embarrassment!",
                "This shame is unbearable!"
            ],
            'guilt': [
                "I feel so guilty about what I did wrong!",
                "I'm consumed by this overwhelming guilt!",
                "I can't stop feeling guilty about this!",
                "This guilt is eating me alive!",
                "I'm drowning in guilt and remorse!",
                "I'm so guilty I can't sleep!",
                "This guilt is unbearable and painful!",
                "I'm filled with deep remorse!",
                "This brings me such profound guilt!",
                "I'm overwhelmed by this guilt!"
            ]
        }
        
        train_data = []
        val_data = []
        
        # Create realistic data for each emotion
        for emotion, templates in isear_templates.items():
            if emotion in self.emotion_mapping['isear']:
                label_idx = self.emotion_mapping['isear'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                # Create 30 variations per emotion (24 train + 6 val)
                for i in range(30):
                    base_text = random.choice(templates)
                    
                    # Add variations
                    variations = [
                        f"{base_text}",
                        f"{base_text} It's true!",
                        f"{base_text} I really mean it!",
                        f"{base_text} This is how I feel!",
                        f"{base_text} I can't help it!",
                        f"{base_text} It's overwhelming!",
                        f"{base_text} I'm being honest!",
                        f"{base_text} This is real!",
                        f"{base_text} I'm serious!",
                        f"{base_text} Believe me!"
                    ]
                    
                    text = random.choice(variations)
                    
                    if i < 24:  # 80% for training
                        train_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'isear_realistic'
                        })
                    else:  # 20% for validation
                        val_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'isear_realistic'
                        })
        
        logger.info(f"✅ Realistic ISEAR created: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    def create_realistic_meld(self) -> Tuple[List[Dict], List[Dict]]:
        """Create realistic MELD-style data"""
        logger.info("🔄 Creating realistic MELD data...")
        
        # More realistic and diverse MELD-style data
        meld_templates = {
            'joy': [
                "I'm so happy and joyful about this amazing news!",
                "This brings me incredible joy and excitement!",
                "I'm overjoyed and thrilled about this!",
                "This is the happiest moment of my life!",
                "I'm filled with pure joy and happiness!",
                "This joy is overwhelming and wonderful!",
                "I'm so happy I could burst!",
                "This brings me such deep joy!",
                "I'm overflowing with joy!",
                "This joy fills my heart completely!"
            ],
            'sadness': [
                "I feel so sad and disappointed about this outcome!",
                "This sadness is overwhelming and crushing!",
                "I'm heartbroken by this terrible news!",
                "This is so sad it breaks my heart!",
                "I'm consumed by this deep sadness!",
                "This sadness fills my entire being!",
                "I'm so sad I can't stop crying!",
                "This brings me such profound sadness!",
                "I'm overwhelmed by this sorrow!",
                "This sadness is all-consuming!"
            ],
            'anger': [
                "I'm angry and frustrated with this situation!",
                "This makes me absolutely furious!",
                "I'm boiling with anger and rage!",
                "This is completely infuriating!",
                "I'm consumed by this overwhelming anger!",
                "This anger is burning inside me!",
                "I'm so angry I can't control myself!",
                "This makes me absolutely livid!",
                "I'm filled with rage and fury!",
                "This anger is consuming me!"
            ],
            'fear': [
                "I'm scared and afraid of what might happen!",
                "This situation terrifies me completely!",
                "I'm paralyzed with fear and dread!",
                "This is absolutely frightening!",
                "I'm overwhelmed by this fear!",
                "This fear is unbearable and crushing!",
                "I'm so afraid I can't think!",
                "This brings me such deep fear!",
                "I'm consumed by this terror!",
                "This fear fills my entire being!"
            ],
            'surprise': [
                "I'm surprised by this unexpected turn of events!",
                "This is so surprising I can't believe it!",
                "I'm shocked by this development!",
                "This is absolutely shocking and unexpected!",
                "I'm so surprised I'm speechless!",
                "This surprise is overwhelming!",
                "I'm stunned by this news!",
                "This brings me such amazement!",
                "I'm filled with wonder!",
                "This surprise is incredible!"
            ],
            'disgust': [
                "This is absolutely disgusting and repulsive!",
                "I'm completely repelled by this!",
                "This makes me feel physically sick!",
                "I can't stand this at all!",
                "This is absolutely revolting!",
                "I'm disgusted beyond words!",
                "This fills me with complete disgust!",
                "I'm repulsed by this completely!",
                "This disgust is overwhelming!",
                "This brings me such deep disgust!"
            ],
            'neutral': [
                "This is okay, I guess.",
                "I don't have strong feelings about this either way.",
                "This is neither good nor bad to me.",
                "I'm pretty indifferent to this situation.",
                "This is just normal and ordinary.",
                "I don't feel strongly about this.",
                "This is average and unremarkable.",
                "I'm neutral about this.",
                "This doesn't affect me much.",
                "This is just standard and typical."
            ]
        }
        
        train_data = []
        val_data = []
        
        # Create realistic data for each emotion
        for emotion, templates in meld_templates.items():
            if emotion in self.emotion_mapping['meld']:
                label_idx = self.emotion_mapping['meld'][emotion]
                labels = [0] * 28
                labels[label_idx] = 1
                
                # Create 40 variations per emotion (32 train + 8 val)
                for i in range(40):
                    base_text = random.choice(templates)
                    
                    # Add variations
                    variations = [
                        f"{base_text}",
                        f"{base_text} Really!",
                        f"{base_text} I mean it!",
                        f"{base_text} This is serious!",
                        f"{base_text} I can't believe it!",
                        f"{base_text} It's incredible!",
                        f"{base_text} I'm telling you!",
                        f"{base_text} Believe me!",
                        f"{base_text} I swear!",
                        f"{base_text} Honestly!",
                        f"{base_text} It's true!",
                        f"{base_text} I really mean it!",
                        f"{base_text} This is how I feel!",
                        f"{base_text} I can't help it!",
                        f"{base_text} It's overwhelming!"
                    ]
                    
                    text = random.choice(variations)
                    
                    if i < 32:  # 80% for training
                        train_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'meld_realistic'
                        })
                    else:  # 20% for validation
                        val_data.append({
                            'text': text,
                            'labels': labels,
                            'dataset': 'meld_realistic'
                        })
        
        logger.info(f"✅ Realistic MELD created: {len(train_data)} train, {len(val_data)} val")
        return train_data, val_data
    
    def combine_datasets(self) -> Tuple[List[Dict], List[Dict]]:
        """Combine all datasets into unified format"""
        logger.info("🔄 Combining all datasets...")
        
        # Load all datasets
        goemotions_train, goemotions_val = self.load_goemotions()
        semeval_train, semeval_val = self.create_realistic_semeval()
        isear_train, isear_val = self.create_realistic_isear()
        meld_train, meld_val = self.create_realistic_meld()
        
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
            'datasets_included': ['goemotions', 'semeval_realistic', 'isear_realistic', 'meld_realistic'],
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
        logger.info("🚀 Starting Realistic Multi-Dataset Preparation Pipeline")
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
    preparator = RealisticMultiDatasetPreparator()
    preparator.run()

if __name__ == "__main__":
    main()