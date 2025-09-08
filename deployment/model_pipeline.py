#!/usr/bin/env python3
"""
Phase 3 Deployment: GoEmotions DeBERTa Multi-label Emotion Detection Pipeline
Loads the best Phase 2 model and provides predict_emotions function for inference.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

# Load emotion labels from metadata
METADATA_PATH = "data/goemotions/metadata.json"
EMOTIONS = []

def load_metadata():
    global EMOTIONS
    with open(METADATA_PATH, 'r') as f:
        data = json.load(f)
        EMOTIONS = data['emotions']

load_metadata()

# Placeholder for Phase 2 winner model directory
MODEL_DIR = "./outputs/phase2_Asymmetric_Parallel"

def predict_emotions(text: str, threshold: float = 0.2) -> dict:
    """
    Predict emotions for input text using the DeBERTa model pipeline.
    
    Args:
        text (str): Input text for emotion detection.
        threshold (float): Probability threshold for labeling emotions.
    
    Returns:
        dict: Contains 'probabilities' (dict of emotion: prob), 'predicted_labels' (list of str).
    
    Raises:
        Exception: If model loading or inference fails.
    """
    try:
        # Try GPU, fallback to CPU
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "text-classification",
            model=MODEL_DIR,
            tokenizer="microsoft/deberta-v3-large",
            device=device,
            return_all_scores=True
        )
        
        # Run inference
        results = pipe(text)
        
        # For multi-label, apply sigmoid to logits and threshold
        logits = np.array([r['score'] for r in results])
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        predicted_indices = np.where(probs > threshold)[0]
        predicted_labels = [EMOTIONS[i] for i in predicted_indices]
        probabilities = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        
        return {
            'probabilities': probabilities,
            'predicted_labels': predicted_labels
        }
    except torch.cuda.OutOfMemoryError:
        # Fallback to CPU
        pipe = pipeline(
            "text-classification",
            model=MODEL_DIR,
            tokenizer="microsoft/deberta-v3-large",
            device=-1,
            return_all_scores=True
        )
        results = pipe(text)
        logits = np.array([r['score'] for r in results])
        probs = 1 / (1 + np.exp(-logits))
        predicted_indices = np.where(probs > threshold)[0]
        predicted_labels = [EMOTIONS[i] for i in predicted_indices]
        probabilities = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        
        return {
            'probabilities': probabilities,
            'predicted_labels': predicted_labels
        }
    except Exception as e:
        raise Exception(f"Inference failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    sample_text = "I feel sad today."
    try:
        result = predict_emotions(sample_text)
        print(f"Text: {sample_text}")
        print(f"Predicted labels: {result['predicted_labels']}")
        print("All probabilities:")
        for emotion, prob in result['probabilities'].items():
            print(f"  {emotion}: {prob:.4f}")
    except Exception as e:
        print(f"Error: {e}")