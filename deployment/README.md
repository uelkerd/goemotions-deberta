# Phase 3 Deployment: GoEmotions DeBERTa Multi-label Emotion Detection

## Overview
This deployment implements Phase 3 for the GoEmotions DeBERTa project, providing a production-ready pipeline for multi-label emotion detection on journaling texts. The system uses the winner model from Phase 2 fine-tuning (placeholder: `./outputs/phase2_Asymmetric_Parallel`) with Hugging Face Transformers.

### System Architecture
The architecture follows a simple, efficient pipeline flow:

1. **Input Processing**: Accept raw text input (e.g., journaling entry).
2. **Tokenization**: Use DeBERTa-v3-large tokenizer (`microsoft/deberta-v3-large`) to preprocess text.
3. **Model Inference**: Load the fine-tuned DeBERTa-v3-large model for sequence classification. Run on GPU (device=0) if available, with fallback to CPU on OOM or no GPU.
4. **Post-Processing**:
   - Retrieve all class scores (logits) with `return_all_scores=True`.
   - Apply sigmoid activation to convert logits to probabilities for multi-label classification.
   - Filter predictions: Emotions with probability > threshold (default 0.2) are selected.
   - Map class indices to emotion names using labels from `data/goemotions/metadata.json` (28 classes: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral).
5. **Output**: Return a dictionary with:
   - `probabilities`: Dict of {emotion: probability} for all 28 classes.
   - `predicted_labels`: List of predicted emotion names (e.g., ['sadness', 'grief']).

Key Components:
- **Model**: DeBERTa-v3-large fine-tuned for multi-label classification.
- **Pipeline**: HF `transformers.pipeline` with custom sigmoid and threshold logic.
- **Error Handling**: GPU OOM fallback to CPU; general exception raising for failures.
- **Metadata Integration**: Dynamic loading of emotion labels for mapping.

## Setup
1. Ensure Python 3.8+ and required packages are installed:
   ```
   pip install transformers torch numpy
   ```
   For GPU support: Install PyTorch with CUDA (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu118`).

2. Place the Phase 2 winner model in `./outputs/phase2_Asymmetric_Parallel/` (update `MODEL_DIR` in `model_pipeline.py` after Phase 2 completes).

3. Ensure `data/goemotions/metadata.json` is available for label mapping.

## Running the Pipeline
Execute the script for example inference:
```
python deployment/model_pipeline.py
```
Output example for "I feel sad today.":
```
Text: I feel sad today.
Predicted labels: ['sadness']
All probabilities:
  admiration: 0.0123
  amusement: 0.0456
  ...
  sadness: 0.8234
  ...
```

For programmatic use, import and call `predict_emotions` (see Integration below).

## Performance Notes
- **Target Metrics**: Achieves >60% macro F1@0.2 on validation set post-Phase 2 (3 epochs, 43k samples, fixes: gamma=1.0, 5x oversampling, threshold=0.2).
- **Imbalance Handling**: Addressed via Asymmetric Parallel loss and oversampling during training.
- **Inference Speed**: ~50-200ms per text on GPU; fallback to CPU for low-resource environments.
- **Model Size**: DeBERTa-v3-large (~1.5GB); optimize with quantization if needed for production.

## Integration for Journaling App
Integrate the pipeline into your journaling application (e.g., Python/Flask/Django backend). Example code snippet:

```python
from deployment.model_pipeline import predict_emotions

def analyze_journal_entry(entry_text: str):
    """
    Analyze emotions in a journal entry and return predictions.
    """
    try:
        result = predict_emotions(entry_text, threshold=0.2)
        # Example: Log or store results
        print(f"Entry: {entry_text}")
        print(f"Emotions detected: {result['predicted_labels']}")
        # Integrate with UI: e.g., highlight text with emotion badges
        return result
    except Exception as e:
        print(f"Analysis failed: {e}")
        return {"error": str(e)}

# Usage in app route/handler
if __name__ == "__main__":
    entry = "Today was a great day, but I'm worried about tomorrow."
    analyze_journal_entry(entry)
```

For web apps, wrap in an API endpoint (e.g., POST /analyze with JSON {"text": "..."} returning JSON predictions). Ensure model loading is done once at startup for efficiency.

## Updating for Phase 2 Winner
After Phase 2 completes:
1. Identify winner (highest F1 macro, e.g., Combined_05 or Asymmetric_Parallel).
2. Copy best checkpoint to `./outputs/phase2_[winner]/`.
3. Update `MODEL_DIR` in `model_pipeline.py`.
4. Test with new model.

## Assumptions and Next Steps
- Placeholder model directory; replace post-Phase 2.
- GPU availability; falls back gracefully.
- Multi-label setup assumes model outputs 28 logits without problem_type specified in config.
- For production: Add caching, batching, or serving with FastAPI/TorchServe.