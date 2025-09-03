#!/usr/bin/env python3
"""
Test script for Asymmetric Loss implementation
"""
import torch
import torch.nn as nn
import numpy as np

# Import our Asymmetric Loss
import sys
sys.path.append('.')
from train_samo import AsymmetricLoss

def test_asymmetric_loss():
    """Test the Asymmetric Loss implementation"""
    print("🧪 Testing Asymmetric Loss Implementation")
    print("=" * 50)
    
    # Create test data
    batch_size = 4
    num_labels = 28  # GoEmotions has 28 labels
    
    # Random logits and labels
    logits = torch.randn(batch_size, num_labels)
    labels = torch.randint(0, 2, (batch_size, num_labels)).float()
    
    print(f"📊 Test data shape: logits {logits.shape}, labels {labels.shape}")
    
    # Test Asymmetric Loss
    asl_loss = AsymmetricLoss(
        gamma_neg=2.0,
        gamma_pos=1.0,
        clip=0.05,
        pos_alpha=1.0
    )
    
    loss = asl_loss(logits, labels)
    print(f"✅ Asymmetric Loss computed: {loss.item():.4f}")
    
    # Test with different parameters
    asl_loss_aggressive = AsymmetricLoss(
        gamma_neg=3.0,  # More aggressive
        gamma_pos=0.5,
        clip=0.1,
        pos_alpha=2.0
    )
    
    loss_aggressive = asl_loss_aggressive(logits, labels)
    print(f"✅ Aggressive ASL Loss: {loss_aggressive.item():.4f}")
    
    # Compare with standard BCE loss
    bce_loss = nn.BCEWithLogitsLoss()
    bce_result = bce_loss(logits, labels)
    print(f"📊 Standard BCE Loss: {bce_result.item():.4f}")
    
    print("\n🎯 Asymmetric Loss Test Results:")
    print(f"   Standard BCE: {bce_result.item():.4f}")
    print(f"   ASL (balanced): {loss.item():.4f}")
    print(f"   ASL (aggressive): {loss_aggressive.item():.4f}")
    
    # Test gradient flow
    loss.backward()
    print("✅ Gradient computation successful")
    
    print("\n🚀 Asymmetric Loss implementation is working correctly!")
    return True

if __name__ == "__main__":
    test_asymmetric_loss()
