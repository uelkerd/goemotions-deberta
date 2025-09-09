#!/usr/bin/env python3
"""
LINE-BY-LINE DEBUGGING: AsymmetricLoss gradient vanishing
Find exactly where gradients are being destroyed
"""

import torch
import torch.nn as nn
import sys
import os

# Add our script path
sys.path.append("/workspace/notebooks/scripts")

def debug_asymmetric_loss():
    print("🔬 ASYMMETRIC LOSS LINE-BY-LINE DEBUG")
    print("=" * 60)
    
    # Import current broken implementation
    from train_deberta_local import AsymmetricLoss
    
    # Test data (realistic for GoEmotions)
    batch_size, num_classes = 4, 28
    logits = torch.randn(batch_size, num_classes, requires_grad=True) 
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    print(f"📊 Input shapes: logits={logits.shape}, targets={targets.shape}")
    print(f"📊 Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"📊 Targets sum: {targets.sum().item():.0f} positives out of {targets.numel()}")
    
    # Create AsymmetricLoss with current parameters
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8)
    print(f"📊 ASL params: gamma_neg={asl.gamma_neg}, gamma_pos={asl.gamma_pos}, clip={asl.clip}")
    
    # MANUAL FORWARD PASS WITH DEBUGGING
    print("\n🔍 STEP-BY-STEP FORWARD PASS:")
    print("-" * 40)
    
    # Step 1: Sigmoid probabilities
    x_sigmoid = torch.sigmoid(logits)
    xs_pos = x_sigmoid  
    xs_neg = 1 - x_sigmoid
    
    print(f"Step 1 - Probabilities:")
    print(f"  xs_pos range: [{xs_pos.min().item():.6f}, {xs_pos.max().item():.6f}]")
    print(f"  xs_neg range: [{xs_neg.min().item():.6f}, {xs_neg.max().item():.6f}]")
    
    # Step 2: Clipping (only negatives)
    if asl.clip > 0:
        xs_neg_clipped = (xs_neg + asl.clip).clamp(max=1)
        print(f"Step 2 - Clipping:")
        print(f"  xs_neg after clip: [{xs_neg_clipped.min().item():.6f}, {xs_neg_clipped.max().item():.6f}]")
        xs_neg = xs_neg_clipped
    
    # Step 3: Basic CE calculation
    los_pos = targets * torch.log(xs_pos + asl.eps)
    los_neg = (1 - targets) * torch.log(xs_neg + asl.eps)
    basic_loss = los_pos + los_neg
    
    print(f"Step 3 - Basic CE:")
    print(f"  los_pos range: [{los_pos.min().item():.6f}, {los_pos.max().item():.6f}]")
    print(f"  los_neg range: [{los_neg.min().item():.6f}, {los_neg.max().item():.6f}]")
    print(f"  basic_loss range: [{basic_loss.min().item():.6f}, {basic_loss.max().item():.6f}]")
    
    # Step 4: Asymmetric Focusing (THE SUSPECT!)
    if asl.gamma_neg > 0 or asl.gamma_pos > 0:
        print(f"Step 4 - Asymmetric Focusing (SUSPECT!):")
        
        # Calculate pt values
        pt0 = xs_pos * targets  
        pt1 = xs_neg * (1 - targets)
        pt = pt0 + pt1
        
        print(f"  pt0 range: [{pt0.min().item():.6f}, {pt0.max().item():.6f}]")
        print(f"  pt1 range: [{pt1.min().item():.6f}, {pt1.max().item():.6f}]")
        print(f"  pt range: [{pt.min().item():.6f}, {pt.max().item():.6f}]")
        
        # Calculate gamma values per element
        one_sided_gamma = asl.gamma_pos * targets + asl.gamma_neg * (1 - targets)
        print(f"  one_sided_gamma range: [{one_sided_gamma.min().item():.6f}, {one_sided_gamma.max().item():.6f}]")
        
        # The critical operation: torch.pow(1 - pt, gamma)
        one_minus_pt = 1 - pt
        print(f"  (1-pt) range: [{one_minus_pt.min().item():.6f}, {one_minus_pt.max().item():.6f}]")
        
        # This is the SUSPECTED CULPRIT!
        one_sided_w = torch.pow(one_minus_pt, one_sided_gamma)
        print(f"  ⚠️  FOCUSING WEIGHTS range: [{one_sided_w.min().item():.2e}, {one_sided_w.max().item():.2e}]")
        
        # Check for problematic values
        if one_sided_w.min() < 1e-10:
            print(f"  🚨 EXTREMELY SMALL WEIGHTS DETECTED!")
            print(f"  🚨 Min weight: {one_sided_w.min().item():.2e}")
            print(f"  🚨 This will KILL gradients!")
            
        # Apply focusing
        focused_loss = basic_loss * one_sided_w
        print(f"  focused_loss range: [{focused_loss.min().item():.6f}, {focused_loss.max().item():.6f}]")
        
        final_loss = -focused_loss.mean()
    else:
        final_loss = -basic_loss.mean()
    
    print(f"Step 5 - Final loss: {final_loss.item():.6f}")
    
    # GRADIENT TEST
    print("\n🧪 GRADIENT FLOW TEST:")
    print("-" * 30)
    
    # Test gradients
    final_loss.backward()
    grad_norm = torch.norm(logits.grad).item()
    
    print(f"📊 Gradient norm: {grad_norm:.2e}")
    print(f"📊 Gradient range: [{logits.grad.min().item():.2e}, {logits.grad.max().item():.2e}]")
    
    if grad_norm < 1e-3:
        print("🚨 GRADIENT VANISHING CONFIRMED!")
        print("🔍 Investigating numerical issues...")
        
        # Check for NaN or Inf
        if torch.isnan(logits.grad).any():
            print("❌ NaN gradients detected!")
        if torch.isinf(logits.grad).any():
            print("❌ Inf gradients detected!")
            
        # Check focusing weights more carefully
        logits.grad.zero_()
        asl_debug = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        
        # Recalculate to inspect weights
        with torch.no_grad():
            x_sig = torch.sigmoid(logits)
            pt_calc = x_sig * targets + (1 - x_sig) * (1 - targets) 
            gamma_calc = asl_debug.gamma_neg * (1 - targets)  # Only negatives get gamma_neg=4.0
            weights = torch.pow(1 - pt_calc, gamma_calc)
            
            print(f"🔍 Detailed weight analysis:")
            print(f"  Weight percentiles: [1%={weights.quantile(0.01):.2e}, 50%={weights.quantile(0.5):.2e}, 99%={weights.quantile(0.99):.2e}]")
            print(f"  Weights < 1e-10: {(weights < 1e-10).sum().item()}/{weights.numel()}")
            print(f"  Weights < 1e-6: {(weights < 1e-6).sum().item()}/{weights.numel()}")
            
    else:
        print("✅ Gradients are healthy!")
    
    return grad_norm

def test_conservative_asl():
    print("\n🔧 TESTING CONSERVATIVE ASYMMETRIC LOSS:")
    print("-" * 40)
    
    # Test with more conservative parameters
    conservative_asl = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8)
    
    logits = torch.randn(4, 28, requires_grad=True)
    targets = torch.randint(0, 2, (4, 28)).float()
    
    loss = conservative_asl(logits, targets)
    loss.backward()
    grad_norm = torch.norm(logits.grad).item()
    
    print(f"📊 Conservative ASL (gamma_neg=2.0):")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.2e}")
    
    if grad_norm > 1e-3:
        print("✅ Conservative parameters work!")
        return True
    else:
        print("❌ Still broken even with conservative parameters")
        return False

def test_reference_implementation():
    print("\n📚 TESTING REFERENCE IMPLEMENTATION:")
    print("-" * 40)
    
    # Simple reference implementation from literature
    class ReferenceASL(nn.Module):
        def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05):
            super().__init__()
            self.gamma_neg = gamma_neg
            self.gamma_pos = gamma_pos
            self.clip = clip

        def forward(self, inputs, targets):
            # Sigmoid probabilities
            xs_pos = torch.sigmoid(inputs)
            xs_neg = 1 - xs_pos

            # Asymmetric clipping (only negatives)
            if self.clip is not None and self.clip > 0:
                xs_neg = xs_neg + self.clip
                xs_neg = torch.clamp(xs_neg, 0, 1)

            # Basic cross entropy
            los_pos = targets * torch.log(xs_pos)
            los_neg = (1 - targets) * torch.log(xs_neg)
            loss = los_pos + los_neg

            # Asymmetric focusing
            if self.gamma_neg > 0 or self.gamma_pos > 0:
                pt = xs_pos * targets + xs_neg * (1 - targets)
                one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
                loss = loss * one_sided_w

            return -loss.sum()  # Note: sum() instead of mean()

    ref_asl = ReferenceASL(gamma_neg=4, gamma_pos=0, clip=0.05)
    
    logits = torch.randn(4, 28, requires_grad=True)
    targets = torch.randint(0, 2, (4, 28)).float()
    
    loss = ref_asl(logits, targets)
    loss.backward()
    grad_norm = torch.norm(logits.grad).item()
    
    print(f"📊 Reference implementation:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {grad_norm:.2e}")
    
    if grad_norm > 1e-3:
        print("✅ Reference implementation works!")
        return True
    else:
        print("❌ Reference implementation also broken")
        return False

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE ASYMMETRIC LOSS DEBUG")
    print("=" * 70)
    
    # Test 1: Debug current implementation
    print("\n1️⃣ DEBUGGING CURRENT IMPLEMENTATION:")
    current_grad = debug_asymmetric_loss()
    
    # Test 2: Try conservative parameters  
    print("\n2️⃣ TESTING CONSERVATIVE PARAMETERS:")
    conservative_works = test_conservative_asl()
    
    # Test 3: Reference implementation
    print("\n3️⃣ TESTING REFERENCE IMPLEMENTATION:")
    reference_works = test_reference_implementation()
    
    # DIAGNOSIS
    print("\n🎯 DIAGNOSTIC SUMMARY:")
    print("=" * 40)
    print(f"Current implementation: {current_grad:.2e} grad norm ({'BROKEN' if current_grad < 1e-3 else 'OK'})")
    print(f"Conservative params: {'WORKS' if conservative_works else 'BROKEN'}")
    print(f"Reference implementation: {'WORKS' if reference_works else 'BROKEN'}")
    
    if conservative_works:
        print("\n✅ SOLUTION: Use gamma_neg=2.0 instead of 4.0")
    elif reference_works:
        print("\n✅ SOLUTION: Replace with reference implementation")
    else:
        print("\n🚨 DEEPER PROBLEM: Even reference fails - fundamental issue")
        
    print("\n💡 RECOMMENDATION:")
    if conservative_works:
        print("- Change AsymmetricLoss defaults to gamma_neg=2.0")
        print("- Should achieve ~30-40% F1 instead of 7.96% disaster")
    elif reference_works:
        print("- Replace current implementation with reference")
        print("- Use sum() instead of mean() for loss reduction")
    else:
        print("- AsymmetricLoss may not be compatible with this setup")
        print("- Focus on BCE (44.71% proven) + Combined loss (now fixed)")