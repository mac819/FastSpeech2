# Optimizer Parameters Analysis for FastSpeech2

## **Issues Found in Current Configuration**

### **1. Learning Rate Calculation Problem** ❌ FIXED
**Problem**: 
```python
# OLD (problematic):
self.init_lr = np.power(config["encoder"]['fft']['d_out'], -0.5)
# This gives: 256^(-0.5) = 0.0625 (WAY TOO HIGH!)
```

**Solution**: 
```python
# NEW (fixed):
self.init_lr = train_config["optimizer"]["learning_rate"]  # 0.0001
```

**Why this matters**: 
- 0.0625 is 625x higher than the intended 0.0001
- This would cause training instability and poor convergence

### **2. Learning Rate Schedule Issues** ✅ IMPROVED

**Before**:
- `warm_up_steps: 4000` - Too short for 900k total steps
- `anneal_steps: [300000, 400000, 500000]` - Too late
- `anneal_rate: 0.3` - Too aggressive

**After**:
- `warm_up_steps: 8000` - Better warmup period
- `anneal_steps: [200000, 400000, 600000]` - Earlier and more gradual
- `anneal_rate: 0.5` - Less aggressive decay

### **3. Adam Optimizer Parameters** ✅ IMPROVED

| Parameter | Before | After | Reason |
|-----------|--------|-------|---------|
| `eps` | 1e-9 | 1e-8 | Better numerical stability |
| `weight_decay` | 0.01 | 0.001 | Less aggressive regularization |
| `betas` | [0.9, 0.98] | [0.9, 0.98] | Good for transformers |

## **Recommended Optimizer Settings for FastSpeech2**

### **Learning Rate Schedule**
```python
# Warmup phase (0-8000 steps): Linear increase
# Plateau phase (8000-200000 steps): Constant LR
# Annealing phase (200000+ steps): Gradual decrease
```

### **Expected Learning Rate Progression**
```
Step 0-8000:    0 → 0.0001 (warmup)
Step 8000-200k: 0.0001 (constant)
Step 200k-400k: 0.0001 → 0.00005 (first anneal)
Step 400k-600k: 0.00005 → 0.000025 (second anneal)
Step 600k+:     0.000025 → 0.0000125 (final anneal)
```

## **Why These Changes Help**

### **1. Proper Learning Rate**
- **0.0001** is standard for transformer-based models
- Allows stable training without gradient explosion
- Enables proper convergence

### **2. Better Warmup**
- **8000 steps** gives model time to stabilize
- Prevents early training instability
- Standard practice for transformer models

### **3. Gradual Annealing**
- **Earlier annealing** prevents overfitting
- **Less aggressive decay** maintains learning longer
- **Multiple steps** provide smooth transitions

### **4. Improved Numerical Stability**
- **eps=1e-8** prevents division by very small numbers
- **weight_decay=0.001** reduces overfitting without hurting performance

## **Expected Improvements**

With these optimizer changes, you should see:

1. **Stable Training**: No more gradient explosions
2. **Better Convergence**: Loss should decrease more steadily
3. **Improved Final Performance**: Better mel-spectrogram quality
4. **Faster Training**: More efficient learning rate schedule

## **Monitoring Points**

### **Check These During Training**:
1. **Learning rate values** in debug output
2. **Loss stability** - should decrease smoothly
3. **Gradient norms** - should stay reasonable
4. **Mel-spectrogram quality** - should improve over time

### **Expected Learning Rate Values**:
```
Step 1000:  ~0.0000125 (warmup)
Step 8000:  ~0.0001 (full LR)
Step 100k:  ~0.0001 (plateau)
Step 300k:  ~0.00005 (after first anneal)
Step 500k:  ~0.000025 (after second anneal)
```

## **Additional Recommendations**

### **For Even Better Training**:
1. **Consider smaller batch size** (8 instead of 16) for more stable gradients
2. **Add gradient clipping** at 0.5 instead of 1.0
3. **Use learning rate finder** to find optimal initial LR
4. **Implement early stopping** based on validation loss

### **Alternative Optimizer Settings**:
```yaml
# For more aggressive training:
learning_rate: 0.0002
warm_up_steps: 4000

# For more conservative training:
learning_rate: 0.00005
warm_up_steps: 12000
weight_decay: 0.0005
```

The most critical fix was correcting the learning rate calculation, which was causing your model to train with a learning rate 625x higher than intended! 