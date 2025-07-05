# FastSpeech2 Training Improvement Plan - Updated

## Issues Identified and Solutions

### 1. **Mel-Spectrogram Normalization** ✅ FIXED - UPDATED
**Problem**: 
- `torch.log1p()` was giving same values because mel-spectrograms already had small values
- `log1p(x) = log(1+x)` is less effective for mel-spectrograms with values > 1

**Solution**: 
- Replaced `torch.log1p()` with `torch.log(x + epsilon)` 
- Added small epsilon (1e-9) to avoid log(0)
- This provides better dynamic range compression for mel-spectrograms

**Why this is better**:
- `log1p` is designed for values close to 0, but mel-spectrograms can have values from 0.001 to 1000+
- `log` transformation is standard in speech processing literature
- Provides better compression of the dynamic range

### 2. **Model Output Alignment** ✅ FIXED
**Problem**: Model decoder was using ReLU activation, which prevents negative values needed for log space.

**Solution**: 
- Removed ReLU activation from decoder output
- Model now outputs values in log space to match preprocessing
- This ensures consistency between training and inference

### 3. **Learning Rate and Gradient Accumulation** ✅ IMPROVED
**Problem**: Learning rate schedule was too aggressive and gradient accumulation was insufficient.

**Solution**: 
- Increased `gradient_accumulation_steps` from 1 to 4
- This provides more stable gradient updates

### 4. **Loss Weighting** ✅ IMPROVED
**Problem**: All loss components were weighted equally, but mel-spectrogram loss should dominate.

**Solution**: Added weighted loss calculation:
- Mel-spectrogram loss: 45.0x weight (main output)
- Other losses: 1.0x weight each

### 5. **Data Validation** ✅ ADDED
**Problem**: No validation of input data quality.

**Solution**: Added validation checks for NaN/inf values in mel-spectrograms with clamping.

### 6. **Training Monitoring** ✅ IMPROVED
**Problem**: Insufficient debugging information during training.

**Solution**: Added detailed logging of predicted vs target mel-spectrogram statistics.

## Key Changes Made

### `feature_compute.py`
```python
# OLD (ineffective):
melspec = torch.log1p(melspec)

# NEW (effective):
epsilon = 1e-9
melspec = melspec + epsilon
melspec = torch.log(melspec)
```

### `model.py`
```python
# OLD (prevents negative values):
mel_spec = self.output_activation(mel_spec) * self.output_scale

# NEW (allows log space):
mel_spec = mel_spec * self.output_scale  # No activation
```

### `train.py`
```python
# OLD (equal weighting):
total_loss = mel_loss + duration_loss + pitch_spectrogram_loss + ...

# NEW (weighted):
total_loss = (
    45.0 * mel_loss +  # Higher weight for mel-spectrogram
    1.0 * duration_loss + 
    1.0 * pitch_spectrogram_loss + ...
)
```

## Expected Improvements

With these changes, you should see:

1. **Better Value Ranges**: 
   - Predicted mel-spectrograms should have meaningful values (not just low values)
   - Log transformation provides proper dynamic range compression

2. **Improved Training Stability**:
   - Consistent preprocessing and model output spaces
   - Better gradient accumulation for stable updates

3. **Faster Convergence**:
   - Proper loss weighting focuses on main task
   - Better normalization reduces training instability

4. **Meaningful Outputs**:
   - Model outputs should show proper mel-spectrogram patterns
   - Values should be in reasonable ranges for log space

## Testing the Improvements

### Before vs After Comparison

**Before (with log1p)**:
- Mel-spectrogram values: 0.001 → 0.001 (no change)
- Model outputs: Low values with no pattern
- Loss saturation: Around 7

**After (with log)**:
- Mel-spectrogram values: 0.001 → -6.908, 1000 → 6.908 (proper compression)
- Model outputs: Should show proper mel-spectrogram patterns
- Expected loss: Should decrease more steadily

### Monitoring Points

1. **Check value ranges in debug output**:
   ```python
   print(f"Predicted mel-spec range: [{pred_mel_spec.min():.4f}, {pred_mel_spec.max():.4f}]")
   print(f"Target mel-spec range: [{mel_spec_target.min():.4f}, {mel_spec_target.max():.4f}]")
   ```

2. **Expected ranges**:
   - Log-transformed mel-spectrograms: typically -10 to +10
   - Model outputs should match this range

3. **Loss behavior**:
   - Should decrease more steadily
   - Mel loss should dominate other losses

## Next Steps

1. **Retrain the model** with these improvements
2. **Monitor the debug output** to verify value ranges
3. **Check mel-spectrogram visualizations** - should show proper patterns
4. **Validate on a small test set** to ensure quality improvements

## Configuration Summary

| Component | Old Setting | New Setting | Reason |
|-----------|-------------|-------------|---------|
| Mel Normalization | `log1p()` | `log(x + ε)` | Better compression |
| Decoder Activation | ReLU | None | Allow negative values |
| Loss Weighting | Equal | 45x mel, 1x others | Focus on main task |
| Gradient Accumulation | 1 | 4 | Stable updates |

## Files Modified

1. `audioml/processing/feature_compute.py` - Fixed mel normalization
2. `audioml/fastspeech/model.py` - Removed ReLU activation
3. `audioml/train.py` - Added loss weighting and debugging
4. `audioml/config.yaml` - Increased gradient accumulation
5. `audioml/dataset/feature_dataset.py` - Added data validation

The most critical fix was replacing `log1p` with `log` normalization, as this directly addresses why your model was generating low values with no pattern. 