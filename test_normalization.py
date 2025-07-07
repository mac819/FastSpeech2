#!/usr/bin/env python3
"""
Test script to demonstrate mel-spectrogram normalization approaches
"""

import torch
import numpy as np

def test_normalization():
    """Test different normalization approaches"""
    
    # Simulate mel-spectrogram values (typical range)
    # Mel-spectrograms can have values from very small (0.001) to large (1000+)
    test_values = torch.tensor([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    
    print("Original mel-spectrogram values:")
    print(test_values)
    print()
    
    # Test log1p approach
    log1p_result = torch.log1p(test_values)
    print("log1p results:")
    print(log1p_result)
    print()
    
    # Test log approach with epsilon
    epsilon = 1e-9
    log_result = torch.log(test_values + epsilon)
    print("log results (with epsilon):")
    print(log_result)
    print()
    
    # Show the compression effect
    print("Compression comparison:")
    print(f"Original range: {test_values.min():.6f} to {test_values.max():.6f}")
    print(f"log1p range: {log1p_result.min():.6f} to {log1p_result.max():.6f}")
    print(f"log range: {log_result.min():.6f} to {log_result.max():.6f}")
    print()
    
    # Show why log is better for mel-spectrograms
    print("Why log is better than log1p for mel-spectrograms:")
    print("1. log1p compresses small values less effectively")
    print("2. log provides better dynamic range compression")
    print("3. log is the standard in speech processing literature")
    print("4. log1p is better for values close to 0, but mel-spectrograms are rarely 0")

if __name__ == "__main__":
    test_normalization() 