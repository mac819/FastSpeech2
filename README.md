# AudioML - FastSpeech2 Implementation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This repository contains an implementation of Microsoft's **FastSpeech 2: Fast and High-Quality End-to-End Text to Speech** with various improvements and optimizations for better training stability and performance.

## 🚀 Features

- **FastSpeech2 Implementation**: Complete end-to-end text-to-speech model
- **Improved Training Stability**: Fixed learning rate calculation and mel-spectrogram normalization
- **Enhanced Monitoring**: Comprehensive logging with mel-spectrogram debugging
- **Modular Architecture**: Clean separation of components (encoder, variance adaptor, decoder)
- **Data Processing Pipeline**: Complete preprocessing for LJSpeech dataset

## 📁 Project Organization

```
├── LICENSE                    <- Open-source license
├── Makefile                   <- Makefile with convenience commands
├── README.md                  <- This file
├── pyproject.toml            <- Project configuration and package metadata
├── requirements.txt           <- Python dependencies
├── config.yaml               <- Model and training configuration
│
├── data/                     <- Data directory
│   ├── external/             <- Data from third party sources
│   ├── interim/              <- Intermediate data (text transcripts)
│   ├── processed/            <- Processed features (mel-spectrograms, pitch, energy)
│   └── raw/                  <- Original LJSpeech dataset
│
├── docs/                     <- Documentation
│   ├── docs/
│   │   ├── getting-started.md
│   │   └── index.md
│   └── mkdocs.yml
│
├── models/                   <- Trained model checkpoints
├── notebooks/                <- Jupyter notebooks for exploration
│   ├── alignment_data_creation.ipynb
│   ├── preprocessing.ipynb
│   ├── train.ipynb
│   └── FastSpeech2.ipynb
│
├── references/               <- Reference materials and images
│   └── images/
│       ├── fastspeech2_model.png
│       └── fastspeech2.drawio.png
│
├── reports/                  <- Generated analysis and figures
│   └── figures/
│
└── audioml/                 <- Main source code
    ├── __init__.py
    ├── config.yaml           <- Model configuration
    ├── train.py              <- Training script with improvements
    ├── synthesize.py          <- Inference script
    │
    ├── dataset/              <- Data loading and processing
    │   └── feature_dataset.py
    │
    ├── fastspeech/           <- FastSpeech2 model implementation
    │   ├── __init__.py
    │   ├── model.py          <- Main model architecture
    │   ├── transformer.py    <- Transformer blocks
    │   ├── speech_feature_predictor.py  <- Duration, pitch, energy predictors
    │   └── utils.py          <- Utility functions
    │
    ├── processing/           <- Data preprocessing
    │   ├── __init__.py
    │   ├── feature_compute.py    <- Feature extraction
    │   ├── prepare_ljspeech_files.py
    │   └── text_speech_alignment.py
    │
    ├── text/                 <- Text processing
    │   ├── __init__.py
    │   ├── cleaners.py
    │   ├── cmudict.py
    │   ├── numbers.py
    │   └── symbols.py
    │
    └── runs/                 <- Training logs and outputs
        └── fastspeech2/
```

## 🏗️ Model Architecture

The FastSpeech2 implementation consists of three main components:

### 1. **Encoder**
- Text embedding with positional encoding
- Multi-layer transformer blocks
- Processes input text tokens

### 2. **Variance Adaptor**
- **Duration Predictor**: Predicts phoneme durations
- **Length Regulator**: Expands encoder outputs based on durations
- **Pitch Predictor**: Predicts pitch contours using CWT
- **Energy Predictor**: Predicts frame-level energy

### 3. **Decoder**
- Multi-layer transformer blocks
- Generates mel-spectrograms
- Outputs log-transformed mel-spectrograms

## 🚀 Quick Start

### **1. Setup Environment**
```bash
# Clone the repository
git clone <repository-url>
cd audio

# Install dependencies
pip install -r requirements.txt
```

### **2. Prepare Data**
```bash
# Download LJSpeech dataset to data/raw/
# Run preprocessing
python audioml/processing/prepare_ljspeech_files.py
python audioml/processing/feature_compute.py
```

### **3. Train Model**
```bash
# Start training
python audioml/train.py
```

### **4. Monitor Training**
- **Console**: Real-time loss and mel-spectrogram statistics
- **Log file**: `audioml/training_log.txt` with detailed metrics
- **TensorBoard**: `runs/fastspeech2/` for visualizations

## 📊 Training Configuration

### **Optimizer Settings**
```yaml
learning_rate: 0.0001          # Fixed from 0.0625 bug
warm_up_steps: 8000           # Better warmup
anneal_steps: [200000, 400000, 600000]  # Gradual decay
weight_decay: 0.001           # Reduced from 0.01
eps: 1e-8                     # Better numerical stability
```

### **Model Architecture**
```yaml
encoder:
  n_fft_layers: 4
  embedding_layer:
    dims: 256
  fft:
    d_in: 256
    d_out: 256
    n_head: 2

decoder:
  n_fft_layers: 4
  n_mels: 128
  fft:
    d_in: 384
    d_out: 384
```

## 📈 Expected Results

With the improvements, you should see:

1. **Stable Training**: No gradient explosions
2. **Better Convergence**: Loss decreases steadily
3. **Meaningful Outputs**: Mel-spectrograms with proper patterns
4. **Improved Quality**: Better speech synthesis results

## 📝 Logging and Monitoring

### **Training Log Format**
```csv
step,total_loss,mel_loss,duration_loss,pitch_spec_loss,pitch_mean_loss,pitch_std_loss,energy_loss,pred_mel_min,pred_mel_max,pred_mel_mean,target_mel_min,target_mel_max,target_mel_mean
100,6.2345,0.1234,0.5678,0.2345,0.3456,0.4567,0.6789,-8.1234,2.5678,-2.3456,-7.8901,3.4567,-1.2345
```

### **Expected Value Ranges**
- **Log-transformed mel-spectrograms**: -10 to +10
- **Predicted outputs**: Should match target ranges
- **Loss values**: Should decrease over time


### **Debug Commands**
```bash
# Check mel-spectrogram ranges
python test_normalization.py

# Monitor training logs
tail -f audioml/training_log.txt

# View TensorBoard
tensorboard --logdir runs/fastspeech2
```

## 📚 References

- **Original Paper**: [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- **Reference Implementation**: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- **Dataset**: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This implementation includes several improvements over the original FastSpeech2 to address common training issues and improve stability.