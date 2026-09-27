# EEG-Based Parkinson's Disease Classification

Cross-dataset EEG classification of Parkinson's disease using Transformer and CNN-Transformer architectures.

## Results

- **90.83%** balanced accuracy on San Diego dataset (Transformer, delta band)
- **87.50%** cross-dataset transfer accuracy (Iowa to San Diego, Transformer, all-band)
- **84.17%** within-dataset accuracy (CNN-Transformer, San Diego, delta+theta)

## Project Structure

```
preprocessing/       Preprocessing pipeline (channel selection, filtering, ICA, normalization)
training/            Model training scripts
  train_transformer.py       Standalone Transformer (channel-as-token)
  train_cnn_transformer.py   Hybrid CNN-Transformer
  train_baselines.py         SVM, Random Forest, Logistic Regression, MLP, 1D CNN
```

## Datasets

Two public resting-state EEG datasets from OpenNeuro:

- **University of Iowa** — 149 subjects (100 PD, 49 HC), 64-channel BioSemi
- **UC San Diego** — 31 subjects (15 PD, 16 HC), 128-channel BioSemi

## Methods

- 29 common channels (10-05 montage), bandpass filtered 1-45 Hz, 250 Hz sampling rate
- ICA artifact rejection (ICLabel, 90% confidence)
- 10 frequency band configurations (delta, theta, alpha, beta, gamma, and combinations)
- 5-fold stratified group cross-validation (subject-level)
- Cross-dataset evaluation: train on all subjects from one dataset, test on the other

## Requirements

- Python 3.8+
- PyTorch
- MNE-Python
- scikit-learn
- NumPy, SciPy, Matplotlib
