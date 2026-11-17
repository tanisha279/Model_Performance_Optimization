# Model Performance Optimization

This repository demonstrates techniques to optimize deep learning models for deployment by reducing model size and latency while maintaining accuracy.

## Techniques Explored
1. **Model Pruning:** Removing redundant weights to reduce parameter count and model size using the TensorFlow Model Optimization Toolkit.
2. **Post-Training Quantization:** Converting model weights and operations to reduced precision (TFLite) for efficient edge and mobile inference.

## Project Structure
```text
Model_Performance_Optimization/
│
├── models/
│   ├── original_model.keras       # Baseline neural network model
│   ├── pruned_model.keras         # Pruned neural network model
│   └── quantized_model.tflite     # Quantized TFLite model
│
├── results/                       # Evaluation outputs and logs
├── accuracy_comparison.png        # Accuracy benchmark chart
├── model_size_comparison.png      # File size reduction chart
├── train.py                       # Trains baseline model on Iris dataset
├── pruning.py                     # Applies weight pruning
├── quantization.py                # Performs post-training quantization
├── evaluate.py                    # Benchmarks accuracy, latency, and sizes
└── graphs.py                      # Generates visual comparisons
```

## Results Summary
- **Baseline Model:** 26.20 KB (~73.3% accuracy)
- **Pruned Model:** 12.99 KB (~50.4% size reduction, ~76.7% accuracy)
- **Quantized Model:** 3.01 KB (~88.5% size reduction)

## Getting Started

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install tensorflow tensorflow-model-optimization scikit-learn matplotlib
```

### 2. Run Pipeline
```bash
# Train baseline model
python train.py

# Prune the model
python pruning.py

# Quantize the model
python quantization.py

# Evaluate models
python evaluate.py

# Generate comparison charts
python graphs.py
```
