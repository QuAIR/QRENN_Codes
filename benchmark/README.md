# QML Benchmark Suite

A comprehensive benchmarking framework for **Quantum Machine Learning (QML)** models using multiple quantum computing frameworks. This project implements and benchmarks various quantum neural network and kernel methods including QRENN, QRNN, QSVM, ProjectQKM, and NeuralQKM.

## Overview

This benchmark suite provides:

- **Multiple QML Architectures**: QRENN, QRNN, QSVM, Projected Quantum Kernel, Neural Quantum Kernel
- **Dual Framework Support**: QuAIRKit (0.4.4+) and PennyLane implementations
- **Comprehensive Dataset Support**: Image, tabular, synthetic, text, and graph datasets
- **Standardized Benchmarking**: Unified evaluation across all models with statistical analysis
- **Visualization Tools**: Professional plotting and comparison utilities
- **Modular Design**: Easy-to-extend framework for quantum ML research

## Features

### Quantum Models
- **QRENN**: Quantum Recurrent Embedding Neural Networks with trainable and control qubits
- **QRNN**: Quantum Recurrent Neural Network using diagonal encoding and data re-uploading
- **QSVM**: Quantum Support Vector Machine with configurable quantum feature maps
- **ProjectQKM**: Projected Quantum Kernel Model with 1-RDM feature extraction
- **NeuralQKM**: Neural Quantum Kernel Model with data re-uploading QNN

### Data Encoding Strategies
- **Amplitude Encoding**: Encode data into quantum state amplitudes
- **Angle Encoding**: Sequential, parallel, or dense rotation-based encoding
- **Hamiltonian Encoding**: Encode via Hamiltonian exponentiation
- **Variational Encoding**: Learnable encoding combined with data
- **Involutory Encoding**: Householder reflection encoding (U² = I)
- **Feature Map Encoding**: Standard quantum feature maps (ZZ, tensor product, high-order)
- **Diagonal Encoding**: Diagonal unitary encoding for PennyLane QNNs

### Dataset Support
- **Image**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- **Tabular**: Iris, Wine, Breast Cancer, Ionosphere
- **Synthetic**: Moons, Circles, Blobs, XOR, S-curve
- **Text**: IMDb, Yelp
- **Graph**: Cora, Citeseer

### Training Infrastructure
- Multiple optimizers: Adam, AdamW, SGD
- Learning rate schedulers: Plateau, Step, Cosine Annealing
- Customizable loss and prediction functions
- Automatic checkpoint saving/loading
- Comprehensive metrics tracking and visualization

## Installation

### Requirements

- Python 3.8+
- PyTorch
- QuAIRKit (0.4.4+)
- PennyLane (for QRNN)
- scikit-learn
- NumPy
- Matplotlib
- (Optional) torchvision, torch_geometric, nltk, tensorflow_datasets for full dataset support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Dragon-John/QRENN_benchmark.git
cd QRENN_benchmark
```

2. Install core dependencies:
```bash
pip install torch quairkit pennylane scikit-learn numpy matplotlib
```

3. For full dataset support:
```bash
pip install torchvision scikit-learn torch_geometric nltk tensorflow_datasets
```

4. (Optional) Download NLTK data for text datasets:
```python
import nltk
nltk.download('movie_reviews')
```

## Quick Start

### QRENN (Quantum Recurrent Embedding Neural Network)

```python
import torch
from QRENN import QRENN
from load_data import load_data
from train import Trainer

# Load and prepare data
X_train, X_test, y_train, y_test, metadata = load_data(
    'mnist',
    use_pca=True,
    pca_dim=4,
    n_train=1000,
    n_test=200,
    random_state=42
)

# Initialize QRENN model
qrenn = QRENN(
    num_trainable=2,
    num_control=1,
    num_layers=3,
    use_conjugate=False
)

# Set data unitary
from utils import encode_ry_batch
data_unitaries = encode_ry_batch(X_train, num_qubits=2)
qrenn.set_data(data_unitaries[0])

# Create trainer
trainer = Trainer(
    learning_rate=0.1,
    iterations=2000,
    trainable_qubits=2,
    ancilla_qubits=7
)

# Train the model
trained_params = trainer.train(qrenn, labels=torch.tensor(y_train))

# Evaluate
results = trainer.evaluate(qrenn, labels=torch.tensor(y_test))
print(f"Accuracy: {results['accuracy']:.4f}")
```

### QRNN (Quantum Recurrent Neural Network - PennyLane)

```python
from QRNN import QuantumBinaryClassifier

# Initialize QRNN
qrnn = QuantumBinaryClassifier(
    n_qubits=2,
    n_data_layers=3,
    n_param_layers=4
)

# Train
weights, history, data = qrnn.train(
    dataset_name='iris',
    use_pca=True,
    n_epochs=100,
    learning_rate=0.1,
    n_train=80,
    n_test=20
)

print(f"Test accuracy: {qrnn.accuracy(weights, data[1], data[3]):.4f}")
```

### QSVM (Quantum Support Vector Machine)

```python
from QSVM import QSVM

# Initialize QSVM
qsvm = QSVM(n_qubits=2, depth=2, encoding_type='iqp')

# Train
X_train, X_test, y_train, y_test, _ = load_data('iris', use_pca=True, pca_dim=4, n_train=80, n_test=20)
qsvm.set_data(X_train, y_train, compute_kernel=True, verbose=True)
qsvm.fit()

# Predict
predictions = qsvm.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Running Benchmarks

```python
from qrenn_benchmark import run_encoding_benchmark
from plot_results import plot_comparison_chart

# Run benchmark
results = run_encoding_benchmark(
    dataset_name='iris',
    encoding_types=['diagonal', 'ry', 'angle', 'pauli', 'iqp'],
    n_runs=10
)

# Visualize results
plot_comparison_chart(results, save_path='benchmark_results.png')
```

## Project Structure

```
QRENN_benchmark/
├── Core Models
│   ├── QRENN.py                    # Quantum Recurrent Embedding Neural Networks (QuAIRKit)
│   ├── QRNN.py                     # Quantum Recurrent Neural Network (PennyLane)
│   ├── QSVM.py                     # Quantum Support Vector Machine (QuAIRKit)
│   ├── ProjectQKM.py                # Projected Quantum Kernel Model (QuAIRKit)
│   └── NeuralQKM.py               # Neural Quantum Kernel Model (QuAIRKit)
│
├── Benchmark Scripts
│   ├── qrenn_benchmark.py          # QRENN encoding and performance benchmarks
│   ├── qrnn_benchmark.py           # QRNN benchmarking
│   ├── qsvm_benchmark.py           # QSVM benchmarking
│   ├── pqkm_benchmark.py          # ProjectQKM benchmarking
│   └── nqkm_benchmark.py          # NeuralQKM benchmarking
│
├── Utilities
│   ├── train.py                    # Training infrastructure for QRENN
│   ├── load_data.py                # Data loading and preprocessing
│   ├── utils.py                   # Encoding functions and utility methods
│   └── plot_results.py            # Visualization and plotting utilities
│
├── Notebooks
│   ├── qkit_qrenn_MINST_learning.ipynb  # MNIST/Fashion-MNIST example notebook
│   ├── qrenn_test.ipynb                   # Test suite notebook
│   └── qsvm_ibm_zzfmap.ipynb              # Quantum SVM examples
│
└── Data Directories
    ├── classical_data/              # Classical dataset storage
    └── quantum_data/              # Quantum dataset storage
```

## Model Documentation

### QRENN.py (Quantum Recurrent Embedding Neural Networks)

The core QRENN model implementing quantum recurrent neural networks with alternating trainable and data-embedding layers.

**Key Parameters:**
- `num_trainable`: Number of trainable qubits with universal gates
- `num_control`: Number of control qubits for data embedding
- `num_layers`: Number of alternating layers
- `use_conjugate`: Whether to use alternating U and U† layers

**Key Methods:**
- `set_data(data_unitary)`: Set data unitary for embedding
- `forward()`: Forward pass through the circuit
- `get_unitary()`: Get the full circuit unitary

**Framework:** QuAIRKit

### QRNN.py (Quantum Recurrent Neural Network)

A PennyLane implementation of quantum recurrent neural networks using diagonal encoding and data re-uploading.

**Key Parameters:**
- `n_qubits`: Number of qubits (features = 2^n_qubits for diagonal encoding)
- `n_data_layers`: Number of data re-uploading iterations
- `n_param_layers`: Number of trainable parameter layers

**Key Features:**
- Diagonal encoding for efficient data upload
- Variational layers with RX-RY-RZ rotations
- Projection measurement loss function
- Support for multiple datasets with automatic PCA dimensionality adjustment

**Framework:** PennyLane

### QSVM.py (Quantum Support Vector Machine)

QuAIRKit implementation of QSVM with configurable quantum feature maps.

**Supported Encodings:**
- `iqp`: Instantaneous Quantum Polynomial encoding
- `angle`: Angle encoding via rotation gates (RY, RZ, RX)
- `pauli`: Mixed Pauli rotations
- `amplitude`: Diagonal/amplitude encoding

**Key Parameters:**
- `n_qubits`: Number of qubits (features)
- `depth`: Depth of feature map circuit
- `encoding_type`: Type of encoding method
- `C`: SVM regularization parameter

**Framework:** QuAIRKit + scikit-learn

### ProjectQKM.py (Projected Quantum Kernel Model)

Implementation of Projected Quantum Kernel Model based on TensorFlow Quantum tutorial.

**Key Components:**
- Single qubit rotation wall (X, Y, Z rotations)
- V(theta) entanglement with Heisenberg-type coupling
- Trotterized evolution for repeated application
- 1-RDM (Reduced Density Matrix) feature extraction

**Key Parameters:**
- `n_qubits`: Number of qubits
- `n_trotter`: Number of Trotter steps for V(theta) evolution
- `use_classical_data`: Whether to use classical data for theta values
- `random_range`: Range for random rotation angles

**Framework:** QuAIRKit

### NeuralQKM.py (Neural Quantum Kernel Model)

Implementation of Neural Quantum Kernel with data re-uploading QNN.

**Kernel Types:**
- `eqk_n_to_n`: Neural EQK using n-to-n approach
- `pqk`: Neural PQK using 1-RDM from trained QNN

**Key Components:**
- Data re-uploading QNN with trainable parameters
- Fidelity-based loss function for QNN training
- SVM classifier using quantum kernel matrix

**Key Parameters:**
- `n_qubits`: Number of qubits
- `n_layers`: Number of data re-uploading layers
- `kernel_type`: Type of quantum kernel
- `entanglement`: Type of entanglement ('cnot', 'cz')

**Framework:** QuAIRKit

### load_data.py

Comprehensive data loading and preparation module.

**Supported Datasets:**
- **Image**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- **Tabular**: Iris, Wine, Breast Cancer, Ionosphere
- **Synthetic**: Moons, Circles, Blobs, XOR, S-curve
- **Text**: IMDb, Yelp
- **Graph**: Cora, Citeseer

**Features:**
- Automatic binary classification conversion
- Normalization (standard or min-max)
- Feature flattening for images
- PCA dimensionality reduction
- Class balancing option
- Flexible sample limiting

### utils.py

Utility functions for quantum data encoding and processing.

**Encoding Functions:**
- `encode_ry_batch()`: RY rotation encoding
- `encode_angle_batch()`: Angle encoding (sequential, parallel, dense)
- `encode_pauli_batch()`: Pauli rotation encoding
- `encode_iqp_batch()`: IQP encoding with entanglement
- `encode_diagonal_batch()`: Diagonal unitary encoding

**Additional Utilities:**
- Data normalization and scaling
- Quantum state preparation helpers
- Unitary operations

### plot_results.py

Visualization utilities for benchmarking results.

**Functions:**
- `plot_comparison_chart()`: Create grouped bar chart comparisons
- `plot_accuracy_comparison()`: Compare model accuracies
- `plot_loss_curves()`: Visualize training loss
- `save_results_summary()`: Generate summary reports
- Support for multi-dataset and multi-model comparisons

## Usage Examples

### Running Benchmarks

**QRENN Encoding Benchmark:**
```bash
python qrenn_benchmark.py
```

**QRNN Multiple Experiments:**
```bash
python qrnn_benchmark.py
```

**QSVM Benchmark:**
```bash
python qsvm_benchmark.py
```

**All Models Comparison:**
```python
from qrenn_benchmark import run_encoding_benchmark
from qrnn_benchmark import run_multiple_experiments
from qsvm_benchmark import run_qsvm_benchmark

# Compare all models on same dataset
results = {
    'QRENN': run_encoding_benchmark('iris'),
    'QRNN': run_multiple_experiments('iris'),
    'QSVM': run_qsvm_benchmark('iris')
}

from plot_results import plot_comparison_chart
plot_comparison_chart(results, save_path='model_comparison.png')
```

### Running Notebooks

**QRENN Tests:**
```bash
jupyter notebook qrenn_test.ipynb
```

**MNIST Learning Example:**
```bash
jupyter notebook qkit_qrenn_MINST_learning.ipynb
```

**Quantum SVM Examples:**
```bash
jupyter notebook qsvm_ibm_zzfmap.ipynb
```

### Custom Model Configuration

**QRENN with Custom Parameters:**
```python
from QRENN import QRENN
from utils import encode_iqp_batch

qrenn = QRENN(
    num_trainable=3,
    num_control=2,
    num_layers=5,
    use_conjugate=True
)

# Use IQP encoding
data_unitaries = encode_iqp_batch(X_train, num_qubits=2, depth=2)
qrenn.set_data(data_unitaries[0])
```

**QSVM with Different Encodings:**
```python
from QSVM import QSVM

# Try different encoding types
for encoding in ['iqp', 'angle', 'pauli']:
    qsvm = QSVM(n_qubits=2, depth=2, encoding_type=encoding)
    qsvm.set_data(X_train, y_train)
    qsvm.fit()
    predictions = qsvm.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"{encoding} encoding: {accuracy:.4f}")
```

**Custom Training Configuration:**
```python
from train import Trainer

trainer = Trainer(
    learning_rate=0.01,
    iterations=5000,
    trainable_qubits=3,
    ancilla_qubits=8,
    optimizer='adamw',
    scheduler='cosine',
    checkpoint_dir='./checkpoints',
    device='cuda'  # or 'cpu'
)
```

### Plotting and Visualization

**Compare Encoding Methods:**
```python
from plot_results import plot_encoding_comparison

results = {
    'diagonal': [0.85, 0.87, 0.86, 0.88],
    'ry': [0.82, 0.84, 0.83, 0.85],
    'iqp': [0.88, 0.90, 0.89, 0.91]
}

plot_encoding_comparison(
    results,
    title='Encoding Methods Comparison',
    save_path='encoding_comparison.png'
)
```

**Multi-Dataset Comparison:**
```python
from plot_results import plot_multi_dataset_comparison

all_results = {
    'iris': run_encoding_benchmark('iris'),
    'breast_cancer': run_encoding_benchmark('breast_cancer'),
    'ionosphere': run_encoding_benchmark('ionosphere')
}

plot_multi_dataset_comparison(
    all_results,
    save_path='multi_dataset_comparison.png'
)
```

## Technical Details

### QRENN Circuit Architecture

The QRENN model implements a quantum circuit with:
- **Trainable Qubits**: Qubits with universal gates (RZ, RY, RZ) for trainable parameters
- **Control Qubits**: Qubits for data embedding via controlled oracles
- **Alternating Layers**: Layers of trainable and data-embedding operations
- **Conjugate Layers**: Optional U and U† layer alternation for enhanced expressivity

**Circuit Structure:**
```
U(θ₁) - U(data) - U(θ₂) - U(data) - ... - U(θ_n)
```

### QRNN Architecture

The QRNN uses data re-uploading with diagonal encoding:
- **Diagonal Encoding**: Data encoded as diagonal unitary operators U(x) = diag(e^(ix₀), e^(ix₁), ...)
- **Variational Layers**: Universal SU(2) rotations (RZ, RY, RZ) for each qubit
- **Entanglement**: CNOT or CZ gates between qubits
- **Measurement**: Pauli-Z measurement on first qubit for classification

**Circuit Structure:**
```
U(θ₀) - U(x) - U(θ₁) - U(x) - ... - U(x) - U(θ_n) - Measure
```

### QSVM Architecture

QSVM uses quantum feature maps combined with classical SVM:
- **Feature Map**: Quantum circuit encoding classical data into quantum states
- **Kernel Computation**: K(i,j) = |⟨Φ(x_i)|Φ(x_j)⟩|² via state overlap
- **Classical SVM**: Standard SVM training with precomputed quantum kernel

**Supported Feature Maps:**
- **IQP**: Instantaneous Quantum Polynomial with RZZ entanglement
- **Angle**: Rotation-based encoding (RY, RZ, RX)
- **Pauli**: Mixed Pauli rotations for rich encoding
- **Amplitude**: Diagonal encoding for efficient state preparation

### ProjectQKM Architecture

Projected Quantum Kernel Model components:
- **Rotation Wall**: Single qubit rotations (RX, RY, RZ) on each qubit
- **V(θ) Evolution**: Heisenberg-type coupling with Trotterization
- **1-RDM Extraction**: Reduced density matrix features from expectation values
- **Classical Classifier**: Neural network on extracted features

**Feature Extraction:**
For each qubit i, compute expectation values:
- rdm[i][0] = ⟨ψ|X_i|ψ⟩
- rdm[i][1] = ⟨ψ|Y_i|ψ⟩
- rdm[i][2] = ⟨ψ|Z_i|ψ⟩

### NeuralQKM Architecture

Neural Quantum Kernel Model with data re-uploading:
- **Data Re-uploading QNN**: Multiple encoding layers with trainable parameters
- **Neural EQK**: n-to-n approach using trained QNN as embedding
- **Neural PQK**: 1-RDM from trained QNN for kernel computation
- **SVM Classification**: Classical SVM on quantum kernel matrix

**Kernel Types:**
- **EQK (n-to-n)**: k_ij = |⟨0|ψ_i⟩⟨ψ_j|0⟩|²
- **PQK**: k_ij = tr(ρ₁(x_i) * ρ₁(x_j))

### Data Processing

**Features:**
- Automatic binary classification conversion from multi-class
- Normalization (standard or min-max) to [0, 2π] for quantum circuits
- Feature flattening for image datasets
- PCA dimensionality reduction to 2^n_qubits
- Class balancing option for imbalanced datasets
- Flexible sample limiting for controlled experiments

**Supported Datasets:**
| Type | Datasets |
|------|----------|
| Image | MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 |
| Tabular | Iris, Wine, Breast Cancer, Ionosphere |
| Synthetic | Moons, Circles, Blobs, XOR, S-curve |
| Text | IMDb, Yelp |
| Graph | Cora, Citeseer |

## Benchmarking

### Available Benchmarks

Each model includes a dedicated benchmark script:

1. **QRENN Benchmark** (`qrenn_benchmark.py`):
   - Compares 5 encoding methods: diagonal, ry, angle, pauli, iqp
   - Statistical analysis with mean and standard deviation
   - Multiple runs for reliability
   - Comprehensive visualization options

2. **QRNN Benchmark** (`qrnn_benchmark.py`):
   - Multi-experiment runs with different random seeds
   - Configurable qubit count and layer parameters
   - Automatic PCA dimensionality adjustment
   - Detailed training history tracking

3. **QSVM Benchmark** (`qsvm_benchmark.py`):
   - Encoding method comparison
   - Quantum kernel matrix analysis
   - SVM parameter tuning (C, gamma)
   - Performance metrics and visualization

4. **ProjectQKM Benchmark** (`pqkm_benchmark.py`):
   - Trotter step optimization
   - Feature dimension analysis
   - Classical classifier hyperparameter tuning

5. **NeuralQKM Benchmark** (`nqkm_benchmark.py`):
   - EQK vs PQK kernel comparison
   - QNN training optimization
   - Entanglement strategies evaluation

### Running Benchmarks

All benchmarks follow a similar pattern:

```python
# Basic benchmark
python [model]_benchmark.py

# Custom configuration
from [model]_benchmark import run_benchmark

results = run_benchmark(
    dataset_name='iris',
    n_runs=10,
    n_qubits=2,
    verbose=True
)

# Visualization
from plot_results import plot_results
plot_results(results, save_path='results.png')
```

## Citation

This implementation includes work based on the following papers:

```
Jing, et al. "Quantum Recurrent Embedding Neural Networks"
```

**Additional References:**
- TensorFlow Quantum: "Quantum data" tutorial (for ProjectQKM)
- "Neural Quantum Kernels" paper (for NeuralQKM)
- Various quantum encoding strategies and kernel methods

## License

This project is provided for research purposes.

## Contributing

Contributions are welcome! Areas for contribution include:
- Additional quantum ML models
- New encoding strategies
- Benchmarking on more datasets
- Performance optimizations
- Documentation improvements

## Acknowledgments

- **QuAIRKit** - A quantum computing framework for quantum machine learning
- **PennyLane** - Quantum machine learning framework
- **PyTorch** - Deep learning framework
- **scikit-learn** - Machine learning utilities

## Contact

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/Dragon-John/QRENN_benchmark).
