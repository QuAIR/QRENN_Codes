"""
NeuralQKM Benchmark using load_data.py
======================================

This script demonstrates how to benchmark the NeuralQKM model on multiple datasets
using the standardized data loading functions from load_data.py.

Datasets supported: mnist, iris, breast_cancer, ionosphere
Features:
- PCA reduction to configurable components
- 10 rounds of experiments with random initialization
- Average test accuracy reporting
- Neural Quantum Kernels with different kernel types (EQK n-to-n, EQK 1-to-n, PQK)
- Data re-uploading QNN for kernel construction
"""

import numpy as np
import torch
import warnings
from typing import Tuple

from load_data import load_data
from utils import scale_to_0_2pi
from NeuralQKM import NeuralQKM

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Each dataset has different sizes and dimensionalities
# datasets_config = [
#     {'name': 'mnist', 'n_train': 1000, 'n_test': 200, 'use_pca': True, 'pca_dim': 16},
#     {'name': 'iris', 'n_train': 80, 'n_test': 20, 'use_pca': True, 'pca_dim': 4},
#     {'name': 'breast_cancer', 'n_train': 400, 'n_test': 100, 'use_pca': True, 'pca_dim': 16},
#     {'name': 'ionosphere', 'n_train': 280, 'n_test': 60, 'use_pca': True, 'pca_dim': 16},
#     {'name': 'heart_disease', 'n_train': 200, 'n_test': 50, 'use_pca': True, 'pca_dim': 8}
# ]

class ExperimentConfig:
    """Configuration class for multi-round NeuralQKM benchmark."""

    def __init__(self):
        # Dataset
        self.dataset_name = 'ionosphere'  # Options: 'mnist', 'iris', 'breast_cancer', 'ionosphere'
        self.n_train = 280
        self.n_test = 60

        # PCA
        self.use_pca = True
        self.n_components = 4  # Number of PCA components

        # Multi-round experiments
        self.num_rounds = 10
        self.random_state = 42

        # NeuralQKM
        self.n_qubits = 4  # Number of qubits
        self.n_layers = 2  # Number of data re-uploading layers
        self.entanglement = 'cnot'  # Type of entanglement: 'cnot', 'cz', 'control'
        self.kernel_type = 'eqk_n_to_n'  # Options: 'eqk_n_to_n', 'eqk_1_to_n', 'pqk'
        self.C = 1.0
        self.gamma = 'scale'

        # QNN training parameters
        self.qnn_epochs = 30
        self.qnn_learning_rate = 0.1
        self.qnn_batch_size = 24

        # Device
        self.device = 'cpu'

        # Verification options
        self.verify_qnn = False  # Whether to verify QNN data encoding

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  dataset={self.dataset_name}, samples={self.n_train} train, {self.n_test} test\n"
            f"  pca={self.use_pca}, n_components={self.n_components}\n"
            f"  rounds={self.num_rounds}\n"
            f"  nqkm=({self.n_qubits}Q, layers={self.n_layers}, kernel={self.kernel_type})\n"
            f"  qnn_training=({self.qnn_epochs} epochs, lr={self.qnn_learning_rate})\n"
            f"  device={self.device}\n"
            f")"
        )


# ============================================================================
# Benchmark Functions
# ============================================================================

def verify_qnn_state(X: np.ndarray, nqkm: NeuralQKM, n_samples: int = 5) -> None:
    """
    Verify that QNN states are computed correctly.

    Args:
        X: Input data array
        nqkm: NeuralQKM instance
        n_samples: Number of samples to check

    Returns:
        None
    """
    try:
        print(f"\n{'='*60}")
        print("QNN STATE VERIFICATION")
        print(f"{'='*60}")
        print(f"Checking first {n_samples} samples...")
        print(f"Data shape: {X.shape}")
        print(f"Qubits: {nqkm.n_qubits}, Layers: {nqkm.n_layers}")

        # Select samples to verify
        samples_to_check = X[:min(n_samples, len(X))]

        # Verify individual state computation
        print(f"\nVerifying individual state computation...")
        for i in range(len(samples_to_check)):
            state = nqkm.qnn.compute_state(samples_to_check[i])
            ket = state.ket
            if isinstance(ket, torch.Tensor):
                ket_np = ket.detach().cpu().numpy()
            else:
                ket_np = np.array(ket)
            norm = np.linalg.norm(ket_np)
            print(f"  Sample {i}: state norm = {norm:.6f}")

        # Verify batched computation
        print(f"\nVerifying batched computation...")
        batched_states = nqkm.qnn.compute_batch_states(samples_to_check, verbose=False)
        print(f"Batched states shape: {batched_states.shape}")

        # Compute norms
        norms = torch.norm(batched_states, dim=1)
        print(f"State norms (first {n_samples}): {norms}")

        # Verify kernel computation
        print(f"\nVerifying kernel computation...")
        kernel = nqkm.compute_kernel_matrix(samples_to_check, samples_to_check, verbose=False)
        print(f"Kernel matrix shape: {kernel.shape}")
        print(f"Kernel diagonal (first {min(n_samples, len(kernel))}): {np.diag(kernel)[:min(n_samples, len(kernel))]}")

        print(f"\n{'='*60}")
        print("VERIFICATION COMPLETE - QNN states verified!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"ERROR during verification: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model(nqkm: NeuralQKM, X_test: np.ndarray, y_test: np.ndarray,
                   config: ExperimentConfig) -> float:
    """Evaluate trained NeuralQKM on test set."""
    predictions = nqkm.predict(X_test, verbose=False)
    accuracy = np.mean(predictions == y_test)
    return accuracy


def run_single_round(config: ExperimentConfig, round_idx: int,
                     X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Run a single round of NeuralQKM training and evaluation."""
    # Build NeuralQKM (fresh initialization for each round)
    nqkm = NeuralQKM(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        entanglement=config.entanglement,
        kernel_type=config.kernel_type,
        C=config.C,
        gamma=config.gamma
    )
    # Move NeuralQKM to device
    nqkm.to(config.device)

    # Training
    nqkm.fit(
        X_train, y_train,
        epochs=config.qnn_epochs,
        learning_rate=config.qnn_learning_rate,
        batch_size=config.qnn_batch_size,
        verbose=True
    )

    # Verify QNN data encoding if enabled (only on first round)
    if config.verify_qnn and round_idx == 0:
        verify_qnn_state(X_train, nqkm, n_samples=3)

    # Evaluate
    test_accuracy = evaluate_model(nqkm, X_test, y_test, config)
    return test_accuracy


# ============================================================================
# Main Benchmark
# ============================================================================

def run_multi_round_benchmark(config: ExperimentConfig):
    """Run NeuralQKM benchmark across multiple rounds with random initialization."""
    print(f"NeuralQKM Benchmark: {config.dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Samples: {config.n_train} train, {config.n_test} test")
    print(f"PCA: {config.n_components} components")
    print(f"Rounds: {config.num_rounds}")
    print(f"Model: {config.n_qubits}Q, layers={config.n_layers}, kernel={config.kernel_type}, C={config.C}")
    print(f"QNN Training: {config.qnn_epochs} epochs, lr={config.qnn_learning_rate}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    # Load data once before the round loop
    X_train, X_test, y_train, y_test, info = load_data(
        dataset_name=config.dataset_name,
        use_pca=True,
        pca_dim=config.n_components,
        n_train=config.n_train,
        n_test=config.n_test,
        random_state=config.random_state
    )

    # Scale data once
    X_train = scale_to_0_2pi(X_train, factor=1.0)
    X_test = scale_to_0_2pi(X_test, factor=1.0)

    test_accuracies = []

    for round_idx in range(config.num_rounds):
        print(f"\n{'='*50}")
        print(f"Round {round_idx + 1}/{config.num_rounds}")
        print(f"{'='*50}")

        # Run single round with pre-loaded data
        test_accuracy = run_single_round(config, round_idx, X_train, X_test, y_train, y_test)

        # Store and report result
        test_accuracies.append(test_accuracy)
        print(f"\nTest accuracy (Round {round_idx + 1}): {test_accuracy:.4f}")

    # Calculate statistics
    avg_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS over {config.num_rounds} rounds:")
    print(f"  Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Min accuracy: {np.min(test_accuracies):.4f}")
    print(f"  Max accuracy: {np.max(test_accuracies):.4f}")
    print(f"  Individual accuracies: {[f'{acc:.4f}' for acc in test_accuracies]}")
    print(f"{'='*60}")

    return test_accuracies, avg_accuracy


if __name__ == "__main__":
    config = ExperimentConfig()
    test_accuracies, avg_accuracy = run_multi_round_benchmark(config)
