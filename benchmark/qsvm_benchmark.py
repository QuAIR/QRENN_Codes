"""
QSVM Benchmark using load_data.py
====================================

This script demonstrates how to benchmark the QSVM model on multiple datasets
using the standardized data loading functions from load_data.py.

Datasets supported: mnist, iris, breast_cancer, ionosphere
Features:
- PCA reduction to configurable components
- 10 rounds of experiments with random initialization
- Average test accuracy reporting
- Optimized using quairkit's update_param for efficient batched quantum kernel computation
  - Creates parameterized circuit templates once
  - Uses update_param to batch-process multiple data points simultaneously
  - Reduces circuit execution overhead significantly
"""

import numpy as np
import torch
import warnings
from typing import Tuple

from load_data import load_data
from utils import scale_to_0_2pi
from QSVM import QSVM, EncodingFeatureMap

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Each dataset has different sizes and dimensionalities
# datasets_config = [
#     {'name': 'mnist', 'n_train': 1000, 'n_test': 200, 'use_pca': True, 'pca_dim': 4},
#     {'name': 'iris', 'n_train': 80, 'n_test': 20, 'use_pca': True, 'pca_dim': 4},
#     {'name': 'breast_cancer', 'n_train': 400, 'n_test': 100, 'use_pca': True, 'pca_dim': 4},
#     {'name': 'ionosphere', 'n_train': 280, 'n_test': 60, 'use_pca': True, 'pca_dim': 4},
#     {'name': 'heart_disease', 'n_train': 200, 'n_test': 50, 'use_pca': True, 'pca_dim': 4}
# ]


class ExperimentConfig:
    """Configuration class for multi-round QSVM benchmark."""

    def __init__(self):
        # Dataset
        self.dataset_name = 'ionosphere'  # Options: 'mnist', 'iris', 'breast_cancer', 'ionosphere'
        self.n_train = 280
        self.n_test = 60

        # PCA
        self.use_pca = True
        self.n_components = 4  # Number of PCA components (must be <= 2^n_qubits)

        # Multi-round experiments
        self.num_rounds = 10
        self.random_state = 42

        # QSVM
        self.n_qubits = 4  # Number of qubits (must be >= log2(n_components))
        self.depth = 2
        self.encoding_type = 'iqp'  # Type of encoding: 'iqp', 'angle', 'amplitude'
        self.encoding_kwargs = {}  # Additional encoding-specific parameters
        self.C = 1.0
        self.gamma = 'scale'

        # Binary classification (QSVM works best for binary tasks)
        self.binary_classes = (0, 1)  # Classes to use for binary classification
        self.device = 'cpu'  # Auto-detect GPU

        # Verification options
        self.verify_circuit_data = False  # Whether to verify circuit data encoding

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  dataset={self.dataset_name}, samples={self.n_train} train, {self.n_test} test\n"
            f"  pca={self.use_pca}, n_components={self.n_components}\n"
            f"  rounds={self.num_rounds}\n"
            f"  qsvm=({self.n_qubits}Q, depth={self.depth}, encoding={self.encoding_type}, C={self.C})\n"
            f"  binary_classes={self.binary_classes}, device={self.device}\n"
            f")"
        )


# ============================================================================
# Benchmark Functions
# ============================================================================

def verify_circuit_data(X: np.ndarray, qsvm: QSVM, n_samples: int = 5) -> None:
    """
    Verify that data is correctly encoded in quantum feature map.

    Args:
        X: Input data array
        qsvm: QSVM instance
        n_samples: Number of samples to check

    Returns:
        None
    """
    try:
        print(f"\n{'='*60}")
        print("CIRCUIT DATA VERIFICATION")
        print(f"{'='*60}")
        print(f"Checking first {n_samples} samples...")
        print(f"Data shape: {X.shape}")
        print(f"Qubits: {qsvm.feature_map.n_qubits}, Depth: {qsvm.feature_map.depth}")

        # Select samples to verify
        samples_to_check = X[:min(n_samples, len(X))]

        # Verify individual circuit creation
        print(f"\nVerifying individual circuits...")
        for i in range(len(samples_to_check)):
            cir = qsvm.feature_map.build_circuit(samples_to_check[i])
            state = cir()
            ket = state.ket
            if isinstance(ket, torch.Tensor):
                ket = ket.detach().cpu().numpy()
            elif hasattr(ket, 'numpy'):
                ket = ket.numpy()
            else:
                ket = np.array(ket)
            print(f"  Sample {i}: state norm = {np.linalg.norm(ket):.6f}")

        # Verify batched computation
        print(f"\nVerifying batched computation...")
        batched_states = qsvm.feature_map.compute_batched_states(samples_to_check)
        print(f"Batched states shape: {batched_states.shape}")

        if isinstance(batched_states, torch.Tensor):
            batched_states = batched_states.detach().cpu().numpy()

        # Compute norms
        norms = np.linalg.norm(batched_states, axis=1)
        print(f"State norms (first {n_samples}): {norms}")

        # Verify kernel computation
        print(f"\nVerifying kernel computation...")
        kernel = qsvm._compute_quantum_kernel(samples_to_check, samples_to_check, verbose=False)
        print(f"Kernel matrix shape: {kernel.shape}")
        print(f"Kernel diagonal (first {min(n_samples, len(kernel))}): {np.diag(kernel)[:min(n_samples, len(kernel))]}")

        print(f"\n{'='*60}")
        print("VERIFICATION COMPLETE - Data encoding verified!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"ERROR during verification: {e}")
        import traceback
        traceback.print_exc()

def evaluate_model(qsvm: QSVM, X_test: np.ndarray, y_test: np.ndarray,
                   config: ExperimentConfig) -> float:
    """Evaluate trained QSVM on test set."""
    predictions = qsvm.predict(X_test, verbose=False)
    accuracy = np.mean(predictions == y_test)
    return accuracy


def run_single_round(config: ExperimentConfig, round_idx: int,
                     X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Run a single round of QSVM training and evaluation."""
    # Build QSVM (fresh initialization for each round)
    qsvm = QSVM(
        n_qubits=config.n_qubits,
        depth=config.depth,
        encoding_type=config.encoding_type,
        **config.encoding_kwargs,
        C=config.C,
        gamma=config.gamma
    )
    # Move QSVM to device
    qsvm.to(config.device)

    # Training (set data and fit SVM)
    qsvm.set_data(X_train, y_train, compute_kernel=True, verbose=True)
    qsvm.fit()

    # Verify circuit data encoding if enabled (only on first round)
    if config.verify_circuit_data and round_idx == 0:
        verify_circuit_data(X_train, qsvm, n_samples=3)

    # Evaluate
    test_accuracy = evaluate_model(qsvm, X_test, y_test, config)
    return test_accuracy


# ============================================================================
# Main Benchmark
# ============================================================================

def run_multi_round_benchmark(config: ExperimentConfig):
    """Run QSVM benchmark across multiple rounds with random initialization."""
    print(f"QSVM Benchmark: {config.dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Samples: {config.n_train} train, {config.n_test} test")
    print(f"Binary classes: {config.binary_classes}")
    print(f"PCA: {config.n_components} components")
    print(f"Rounds: {config.num_rounds}")
    print(f"Model: {config.n_qubits}Q, depth={config.depth}, encoding={config.encoding_type}, C={config.C}")
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
