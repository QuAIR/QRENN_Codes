"""
ProjectQKM Benchmark using load_data.py
======================================

This script demonstrates how to benchmark the ProjectQKM model on multiple datasets
using the standardized data loading functions from load_data.py.

Datasets supported: mnist, iris, breast_cancer, ionosphere
Features:
- PCA reduction to configurable components
- 10 rounds of experiments with random initialization
- Average test accuracy reporting
- Projected Quantum Kernel (PQK) feature extraction
- Classical neural network classifier on PQK features
"""

import numpy as np
import torch
import warnings
from typing import Tuple

from load_data import load_data
from utils import scale_to_0_2pi
from ProjectQKM import ProjectQKM

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
    """Configuration class for multi-round ProjectQKM benchmark."""

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

        # ProjectQKM
        self.n_qubits = 4  # Number of qubits
        self.n_trotter = 10  # Number of Trotter steps
        self.random_range = (0.0, 2*np.pi)  # Range for random rotation angles
        self.use_classical_data = True  # Whether to use classical data for theta values
        self.n_classes = 2  # Number of output classes

        # Training parameters
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.1

        # Device
        self.device = 'cpu'

        # Verification options
        self.verify_features = False  # Whether to verify PQK features

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  dataset={self.dataset_name}, samples={self.n_train} train, {self.n_test} test\n"
            f"  pca={self.use_pca}, n_components={self.n_components}\n"
            f"  rounds={self.num_rounds}\n"
            f"  eqkm=({self.n_qubits}Q, n_trotter={self.n_trotter}, use_classical={self.use_classical_data})\n"
            f"  training=({self.epochs} epochs, lr={self.learning_rate})\n"
            f"  device={self.device}\n"
            f")"
        )


# ============================================================================
# Benchmark Functions
# ============================================================================

def verify_pqk_features(X: np.ndarray, pqkm: ProjectQKM, n_samples: int = 5) -> None:
    """
    Verify that PQK features are computed correctly.

    Args:
        X: Input data array
        pqkm: ProjectQKM instance
        n_samples: Number of samples to check

    Returns:
        None
    """
    try:
        print(f"\n{'='*60}")
        print("PQK FEATURES VERIFICATION")
        print(f"{'='*60}")
        print(f"Checking first {n_samples} samples...")
        print(f"Data shape: {X.shape}")
        print(f"Qubits: {pqkm.n_qubits}, Trotter steps: {pqkm.n_trotter}")

        # Select samples to verify
        samples_to_check = X[:min(n_samples, len(X))]

        # Verify individual feature computation
        print(f"\nVerifying individual feature computation...")
        for i in range(len(samples_to_check)):
            feature = pqkm.compute_single_feature(samples_to_check[i])
            print(f"  Sample {i}: feature shape = {feature.shape}, "
                  f"norm = {np.linalg.norm(feature):.6f}")

        # Verify batched computation
        print(f"\nVerifying batched computation...")
        batched_features = pqkm.compute_features(samples_to_check, verbose=False)
        print(f"Batched features shape: {batched_features.shape}")

        # Compute statistics
        feature_means = np.mean(batched_features, axis=(1, 2))
        feature_stds = np.std(batched_features, axis=(1, 2))
        print(f"\nFeature statistics:")
        for i in range(len(samples_to_check)):
            print(f"  Sample {i}: mean = {feature_means[i]:.6f}, std = {feature_stds[i]:.6f}")

        print(f"\n{'='*60}")
        print("VERIFICATION COMPLETE - PQK features verified!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"ERROR during verification: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model(pqkm: ProjectQKM, X_test: np.ndarray, y_test: np.ndarray,
                   config: ExperimentConfig) -> float:
    """Evaluate trained ProjectQKM on test set."""
    predictions = pqkm.predict(X_test, verbose=False)
    accuracy = np.mean(predictions == y_test)
    return accuracy


def run_single_round(config: ExperimentConfig, round_idx: int,
                     X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Run a single round of ProjectQKM training and evaluation."""
    # Build ProjectQKM (fresh initialization for each round)
    pqkm = ProjectQKM(
        n_qubits=config.n_qubits,
        n_trotter=config.n_trotter,
        random_range=config.random_range,
        use_classical_data=config.use_classical_data,
        n_classes=config.n_classes
    )
    # Move EmbedQKM to device
    pqkm.to(config.device)

    # Training
    pqkm.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        verbose=True
    )

    # Verify features if enabled (only on first round)
    if config.verify_features and round_idx == 0:
        verify_pqk_features(X_train, pqkm, n_samples=3)

    # Evaluate
    test_accuracy = evaluate_model(pqkm, X_test, y_test, config)
    return test_accuracy


# ============================================================================
# Main Benchmark
# ============================================================================

def run_multi_round_benchmark(config: ExperimentConfig):
    """Run ProjectQKM benchmark across multiple rounds with random initialization."""
    print(f"ProjectQKM Benchmark: {config.dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Samples: {config.n_train} train, {config.n_test} test")
    print(f"PCA: {config.n_components} components")
    print(f"Rounds: {config.num_rounds}")
    print(f"Model: {config.n_qubits}Q, n_trotter={config.n_trotter}, "
          f"use_classical={config.use_classical_data}")
    print(f"Training: {config.epochs} epochs, lr={config.learning_rate}")
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
