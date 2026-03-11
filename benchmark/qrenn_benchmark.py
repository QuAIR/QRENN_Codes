"""
QRENN Benchmark using load_data.py
==================================

This script demonstrates how to benchmark the QRENN model on multiple datasets
using the standardized data loading functions from load_data.py.

Datasets supported: mnist, iris, breast_cancer, ionosphere
Features:
- PCA reduction to 16 components
- 10 rounds of experiments with random initialization
- Average test accuracy reporting
"""

import numpy as np
import torch
import warnings
from typing import Tuple

import quairkit as qkit
from quairkit import Hamiltonian

from load_data import load_data, StateLoader
from utils import *
from QRENN import QRENN
from train import Trainer

warnings.filterwarnings('ignore')
qkit.set_dtype('complex128')


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
    """Configuration class for multi-round QRENN benchmark."""

    def __init__(self):
        # Dataset
        self.dataset_name = 'ionosphere'  # Options: 'mnist', 'iris', 'breast_cancer', 'ionosphere'
        self.n_train = 280
        self.n_test = 60

        # PCA (fixed at 16 components as required)
        self.use_pca = True
        self.n_components = 4

        # Multi-round experiments
        self.num_rounds = 10
        self.random_state = 42

        # QRENN
        self.num_trainable = 1
        self.num_control = 4  # 4 control qubits needed for 16-dimensional data (2^4 = 16)
        self.num_layers = 4
        self.use_conjugate = True
        self.num_qubits = self.num_trainable + self.num_control  # Total qubits = 1 + 4 = 5
        self.encoding_fn = encode_pauli_batch
        self.state_loader = StateLoader(num_qubits=self.num_qubits)

        # Training
        self.learning_rate = 0.1
        self.iterations = 500
        self.batch_size = None
        self.device = 'cpu'  # Auto-detect GPU
        self.optimizer = 'adam'
        self.scheduler = 'plateau'
        self.threshold = 0.0

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  dataset={self.dataset_name}, samples={self.n_train} train, {self.n_test} test\n"
            f"  pca={self.use_pca}, n_components={self.n_components}\n"
            f"  rounds={self.num_rounds}\n"
            f"  qrenn=({self.num_trainable}T, {self.num_control}C, {self.num_layers}L)\n"
            f"  lr={self.learning_rate}, iter={self.iterations}, device={self.device}\n"
            f")"
        )


# ============================================================================
# Benchmark Functions
# ============================================================================

def evaluate_model(qrenn: QRENN, X_test: np.ndarray, y_test: np.ndarray,
                   config: ExperimentConfig) -> float:
    """
    Evaluate trained QRENN on test set.

    Note: We create a new QRENN instance for evaluation to avoid the issue where
    set_data() rebuilds the circuit with new parameters. We manually copy the
    trained parameters to the new model.
    """
    # Save trained parameters (these are already on the device)
    trained_params = [p.data.clone() for p in qrenn.parameters()]

    # Create a new QRENN instance with test data
    # data_u = encode_diagonal_batch(X_test, config.num_control)
    data_u = config.encoding_fn(X_test, config.num_control)  # Use same encoding as training for evaluation
    # Move test data to device
    data_u = data_u.to(config.device)
    qrenn_test = QRENN(
        num_trainable=config.num_trainable,
        num_control=config.num_control,
        input_state=config.state_loader.load_state(config.dataset_name),
        num_layers=config.num_layers,
        use_conjugate=config.use_conjugate
    )
    # Move test model to device
    qrenn_test.to(config.device)
    qrenn_test.set_data(data_u)

    # Manually copy trained parameters to the new model
    test_params = list(qrenn_test.parameters())
    for i, saved_param in enumerate(trained_params):
        if i < len(test_params):
            test_params[i].data.copy_(saved_param)

    # Compute predictions using the test model
    z0 = Hamiltonian([[1.0, ",".join([f"Z{j}" for j in range(config.num_trainable)])]])
    pred = np.array(qrenn_test().expec_val(z0).real.detach())

    pred_labels = (pred >= config.threshold).astype(int)
    accuracy = 1 - np.mean(np.abs(y_test - pred_labels))
    return accuracy


def run_single_round(config: ExperimentConfig, round_idx: int,
                     X_train: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Run a single round of QRENN training and evaluation."""
    # Encode data as diagonal unitaries
    # data_unitary = encode_diagonal_batch(X_train, config.num_control)
    data_unitary = config.encoding_fn(X_train, config.num_control)

    # Build QRENN (fresh initialization for each round)
    qrenn = QRENN(
        num_trainable=config.num_trainable,
        num_control=config.num_control,
        num_layers=config.num_layers,
        input_state=config.state_loader.load_state(config.dataset_name),
        use_conjugate=config.use_conjugate
    )
    # Move model to device
    qrenn.to(config.device)
    # Move data to device
    data_unitary = data_unitary.to(config.device)
    qrenn.set_data(data_unitary)

    # Setup trainer
    trainer = Trainer(
        learning_rate=config.learning_rate,
        iterations=config.iterations,
        batch_size=config.batch_size,
        device=config.device,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        threshold=config.threshold,
        trainable_qubits=config.num_trainable,
        ancilla_qubits=config.num_control
    )

    # Train - move labels to device
    labels_tensor = torch.tensor(y_train, dtype=torch.float32).to(config.device)
    trained_params = trainer.train(qrenn, labels=labels_tensor)
    # torch.save(qrenn.state_dict(), f'qrenn_params_{config.dataset_name}.pth')

    # Evaluate
    test_accuracy = evaluate_model(qrenn, X_test, y_test, config)
    return test_accuracy


# ============================================================================
# Main Benchmark
# ============================================================================

def run_multi_round_benchmark(config: ExperimentConfig):
    """Run QRENN benchmark across multiple rounds with random initialization."""
    print(f"QRENN Benchmark: {config.dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Samples: {config.n_train} train, {config.n_test} test")
    print(f"PCA: {config.n_components} components")
    print(f"Rounds: {config.num_rounds}")
    print(f"Model: {config.num_trainable}T, {config.num_control}C, {config.num_layers}L")
    print(f"Training: {config.iterations} iterations, lr={config.learning_rate}")
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
    X_train = scale_to_0_2pi(X_train)
    X_test = scale_to_0_2pi(X_test)

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
