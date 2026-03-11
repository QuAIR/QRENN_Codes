"""
QRNN Benchmark Example - MNIST Classification
==============================================

This script demonstrates how to benchmark the Quantum Data Reuploading Neural Network (QRNN)
on MNIST-like data using sklearn's digits dataset.
"""

import numpy as np
import torch
import warnings
from typing import Tuple

import quairkit as qkit
from sklearn.decomposition import PCA
from quairkit import Hamiltonian

from QRNN import QRNN
from train import Trainer
from utils import scale_to_0_2pi

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
    """Configuration class for QRNN experiments."""

    def __init__(self):
        # Dataset
        self.dataset_name = 'digits'
        self.num_samples = 100
        self.binary_classes = (0, 1)
        self.test_size = 0.2

        # PCA
        self.use_pca = True
        self.n_components = 4

        # QRNN
        self.num_qubits = 4
        self.num_layers = 3
        self.encoding = 'angle'
        self.entanglement = 'linear'
        self.trainable_rots = ['RX', 'RY', 'RZ']

        # Training
        self.learning_rate = 0.1
        self.iterations = 1000
        self.batch_size = None
        self.device = 'cpu'
        self.optimizer = 'adam'
        self.scheduler = 'plateau'
        self.threshold = 0.0

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(\n"
            f"  dataset={self.dataset_name}, samples={self.num_samples}, classes={self.binary_classes}\n"
            f"  pca={self.use_pca}, n_components={self.n_components}\n"
            f"  qrnn=({self.num_qubits}q, {self.num_layers}L, '{self.encoding}', '{self.entanglement}')\n"
            f"  lr={self.learning_rate}, iter={self.iterations}\n"
            f")"
        )


# ============================================================================
# Benchmark Functions
# ============================================================================

def prepare_data(config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare dataset for QRNN training."""
    X_train, y_train, X_test, y_test, _ = load_dataset(
        config.dataset_name,
        test_size=config.test_size,
        binary_classes=config.binary_classes,
        num_samples=config.num_samples
    )

    if config.use_pca:
        pca = PCA(n_components=config.n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    X_train = scale_to_0_2pi(X_train)
    X_test = scale_to_0_2pi(X_test)

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def evaluate_model(qrnn, X_test: np.ndarray, y_test: np.ndarray,
                   config: ExperimentConfig) -> float:
    """Evaluate trained QRNN on test set."""
    # Compute predictions on test set
    predictions = []
    pred_labels = []

    for i in range(len(X_test)):
        x = torch.tensor(X_test[i], dtype=torch.float32)

        # Set test data
        qrnn.set_data(x)

        # Compute prediction
        cir = qrnn.circuit
        z0 = Hamiltonian([[1.0, ",".join([f"Z{j}" for j in range(config.num_qubits)])]])
        pred = cir().expec_val(z0).real.item()

        predictions.append(pred)
        pred_labels.append(1 if pred >= config.threshold else 0)

    pred_labels = np.array(pred_labels)
    accuracy = 1 - np.mean(np.abs(y_test - pred_labels))
    return accuracy


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(config: ExperimentConfig):
    """Run complete QRNN benchmark."""
    print(config)
    print("-" * 50)

    # Prepare data
    print("\nData preparation...")
    X_train, X_test, y_train, y_test = prepare_data(config)

    # Build QRNN
    print("\nBuilding QRNN...")
    qrnn = QRNN(
        num_qubits=config.num_qubits,
        num_layers=config.num_layers,
        encoding=config.encoding,
        entanglement=config.entanglement,
        trainable_rots=config.trainable_rots
    )
    qrnn.set_data(torch.tensor(X_train[0], dtype=torch.float32))
    print(f"Model: {qrnn}")
    print(f"Parameters: {qrnn.num_parameters()}")

    # Setup trainer
    print("\nTraining...")
    trainer = Trainer(
        learning_rate=config.learning_rate,
        iterations=config.iterations,
        batch_size=config.batch_size,
        device=config.device,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        threshold=config.threshold,
        trainable_qubits=config.num_qubits,
        ancilla_qubits=0
    )

    # Train - need custom training loop for QRNN since data changes each forward
    # We'll iterate through training samples
    from quairkit.qinfo import trace

    params = list(qrnn.parameters())
    opt = torch.optim.Adam(params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5
    )

    history = {'loss': [], 'lr': [], 'accuracy': []}

    print("Training with data reuploading...")

    for i in range(config.iterations):
        opt.zero_grad()

        # Accumulate loss over all training samples
        total_loss = 0.0
        for j in range(len(y_train)):
            x = torch.tensor(X_train[j], dtype=torch.float32)
            labels_tensor_j = torch.tensor([y_train[j]], dtype=torch.float32)

            qrnn.set_data(x)

            # Compute loss for this sample
            M = trainer.get_measurement(labels_tensor_j)
            rho = qrnn().density_matrix
            loss = 1 - torch.mean(trace(rho @ M).real)
            total_loss += loss

        avg_loss = total_loss / len(y_train)
        avg_loss.backward()
        opt.step()
        scheduler.step(avg_loss)

        loss_val = avg_loss.item()
        current_lr = opt.param_groups[0]['lr']

        history['loss'].append(loss_val)
        history['lr'].append(current_lr)

        if i % 100 == 0 or i == config.iterations - 1:
            # Compute accuracy on training set
            train_acc = 0
            for j in range(len(y_train)):
                x = torch.tensor(X_train[j], dtype=torch.float32)
                qrnn.set_data(x)
                cir = qrnn.circuit
                z0 = Hamiltonian([[1.0, ",".join([f"Z{k}" for k in range(config.num_qubits)])]])
                pred = cir().expec_val(z0).real.item()
                pred_label = 1 if pred >= config.threshold else 0
                if pred_label == y_train[j]:
                    train_acc += 1
            train_acc = train_acc / len(y_train)

            history['accuracy'].append(train_acc)
            print(f"iter: {i}, loss: {loss_val:.8f}, accuracy: {train_acc:.4f}, lr: {current_lr:.2E}")

    final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0
    print(f"Final training accuracy: {final_train_acc:.4f}")

    # Evaluate
    print("\nEvaluation...")
    test_accuracy = evaluate_model(qrnn, X_test, y_test, config)
    print(f"Test accuracy: {test_accuracy:.4f}")

    return params, test_accuracy, history


if __name__ == "__main__":
    config = ExperimentConfig()
    trained_params, test_accuracy, history = run_benchmark(config)
