"""
Utility Functions for QRENN Benchmarking
==========================================
This module provides processing and encoding functions for quantum machine learning
classification research.

Functions:
    - pdf_coverter: Convert PDF documents to Markdown format
    - scale_to_0_2pi: Scale data to [0, 2π] range
    - encode_diagonal_batch: Encode data using diagonal unitary encoding
    - encode_ry_batch: Encode data using Ry rotation tensor product encoding (Circuit API)
    - encode_angle_batch: Encode data using RX rotation tensor product encoding (Circuit API)
    - encode_pauli_batch: Encode data using mixed Pauli rotations (Circuit API)
    - encode_iqp_batch: Encode data using IQP encoding with RZZ entanglement (Circuit API)
    - to_binary_classification: Convert multi-class labels to binary classification
    - normalize_features: Normalize features using StandardScaler

Author: QRENN Benchmarking Team
"""

import numpy as np
import torch
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler

from quairkit import Circuit

def pdf_coverter(name:str):
    import pymupdf
    doc = pymupdf.open(name)
    text = ""
    for page in doc:
        text += page.get_text()

    with open(f"{name}.md", "w", encoding="utf-8") as f:
        f.write(text)

# ============================================================================
# Data Processing Functions
# ============================================================================

def scale_to_0_2pi(data: np.ndarray, factor:float=2*np.pi) -> np.ndarray:
    """Scale data to [0, 2π] range."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min) * factor


def normalize_features(X: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize features using sklearn StandardScaler."""
    X_float = X.astype(np.float32)

    if method == 'standard':
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_float)
    elif method == 'minmax':
        X_min = X_float.min(axis=0)
        X_max = X_float.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X_normalized = (X_float - X_min) / X_range
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_normalized


def to_binary_classification(
    X: np.ndarray,
    y: np.ndarray,
    class_indices: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Convert multi-class labels to binary classification."""
    unique_classes = np.unique(y)

    if len(unique_classes) <= 2:
        y_binary = (y == unique_classes[-1]).astype(int)
        class_mapping = {
            'original_classes': unique_classes.tolist(),
            'selected_classes': unique_classes.tolist(),
            'mapping': {int(unique_classes[0]): 0, int(unique_classes[-1]): 1}
        }
        return X, y_binary, class_mapping

    if class_indices is None:
        class_indices = tuple(sorted(unique_classes[:2]))

    if class_indices[0] not in unique_classes or class_indices[1] not in unique_classes:
        raise ValueError(f"Class indices {class_indices} not found in data: {unique_classes}")

    mask = np.isin(y, list(class_indices))
    X_filtered = X[mask]
    y_filtered = y[mask]

    y_binary = (y_filtered == class_indices[1]).astype(int)

    class_mapping = {
        'original_classes': unique_classes.tolist(),
        'selected_classes': list(class_indices),
        'mapping': {int(class_indices[0]): 0, int(class_indices[1]): 1}
    }

    return X_filtered, y_binary, class_mapping


# ============================================================================
# Quantum Encoding Functions
# ============================================================================

def encode_ry_batch(X: np.ndarray, num_qubits: int, depth: int = 1) -> np.ndarray:
    """
    Encode data using Ry rotation tensor product encoding.

    Each feature becomes a Ry rotation angle: U = Ry(x[0]) ⊗ Ry(x[1]) ⊗ ...
    The encoding is repeated `depth` times.

    Args:
        X: Data array of shape (n_samples, n_features)
        num_qubits: Number of qubits for the unitary
        depth: Depth of encoding (number of repetitions)

    Returns:
        Batch unitaries of shape (n_samples, 2^num_qubits, 2^num_qubits)
    """
    import torch

    n_samples = X.shape[0]
    dim = 2 ** num_qubits
    batch_unitaries = torch.zeros((n_samples, dim, dim), dtype=torch.complex64)
    qubits_idx = list(range(num_qubits))

    for i in range(n_samples):
        x = X[i].flatten()
        if len(x) > num_qubits:
            x = x[:num_qubits]
        elif len(x) < num_qubits:
            x = np.pad(x, (0, num_qubits - len(x)))

        # Convert to torch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Create circuit and apply Ry rotations for depth layers
        cir = Circuit(num_qubits)
        for _ in range(depth):
            cir.ry(param=x_tensor, qubits_idx=qubits_idx)

        # Get the unitary matrix
        batch_unitaries[i] = cir.matrix

    return batch_unitaries



def get_default_device() -> str:
    """Get the default device (CUDA if available, else CPU)."""
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        return f'cuda ({device_name})'
    return 'cpu'


def encode_diagonal_batch(X: np.ndarray, num_qubits: int) -> torch.Tensor:
    """Encode data using diagonal unitary encoding: U = diag(exp(i * X))."""
    import torch

    n_samples = X.shape[0]
    # For diagonal encoding, we can handle any number of features
    # The dimension of the unitary is max(2^num_qubits, n_features)
    unitary_dim = max(2 ** num_qubits, X.shape[1])
    batch_unitaries = torch.zeros((n_samples, unitary_dim, unitary_dim), dtype=torch.complex64)

    for i in range(n_samples):
        # Pad or truncate to match unitary dimension
        if X.shape[1] < unitary_dim:
            # Pad with zeros if needed
            padded_X = np.pad(X[i], (0, unitary_dim - X.shape[1]))
            batch_unitaries[i] = torch.diag(torch.exp(1j * torch.tensor(padded_X, dtype=torch.complex64)))
        else:
            # Truncate if needed
            truncated_X = X[i][:unitary_dim]
            batch_unitaries[i] = torch.diag(torch.exp(1j * torch.tensor(truncated_X, dtype=torch.complex64)))

    return batch_unitaries



def encode_angle_batch(X: np.ndarray, num_qubits: int, depth: int = 1) -> torch.Tensor:
    """
    Encode data using angle encoding with RX rotation tensor product.

    Each feature becomes an RX rotation angle: U = RX(x[0]) ⊗ RX(x[1]) ⊗ ...
    The encoding is repeated `depth` times.

    Args:
        X: Data array of shape (n_samples, n_features)
        num_qubits: Number of qubits for the unitary
        depth: Depth of encoding (number of repetitions)

    Returns:
        Batch unitaries of shape (n_samples, 2^num_qubits, 2^num_qubits)
    """
    n_samples = X.shape[0]
    dim = 2 ** num_qubits
    batch_unitaries = torch.zeros((n_samples, dim, dim), dtype=torch.complex64)
    qubits_idx = list(range(num_qubits))

    for i in range(n_samples):
        x = X[i].flatten()
        # Pad or truncate to match num_qubits
        if len(x) > num_qubits:
            x = x[:num_qubits]
        elif len(x) < num_qubits:
            x = np.pad(x, (0, num_qubits - len(x)))

        # Convert to torch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Create circuit and apply RX rotations for depth layers
        cir = Circuit(num_qubits)
        for _ in range(depth):
            cir.rx(param=x_tensor, qubits_idx=qubits_idx)

        # Get the unitary matrix
        batch_unitaries[i] = cir.matrix

    return batch_unitaries



def encode_pauli_batch(X: np.ndarray, num_qubits: int, depth: int = 1) -> torch.Tensor:
    """
    Encode data using mixed Pauli rotations (RX, RY, RZ).

    Rotations are applied in a repeating pattern: RX, RY, RZ, RX, RY, RZ, ...
    The unitary is tensor product: U = RX(x[0]) ⊗ RY(x[1]) ⊗ RZ(x[2]) ⊗ ...
    The encoding is repeated `depth` times.

    Args:
        X: Data array of shape (n_samples, n_features)
        num_qubits: Number of qubits for the unitary
        depth: Depth of encoding (number of repetitions)

    Returns:
        Batch unitaries of shape (n_samples, 2^num_qubits, 2^num_qubits)
    """
    n_samples = X.shape[0]
    dim = 2 ** num_qubits
    batch_unitaries = torch.zeros((n_samples, dim, dim), dtype=torch.complex64)
    qubits_idx = list(range(num_qubits))

    for i in range(n_samples):
        x = X[i].flatten()
        # Pad or truncate to match num_qubits
        if len(x) > num_qubits:
            x = x[:num_qubits]
        elif len(x) < num_qubits:
            x = np.pad(x, (0, num_qubits - len(x)))

        # Convert to torch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Create circuit and apply mixed rotations for depth layers
        cir = Circuit(num_qubits)
        for _ in range(depth):
            # Apply rotations in pattern: RX, RY, RZ, RX, RY, RZ, ...
            for j, angle in enumerate(x):
                gate_type = j % 3
                if gate_type == 0:
                    # RX rotation
                    cir.rx(param=torch.tensor([angle]), qubits_idx=[j])
                elif gate_type == 1:
                    # RY rotation
                    cir.ry(param=torch.tensor([angle]), qubits_idx=[j])
                else:
                    # RZ rotation
                    cir.rz(param=torch.tensor([angle]), qubits_idx=[j])

        # Get the unitary matrix
        batch_unitaries[i] = cir.matrix

    return batch_unitaries



def encode_iqp_batch(X: np.ndarray, num_qubits: int, depth: int = 1) -> torch.Tensor:
    """
    Encode data using Instantaneous Quantum Polynomial (IQP) encoding.

    IQP encoding structure:
    1. Initial Hadamard layer: H on all qubits
    2. Z-rotations: RZ on each qubit with data angles
    3. RZZ entanglement: Between qubit pairs with angles θ_ij = x_i * x_j
    4. Final Hadamard layer: H on all qubits

    Full unitary: U_IQP(x) = (H^⊗n · U_RZZ(x) · U_RZ(x) · H^⊗n)^depth

    Args:
        X: Data array of shape (n_samples, n_features)
        num_qubits: Number of qubits for the unitary
        depth: Depth of encoding (number of repetitions)

    Returns:
        Batch unitaries of shape (n_samples, 2^num_qubits, 2^num_qubits)
    """
    n_samples, n_features = X.shape
    dim = 2 ** num_qubits
    batch_unitaries = torch.zeros((n_samples, dim, dim), dtype=torch.complex64)

    # Build entanglement pattern: linear pairs [[0,1], [1,2], ..., [n-2,n-1]]
    entanglement = [[i, i+1] for i in range(num_qubits - 1)]
    qubits_idx = list(range(num_qubits))

    for i in range(n_samples):
        x = X[i].flatten()

        # Pad or truncate to match num_qubits
        if len(x) > num_qubits:
            x = x[:num_qubits]
        elif len(x) < num_qubits:
            x = np.pad(x, (0, num_qubits - len(x)))

        # Convert to torch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Create circuit and build IQP encoding
        cir = Circuit(num_qubits)

        # Apply IQP encoding for depth layers
        for _ in range(depth):
            # Step 1: Initial Hadamard layer
            cir.h(qubits_idx=qubits_idx)

            # Step 2: Z rotations on each qubit
            cir.rz(param=x_tensor, qubits_idx=qubits_idx)

            # Step 3: RZZ entanglement gates
            for qubit_i, qubit_j in entanglement:
                # Entanglement angle: θ_ij = x_i * x_j
                entanglement_angle = x_tensor[qubit_i] * x_tensor[qubit_j]
                cir.rzz(param=entanglement_angle, qubits_idx=[qubit_i, qubit_j])

            # Step 4: Final Hadamard layer
            cir.h(qubits_idx=qubits_idx)

        # Get the unitary matrix
        batch_unitaries[i] = cir.matrix

    return batch_unitaries



