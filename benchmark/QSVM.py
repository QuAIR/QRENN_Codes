"""
Quantum Support Vector Machine (QSVM)
=======================================
A QuAIRKit implementation of QSVM using configurable encoding methods.

This module provides a QSVM implementation compatible with the Trainer class
for quantum supervised learning tasks.

The QSVM approach uses:
1. A quantum feature map (configurable encoding) to encode classical data
2. Quantum kernel computation via state overlap
3. Classical SVM training with the quantum kernel

Supported encoding methods:
- iqp_encoding: Instantaneous Quantum Polynomial
- angle_encoding: Angle encoding via rotation gates (RY, RZ, RX)
- amplitude_encoding: Amplitude encoding for normalized vectors
- zz_encoding: Traditional ZZ feature map

Author: Rebuilt with quairkit encoding methods
"""

import numpy as np
import torch
from typing import Optional, Union, List, Tuple, Dict, Any
from sklearn.svm import SVC
from sklearn.metrics import pairwise_kernels

from quairkit import Circuit, State
from quairkit.database import *
from quairkit.qinfo import dagger
from utils import (
    encode_ry_batch,
    encode_angle_batch,
    encode_pauli_batch,
    encode_iqp_batch,
    encode_diagonal_batch
)


# =============================================================================
# Encoding Feature Map Class
# =============================================================================

class EncodingFeatureMap:
    """
    Base class for quantum feature maps using encoding methods from utils.py.

    This class provides a unified interface for different encoding methods:
    - iqp_encoding: Instantaneous Quantum Polynomial with RZZ entanglement (uses utils.py)
    - angle_encoding: Angle encoding via rotation gates (uses utils.py)
    - pauli_encoding: Mixed Pauli rotations (uses utils.py)
    - amplitude_encoding: Diagonal encoding (uses utils.py)
    """

    def __init__(
        self,
        n_qubits: int = 2,
        encoding_type: str = 'iqp',
        depth: int = 1,
        **encoding_kwargs
    ):
        """
        Initialize the encoding feature map.

        Args:
            n_qubits: Number of qubits (corresponds to number of features)
            encoding_type: Type of encoding ('iqp', 'angle', 'amplitude')
            depth: Depth of the encoding circuit
            **encoding_kwargs: Additional encoding-specific parameters
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type.lower()
        self.depth = depth

        # Encoding-specific parameters
        self.encoding_kwargs = encoding_kwargs

        # Set default parameters for each encoding type
        self._set_default_encoding_params()

    def _set_default_encoding_params(self):
        """Set default parameters for each encoding type."""
        if self.encoding_type == 'iqp':
            # Default entanglement for IQP encoding
            if 'entanglement' not in self.encoding_kwargs:
                # Create linear entanglement: [0,1], [1,2], ..., [n-2,n-1]
                entanglement = [[i, i+1] for i in range(self.n_qubits - 1)]
                self.encoding_kwargs['entanglement'] = entanglement
        elif self.encoding_type == 'angle':
            # Default rotation gate type for angle encoding
            if 'rotation' not in self.encoding_kwargs:
                self.encoding_kwargs['rotation'] = 'RY'
        # Note: 'zz' encoding has been removed as it belongs to IQP family

    def _prepare_data(
        self,
        x_data: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Prepare data for encoding.

        Args:
            x_data: Input data

        Returns:
            Prepared data as torch tensor
        """
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data).float()
        elif isinstance(x_data, torch.Tensor):
            x_data = x_data.float()
        else:
            x_data = torch.tensor(x_data, dtype=torch.float32)

        # Handle single data point
        if x_data.ndim == 1:
            x_data = x_data.unsqueeze(0)  # Shape: (1, n_features)

        # Ensure we have right number of features
        if x_data.shape[1] < self.n_qubits:
            # Pad with zeros if necessary
            padding = torch.zeros(x_data.shape[0], self.n_qubits - x_data.shape[1])
            x_data = torch.cat([x_data, padding], dim=1)
        elif x_data.shape[1] > self.n_qubits:
            # Truncate if too many features
            x_data = x_data[:, :self.n_qubits]

        return x_data

    def build_circuit(self, x_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Build's quantum circuit encoding for a single data point.
        Returns unitary matrix from utils.py encoding functions.

        Args:
            x_data: Classical data point (features)

        Returns:
            Unitary matrix from encoding (torch.Tensor)
        """
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.detach().cpu().numpy()

        # Ensure we have enough features
        if len(x_data) < self.n_qubits:
            x_data = np.pad(x_data, (0, self.n_qubits - len(x_data)))
        elif len(x_data) > self.n_qubits:
            x_data = x_data[:self.n_qubits]

        # Reshape to batch format for utils.py functions
        x_batch = x_data.reshape(1, -1)

        # Use utils.py encoding functions to get unitary matrix
        if self.encoding_type == 'iqp':
            unitary = encode_iqp_batch(x_batch, self.n_qubits, depth=self.depth)
        elif self.encoding_type == 'angle':
            rotation = self.encoding_kwargs.get('rotation', 'RY')
            if rotation == 'RY':
                unitary = encode_ry_batch(x_batch, self.n_qubits, depth=self.depth)
            else:
                unitary = encode_angle_batch(x_batch, self.n_qubits, depth=self.depth)
        elif self.encoding_type == 'pauli':
            unitary = encode_pauli_batch(x_batch, self.n_qubits, depth=self.depth)
        elif self.encoding_type == 'amplitude':
            # Diagonal encoding doesn't support depth (it's a full unitary)
            unitary = encode_diagonal_batch(x_batch, self.n_qubits)
        else:
            raise ValueError(f"Unsupported encoding type: {self.encoding_type}. Use 'iqp', 'angle', 'pauli', or 'amplitude'.")

        # Return first (and only) unitary from batch
        return unitary[0]

    def compute_batched_states(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute quantum states for a batch of data points.
        Uses utils.py encoding functions directly for batch processing.

        Args:
            X: Batch of data points (N samples, n_features)

        Returns:
            Batch of quantum states (N, 2**n_qubits)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        # Prepare data
        X = self._prepare_data(X)
        batch_size = X.shape[0]

        # Use utils.py encoding functions directly for batch processing
        if self.encoding_type == 'iqp':
            unitaries = encode_iqp_batch(X.numpy() if torch.is_tensor(X) else X, self.n_qubits, depth=self.depth)
        elif self.encoding_type == 'angle':
            rotation = self.encoding_kwargs.get('rotation', 'RY')
            if rotation == 'RY':
                unitaries = encode_ry_batch(X.numpy() if torch.is_tensor(X) else X, self.n_qubits, depth=self.depth)
            else:
                unitaries = encode_angle_batch(X.numpy() if torch.is_tensor(X) else X, self.n_qubits, depth=self.depth)
        elif self.encoding_type == 'pauli':
            unitaries = encode_pauli_batch(X.numpy() if torch.is_tensor(X) else X, self.n_qubits, depth=self.depth)
        elif self.encoding_type == 'amplitude':
            # Diagonal encoding doesn't support depth (it's a full unitary)
            unitaries = encode_diagonal_batch(X.numpy() if torch.is_tensor(X) else X, self.n_qubits)
        else:
            raise ValueError(f"Unsupported encoding type: {self.encoding_type}")

        # Extract quantum states from unitaries (first column corresponds to |0>^⊗n)
        states = unitaries[:, :, 0]  # Shape: (N, 2**n_qubits)

        return states


# =============================================================================
# Quantum Kernel Estimator
# =============================================================================

class QuantumKernelEstimator:
    """
    Computes quantum kernel matrices using a feature map.
    The kernel K(i,j) = |<Phi(x_i)|Phi(x_j)>|^2
    """
    def __init__(self, feature_map: EncodingFeatureMap):
        """
        Initialize the quantum kernel estimator.

        Args:
            feature_map: Quantum feature map instance
        """
        self.feature_map = feature_map
        self.n_qubits = feature_map.n_qubits

    def compute_kernel_matrix(
        self,
        X1: Union[np.ndarray, torch.Tensor],
        X2: Union[np.ndarray, torch.Tensor],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Kernel computation using the feature map.

        Args:
            X1: First dataset (N1 samples, n_features)
            X2: Second dataset (N2 samples, n_features)
            verbose: Print progress messages

        Returns:
            Kernel matrix of shape (N1, N2)
        """
        if isinstance(X1, torch.Tensor):
            X1 = X1.detach().cpu().numpy()
        if isinstance(X2, torch.Tensor):
            X2 = X2.detach().cpu().numpy()

        N1 = len(X1)
        N2 = len(X2)

        if verbose:
            print(f"Computing {N1}x{N2} kernel matrix using {self.feature_map.encoding_type} encoding...")

        # Compute batched states
        X1_states = self.feature_map.compute_batched_states(X1)  # Shape: (N1, 2**n_qubits)
        X2_states = self.feature_map.compute_batched_states(X2)  # Shape: (N2, 2**n_qubits)

        # Convert to numpy if needed
        if isinstance(X1_states, torch.Tensor):
            X1_states = X1_states.detach().cpu().numpy()
        if isinstance(X2_states, torch.Tensor):
            X2_states = X2_states.detach().cpu().numpy()

        # Compute kernel matrix using matrix multiplication
        # For complex vectors: <psi_i|psi_j> = conj(psi_i)^T @ psi_j
        kernel_matrix = np.abs(X1_states.conj() @ X2_states.T)**2

        return kernel_matrix


# =============================================================================
# QSVM Class
# =============================================================================

class QSVM(torch.nn.Module):
    """
    Quantum Support Vector Machine implementation as a PyTorch module.

    This class wraps the QSVM approach (quantum feature map + classical SVM)
    into a torch.nn.Module format compatible with the Trainer class.

    Note: The actual SVM training is done using scikit-learn's SVC,
    which does not support gradient-based optimization. This wrapper
    provides the same interface as QRENN for consistency.

    Supports configurable encoding methods (default: zz_encoding).

    Usage:
        >>> qsvm = QSVM(n_qubits=2, depth=2, encoding_type='iqp')
        >>> predictions = qsvm.predict(X_test, y_train=y_train, X_train=X_train)
    """

    def __init__(
        self,
        n_qubits: int = 2,
        depth: int = 2,
        encoding_type: str = 'iqp',
        feature_map: Optional[EncodingFeatureMap] = None,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        **encoding_kwargs
    ):
        """
        Initialize QSVM model.

        Args:
            n_qubits: Number of qubits (features)
            depth: Depth of feature map circuit
            encoding_type: Type of encoding ('iqp', 'angle', 'amplitude')
            feature_map: Custom feature map (if None, uses EncodingFeatureMap)
            C: SVM regularization parameter
            kernel: Kernel type ('rbf', 'linear', 'poly', etc.)
            gamma: Kernel coefficient ('scale', 'auto', or float)
            **encoding_kwargs: Additional encoding-specific parameters
        """
        super().__init__()

        self._n_qubits = n_qubits
        self._depth = depth
        self._encoding_type = encoding_type.lower()
        self._C = C
        self._kernel_type = kernel
        self._gamma = gamma

        # Initialize feature map
        if feature_map is None:
            self.feature_map = EncodingFeatureMap(
                n_qubits=n_qubits,
                encoding_type=encoding_type,
                depth=depth,
                **encoding_kwargs
            )
        else:
            self.feature_map = feature_map

        # Initialize kernel estimator
        self.kernel_estimator = QuantumKernelEstimator(self.feature_map)

        # State and data storage
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._kernel_matrix: Optional[np.ndarray] = None
        self._predictions: Optional[np.ndarray] = None
        self._fitted = False

        # SVM model (will be fitted during training)
        self._svm = SVC(
            C=C,
            kernel='precomputed',
            gamma=gamma,
            random_state=42
        )

    def _compute_quantum_kernel(
        self,
        X1: Union[np.ndarray, torch.Tensor],
        X2: Union[np.ndarray, torch.Tensor],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Compute the quantum kernel matrix between two datasets.

        Args:
            X1: First dataset
            X2: Second dataset
            verbose: Print progress

        Returns:
            Quantum kernel matrix
        """
        return self.kernel_estimator.compute_kernel_matrix(X1, X2, verbose=verbose)

    def set_data(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        compute_kernel: bool = True,
        verbose: bool = True
    ) -> 'QSVM':
        """
        Set training data and optionally precompute the kernel matrix.

        Args:
            X_train: Training features
            y_train: Training labels
            compute_kernel: Whether to precompute the kernel matrix
            verbose: Print progress messages

        Returns:
            self (for chain calling)
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        self._X_train = X_train
        self._y_train = y_train

        if compute_kernel:
            self._kernel_matrix = self._compute_quantum_kernel(X_train, X_train, verbose=verbose)

        return self

    def fit(self, X_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_train: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """
        Fit SVM model on the quantum kernel.

        Args:
            X_train: Training features (optional if already set via set_data)
            y_train: Training labels (optional if already set via set_data)
        """
        # Use provided data or stored data
        if X_train is not None and y_train is not None:
            self.set_data(X_train, y_train, compute_kernel=True)

        if self._kernel_matrix is None:
            raise ValueError("No kernel matrix computed. Call set_data() first or provide X_train, y_train.")

        self._svm.fit(self._kernel_matrix, self._y_train)
        self._fitted = True

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        compute_kernel: bool = True,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Predict labels for new data.

        Args:
            X: Test features
            compute_kernel: Whether to compute kernel matrix
            verbose: Print progress

        Returns:
            Predicted labels
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        if compute_kernel:
            kernel_test = self._compute_quantum_kernel(X, self._X_train, verbose=verbose)
        else:
            # Assume kernel is precomputed (not commonly used)
            kernel_test = X

        return self._svm.predict(kernel_test)

    def forward(
        self,
        state: Optional[State] = None,
        X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
        X_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> State:
        """
        Forward pass - returns quantum state for compatibility with QRENN.

        For QSVM, this returns a mock quantum state that encodes predictions.
        This is a compatibility layer to work with the Trainer interface.

        Args:
            state: Input quantum state (optional, ignored in QSVM)
            X_test: Test data for prediction
            y_train: Training labels (for fitting)
            X_train: Training features (for fitting)

        Returns:
            A mock State object for compatibility
        """
        # If training data provided, fit the model
        if X_train is not None and y_train is not None:
            self.set_data(X_train, y_train)
            self.fit()

        # If test data provided, make predictions
        if X_test is not None:
            self._predictions = self.predict(X_test)

        # Return a mock state for compatibility with Trainer
        # Create a simple 2-qubit state that encodes prediction info
        mock_state = State(2)
        return mock_state

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def encoding_type(self) -> str:
        return self._encoding_type

    @property
    def C(self) -> float:
        return self._C

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def predictions(self) -> Optional[np.ndarray]:
        return self._predictions

    @property
    def kernel_matrix(self) -> Optional[np.ndarray]:
        return self._kernel_matrix

    def get_info(self) -> Dict[str, Any]:
        """Get model configuration info."""
        return {
            'n_qubits': self._n_qubits,
            'depth': self._depth,
            'encoding_type': self._encoding_type,
            'C': self._C,
            'kernel_type': self._kernel_type,
            'gamma': self._gamma,
            'is_fitted': self._fitted,
        }

    def reset(self) -> None:
        """Reset model, clearing training data and fitted state."""
        self._X_train = None
        self._y_train = None
        self._kernel_matrix = None
        self._predictions = None
        self._fitted = False
        self._svm = SVC(
            C=self._C,
            kernel='precomputed',
            gamma=self._gamma,
            random_state=42
        )

    def to(self, device: Union[str, torch.device]) -> 'QSVM':
        """Mock method for PyTorch compatibility."""
        return self

    def __repr__(self) -> str:
        fitted_status = "fitted" if self._fitted else "not fitted"
        return (
            f"QSVM(n_qubits={self._n_qubits}, "
            f"encoding_type={self._encoding_type}, "
            f"depth={self._depth}, "
            f"C={self._C}, "
            f"status={fitted_status})"
        )


# =============================================================================
# Helper Functions for Trainer Compatibility
# =============================================================================

def create_qsvm_loss_fn(qsvm_model: QSVM, labels: torch.Tensor):
    """
    Create a loss function for QSVM that works with Trainer class.

    Note: Since QSVM uses non-differentiable SVM, this is a mock loss
    function for compatibility. Actual training is done via fit().

    Args:
        qsvm_model: QSVM instance
        labels: Training labels

    Returns:
        Loss function compatible with Trainer
    """
    def loss_fn(model, labels):
        # For QSVM, we use a pseudo-loss based on prediction accuracy
        if qsvm_model.is_fitted and qsvm_model.predictions is not None:
            pred_tensor = torch.tensor(qsvm_model.predictions, dtype=torch.float32)
            label_tensor = labels.float() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float32)
            return 1.0 - torch.mean(1.0 - torch.abs(pred_tensor - label_tensor))
        else:
            # Return a mock loss if not fitted
            return torch.tensor(0.5, requires_grad=True)

    return loss_fn


def create_qsvm_predict_fn(qsvm_model: QSVM):
    """
    Create a predict function for QSVM that works with Trainer class.

    Args:
        qsvm_model: QSVM instance

    Returns:
        Predict function compatible with Trainer
    """
    def predict_fn(model):
        if qsvm_model.predictions is not None:
            return torch.tensor(qsvm_model.predictions, dtype=torch.float32)
        else:
            return torch.tensor([0.5], dtype=torch.float32)

    return predict_fn


# =============================================================================
# Main Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split

    # Generate test data
    X, y = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)

    # Scale data to [0, 2pi] range
    min_val, max_val = X.min(), X.max()
    X_scaled = 2 * np.pi * (X - min_val) / (max_val - min_val)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Test different encoding types
    for encoding_type in ['iqp', 'angle']:
        print(f"\n{'='*60}")
        print(f"Testing {encoding_type.upper()} encoding")
        print(f"{'='*60}")

        # Initialize QSVM
        qsvm = QSVM(n_qubits=2, depth=2, encoding_type=encoding_type, C=1.0)

        # Train model
        print(f"Training QSVM with {encoding_type} encoding...")
        qsvm.set_data(X_train, y_train, compute_kernel=True, verbose=True)
        qsvm.fit()

        # Make predictions
        print("Making predictions...")
        predictions = qsvm.predict(X_test, verbose=True)

        # Compute accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Test accuracy: {accuracy:.4f}")
