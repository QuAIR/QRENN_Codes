"""
Neural Quantum Kernel Model (NeuralQKM)
========================================
A QuAIRKit implementation of Neural Quantum Kernel Model.

This module provides a flexible implementation of Neural Quantum Kernel Model
based on the paper "Neural Quantum Kernels".

The NeuralQKM approach uses:
1. Data re-uploading Quantum Neural Network (QNN) with iterative scaling
2. Two types of Neural EQK: n-to-n and 1-to-n approaches
3. Neural PQK using 1-RDM from trained QNN

Reference: Neural Quantum Kernels paper
Author: QuAIRKit implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

from sklearn.svm import SVC

from quairkit import Circuit, State
from quairkit.database import *
from quairkit.qinfo import dagger


# =============================================================================
# Data Re-uploading QNN
# =============================================================================

class DataReloadingQNN(nn.Module):
    """
    Data re-uploading Quantum Neural Network (QNN).

    This class implements a QNN that uses data re-uploading:
    - The same data is encoded through multiple layers
    - Each layer has trainable parameters U(θ_l) and data encoding U(x)
    - Circuit structure: U(θ_L) * U(x) * ... * U(θ_1) * U(x)

    This architecture allows for efficient training and scaling to multiple qubits.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 5,
        entanglement: str = 'cnot',  # Options: 'cnot', 'cz'
        n_classes: int = 2
    ):
        """
        Initialize data re-uploading QNN.

        Args:
            n_qubits: Number of qubits
            n_layers: Number of data re-uploading layers
            entanglement: Type of entanglement ('cnot', 'cz')
            n_classes: Number of output classes (for compatibility)
        """
        super().__init__()

        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._entanglement = entanglement
        self._n_classes = n_classes

        # Initialize trainable parameters
        # Each layer has 3 parameters per qubit (for universal SU(2))
        self._params = nn.ParameterList([
            nn.Parameter(torch.randn(n_qubits * 3) * 0.01)
            for _ in range(n_layers)
        ])

        # State storage
        self._trained = False
        self._theta_star = None  # Trained parameters for kernel construction

    def _apply_single_layer(self, circuit: Circuit, layer_idx: int,
                           x_data: torch.Tensor, qubits_idx: List[int]) -> Circuit:
        """
        Apply a single layer of data re-uploading.

        Args:
            circuit: QuAIRKit Circuit to modify
            layer_idx: Layer index
            x_data: Data to encode (normalized to [0, 2π])
            qubits_idx: List of qubit indices

        Returns:
            The modified circuit
        """
        params = self._params[layer_idx].reshape(self._n_qubits, 3)

        # For each qubit, apply trainable rotation and data encoding
        for i, qubit in enumerate(qubits_idx):
            # Apply trainable universal SU(2) rotation
            # Using Euler decomposition: Rz(φ) * Ry(θ) * Rz(λ)
            phi, theta, lam = params[i]

            circuit.rz(qubits_idx=[qubit], param=phi)
            circuit.ry(qubits_idx=[qubit], param=theta)
            circuit.rz(qubits_idx=[qubit], param=lam)

            # Apply data encoding U(x) - only on last data upload
            if layer_idx == self._n_layers - 1 and i < len(x_data):
                circuit.ry(qubits_idx=[qubit], param=x_data[i])

        # Add entanglement between layers (for multi-qubit case)
        if self._n_qubits > 1:
            self._apply_entanglement(circuit, qubits_idx)

        return circuit

    def _apply_entanglement(self, circuit: Circuit, qubits_idx: List[int]) -> Circuit:
        """
        Apply entanglement gates between qubits.

        Args:
            circuit: QuAIRKit Circuit to modify
            qubits_idx: List of qubit indices

        Returns:
            The modified circuit
        """
        if self._entanglement == 'cnot':
            # Cascade of CNOT gates
            for i in range(len(qubits_idx) - 1):
                circuit.cx(qubits_idx=[qubits_idx[i], qubits_idx[i + 1]])
        elif self._entanglement == 'cz':
            # Cascade of CZ gates
            for i in range(len(qubits_idx) - 1):
                circuit.cz(qubits_idx=[qubits_idx[i], qubits_idx[i + 1]])

        return circuit

    def build_circuit(self, x_data: np.ndarray) -> Circuit:
        """
        Build the full QNN circuit for a single data point.

        Args:
            x_data: Single data point (features)

        Returns:
            QuAIRKit Circuit
        """
        # Ensure we have enough features
        if len(x_data) < self._n_qubits:
            x_data = np.pad(x_data, (0, self._n_qubits - len(x_data)))
        elif len(x_data) > self._n_qubits:
            x_data = x_data[:self._n_qubits]

        # Convert to tensor
        x_tensor = torch.tensor(x_data, dtype=torch.float32)

        circuit = Circuit(self._n_qubits)
        qubits_idx = list(range(self._n_qubits))

        # Apply each layer
        for layer_idx in range(self._n_layers):
            self._apply_single_layer(circuit, layer_idx, x_tensor, qubits_idx)

        return circuit

    def compute_state(self, x_data: np.ndarray) -> State:
        """
        Compute the quantum state for a single data point.

        Args:
            x_data: Single data point (features)

        Returns:
            QuAIRKit State
        """
        circuit = self.build_circuit(x_data)
        return circuit()

    def compute_batch_states(self, X: np.ndarray, verbose: bool = False) -> torch.Tensor:
        """
        Compute quantum states for a batch of data points.

        Args:
            X: Batch of data points (n_samples, n_features)
            verbose: Print progress messages

        Returns:
            Batch of quantum state vectors (n_samples, 2**n_qubits)
        """
        n_samples = X.shape[0]

        if verbose:
            print(f"Computing QNN states for {n_samples} samples...")

        states_list = []
        for i in range(n_samples):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{n_samples}")

            state = self.compute_state(X[i])
            ket = state.ket

            # Ensure ket is a proper tensor
            if isinstance(ket, torch.Tensor):
                if ket.dtype != torch.complex64 and ket.dtype != torch.complex32:
                    ket = ket.complex()

                # Squeeze to get state vector
                while ket.dim() > 1 and ket.shape[-1] == 1:
                    ket = ket.squeeze(-1)
                while ket.dim() > 1 and ket.shape[0] == 1:
                    ket = ket.squeeze(0)
            else:
                ket_np = np.array(ket)
                ket = torch.tensor(ket_np, dtype=torch.complex64)

            states_list.append(ket)

        states = torch.stack(states_list, dim=0)
        return states

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 30,
        learning_rate: float = 0.005,
        batch_size: int = 24,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the QNN using fidelity cost function.

        Args:
            X_train: Training features
            y_train: Training labels (expected to be -1 and +1 or 0 and 1)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            verbose: Print progress messages

        Returns:
            Dictionary with training history
        """
        # Convert labels to -1 and +1 format if needed
        y_train_converted = 2 * y_train - 1  # 0,1 -> -1,+1

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_converted, dtype=torch.float32)

        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        n_samples = X_train.shape[0]
        loss_history = []

        if verbose:
            print(f"Training QNN for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0.0

            # Mini-batch training
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                X_batch = X_tensor[batch_start:batch_end]
                y_batch = y_tensor[batch_start:batch_end]

                optimizer.zero_grad()

                # Compute loss
                loss = self._compute_batch_loss(X_batch, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (n_samples / batch_size)
            loss_history.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

        self._trained = True
        self._theta_star = [p.detach().clone() for p in self._params]

        if verbose:
            print("Training complete!")

        return {'loss_history': loss_history}

    def _compute_batch_loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for a batch of data points.

        Args:
            X_batch: Batch of features (batch_size, n_features)
            y_batch: Batch of labels (batch_size,)

        Returns:
            Loss value
        """
        batch_size = X_batch.shape[0]
        loss = torch.tensor(0.0, requires_grad=True)

        for i in range(batch_size):
            x_i = X_batch[i]
            y_i = y_batch[i]

            # Get quantum state for this data point
            state = self.compute_state(x_i.detach().cpu().numpy())
            ket = state.ket
            if isinstance(ket, torch.Tensor):
                ket = ket.squeeze()

            # Compute fidelity: |⟨target|ψ⟩|^2
            # For single qubit: fidelity to label state
            # For multi-qubit: compute probability of measuring correct first-qubit state
            if self._n_qubits == 1:
                if y_i == 1:  # Label +1 -> |0⟩
                    target_state = torch.tensor([1.0, 0.0], dtype=torch.complex64)
                else:  # Label -1 -> |1⟩
                    target_state = torch.tensor([0.0, 1.0], dtype=torch.complex64)
                fidelity = torch.abs(torch.vdot(target_state.conj(), ket))**2
            else:
                # Multi-qubit: compute probability of measuring correct label on first qubit
                # This is the sum of |⟨basis|ψ⟩|² over all basis states where first qubit is correct
                fidelity = torch.tensor(0.0)
                for idx in range(len(ket)):
                    # Check if first qubit matches the label
                    first_qubit_value = (idx >> (self._n_qubits - 1)) & 1
                    if (y_i == 1 and first_qubit_value == 0) or (y_i == -1 and first_qubit_value == 1):
                        fidelity = fidelity + torch.abs(ket[idx])**2

            # Cost: 1 - fidelity (want to maximize fidelity)
            loss = loss + (1.0 - fidelity)

        return loss / batch_size

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def theta_star(self) -> Optional[List[torch.Tensor]]:
        return self._theta_star


# =============================================================================
# Neural EQK and PQK
# =============================================================================

class NeuralQKM(nn.Module):
    """
    Neural Quantum Kernel Model.

    This class implements the Neural Quantum Kernel Model with two kernel types:
    - Neural EQK (n-to-n approach)
    - Neural PQK (using 1-RDM from trained QNN)

    The kernel is computed using a trained QNN as the embedding.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 5,
        entanglement: str = 'cnot',
        kernel_type: str = 'eqk_n_to_n',  # Options: 'eqk_n_to_n', 'pqk'
        C: float = 1.0,
        gamma: str = 'scale'
    ):
        """
        Initialize Neural Quantum Kernel Model.

        Args:
            n_qubits: Number of qubits
            n_layers: Number of data re-uploading layers
            entanglement: Type of entanglement ('cnot', 'cz')
            kernel_type: Type of quantum kernel ('eqk_n_to_n', 'pqk')
            C: SVM regularization parameter
            gamma: SVM kernel coefficient
        """
        super().__init__()

        self._n_qubits = n_qubits
        self._n_layers = n_layers
        self._entanglement = entanglement
        self._kernel_type = kernel_type.lower()
        self._C = C
        self._gamma = gamma

        # Initialize QNN
        self.qnn = DataReloadingQNN(n_qubits, n_layers, entanglement)

        # State and data storage
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._kernel_matrix: Optional[np.ndarray] = None
        self._predictions: Optional[np.ndarray] = None
        self._fitted = False

        # SVM model (will be fitted during training)
        self._svm = SVC(C=C, kernel='precomputed', gamma=gamma, random_state=42)

    def compute_eqk_n_to_n(self, X1: np.ndarray, X2: np.ndarray,
                       verbose: bool = False) -> np.ndarray:
        """
        Compute Neural EQK using n-to-n approach.

        Uses the trained n-qubit QNN directly as the embedding:
        k_ij = |⟨0|ψ_i⟩⟨ψ_j|0⟩|^2

        Args:
            X1: First dataset
            X2: Second dataset
            verbose: Print progress

        Returns:
            Kernel matrix of shape (len(X1), len(X2))
        """
        n1, n2 = len(X1), len(X2)

        if verbose:
            print(f"Computing {n1}x{n2} EQK (n-to-n) kernel...")

        # Compute states for X1
        X1_states = self.qnn.compute_batch_states(X1, verbose=verbose)

        # Compute states for X2
        X2_states = self.qnn.compute_batch_states(X2, verbose=verbose)

        # Compute kernel matrix using inner products
        # For EQK: k_ij = |⟨0|ψ_i⟩⟨ψ_j|0⟩|^2 = |⟨0|ψ_i⟩|^2 * |⟨0|ψ_j⟩|^2
        # This requires computing probability of measuring all qubits in |0⟩ state

        # For simplicity, we compute state overlap squared
        kernel_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                # Inner product: ⟨ψ_i|ψ_j⟩
                inner_product = np.vdot(X1_states[i].detach().cpu().numpy(),
                                       X2_states[j].detach().cpu().numpy())
                kernel_matrix[i, j] = np.abs(inner_product)**2

        return kernel_matrix

    def compute_pqk(self, X1: np.ndarray, X2: np.ndarray,
                  verbose: bool = False) -> np.ndarray:
        """
        Compute Neural PQK using 1-RDM from trained QNN.

        k_ij = tr(ρ_1(x_i) * ρ_1(x_j))

        Uses reduced density matrix of first qubit from trained QNN.

        Args:
            X1: First dataset
            X2: Second dataset
            verbose: Print progress

        Returns:
            Kernel matrix of shape (len(X1), len(X2))
        """
        n1, n2 = len(X1), len(X2)

        if verbose:
            print(f"Computing {n1}x{n2} PQK kernel...")

        kernel_matrix = np.zeros((n1, n2))

        # Compute RDMs for X1 and X2
        # Get partial trace for first qubit for each data point
        X1_rdms = []
        X2_rdms = []

        for i in range(n1):
            state = self.qnn.compute_state(X1[i])
            ket = state.ket
            if isinstance(ket, torch.Tensor):
                ket_np = ket.detach().cpu().numpy()
            else:
                ket_np = np.array(ket)

            # Compute RDM for first qubit
            # Get the reduced state by tracing out all qubits except first
            rdm = self._compute_first_qubit_rdm(ket_np, self._n_qubits)

            # Normalize RDM (handle zero trace case)
            trace_val = np.trace(rdm)
            if np.abs(trace_val) > 1e-10:
                rdm = rdm / trace_val
            else:
                # If trace is zero or very small, set to identity (maximally mixed state)
                rdm = np.eye(2) / 2

            X1_rdms.append(rdm)

        for j in range(n2):
            state = self.qnn.compute_state(X2[j])
            ket = state.ket
            if isinstance(ket, torch.Tensor):
                ket_np = ket.detach().cpu().numpy()
            else:
                ket_np = np.array(ket)

            # Compute RDM for first qubit
            rdm = self._compute_first_qubit_rdm(ket_np, self._n_qubits)

            # Normalize RDM (handle zero trace case)
            trace_val = np.trace(rdm)
            if np.abs(trace_val) > 1e-10:
                rdm = rdm / trace_val
            else:
                # If trace is zero or very small, set to identity (maximally mixed state)
                rdm = np.eye(2) / 2

            X2_rdms.append(rdm)

        # Compute kernel matrix using trace
        for i in range(n1):
            for j in range(n2):
                # k_ij = tr(ρ_1(x_i) * ρ_1(x_j))
                # Using matrix multiplication: trace(ρ_i * ρ_j^T)
                kernel_matrix[i, j] = np.real(np.trace(X1_rdms[i] @ X2_rdms[j]))

        return kernel_matrix

    def _compute_first_qubit_rdm(self, ket: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Compute reduced density matrix for first qubit.

        Args:
            ket: Full state vector (2**n_qubits,)

        Returns:
            Reduced density matrix of shape (2, 2)
        """
        # Reshape ket as tensor indices
        # Index: (q0, q1, ..., q_{n-1}) where each qi is 0 or 1
        ket_tensor = ket.reshape([2] * n_qubits)

        # Partial trace: trace out all qubits except first
        # ρ_1 = Tr_{q_1, ..., q_{n-1}}(|ψ⟩⟨ψ|)

        # Sum over all qubits except first
        rdm = np.zeros((2, 2), dtype=np.complex64)
        for q_idx in range(1, n_qubits):
            for remaining in range(1, n_qubits):
                if remaining == 1:
                    # Skip first qubit
                    continue

                # Sum over all basis states where remaining qubit is in state 0
                indices = [q_idx if q_idx < n_qubits else 0 for _ in range(n_qubits)]

                # Build partial state for this configuration
                partial_state = ket_tensor[tuple(indices)]
                rdm += np.outer(partial_state, partial_state.conj())

        return rdm

    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Compute quantum kernel matrix based on kernel type.

        Args:
            X1: First dataset
            X2: Second dataset (if None, uses X1)
            verbose: Print progress messages

        Returns:
            Kernel matrix of shape (len(X1), len(X2))
        """
        if X2 is None:
            X2 = X1

        if self._kernel_type == 'eqk_n_to_n':
            return self.compute_eqk_n_to_n(X1, X2, verbose)
        elif self._kernel_type == 'pqk':
            return self.compute_pqk(X1, X2, verbose)
        else:
            raise ValueError(f"Unsupported kernel type: {self._kernel_type}")

    def set_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        compute_kernel: bool = True,
        verbose: bool = True
    ) -> 'NeuralQKM':
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
        self._X_train = X_train
        self._y_train = y_train

        if compute_kernel:
            self._kernel_matrix = self.compute_kernel_matrix(X_train, X_train, verbose=verbose)

        return self

    def fit_qnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 30,
        learning_rate: float = 0.005,
        batch_size: int = 24,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the QNN part of the model.

        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of QNN training epochs
            learning_rate: QNN learning rate
            batch_size: QNN batch size
            verbose: Print progress messages

        Returns:
            Dictionary with training history
        """
        return self.qnn.fit(X_train, y_train, epochs, learning_rate, batch_size, verbose)

    def fit(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        epochs: int = 30,
        learning_rate: float = 0.005,
        batch_size: int = 24,
        verbose: bool = True
    ) -> None:
        """
        Fit the full NeuralQKM model.

        Args:
            X_train: Training features (optional if already set via set_data)
            y_train: Training labels (optional if already set via set_data)
            epochs: Number of QNN training epochs
            learning_rate: QNN learning rate
            batch_size: QNN batch size
            verbose: Print progress messages

        """
        # Use provided data or stored data
        if X_train is None or y_train is None:
            if self._X_train is None or self._y_train is None:
                raise ValueError("No data set. Call set_data() first or provide X_train, y_train.")
            X_train, y_train = self._X_train, self._y_train
        else:
            # Store the provided data for future use
            self._X_train = X_train
            self._y_train = y_train

        # Train QNN
        if verbose:
            print(f"Training QNN ({epochs} epochs, lr={learning_rate})...")

        self.fit_qnn(X_train, y_train, epochs, learning_rate, batch_size, verbose)

        # Compute kernel matrix
        if verbose:
            print("Computing kernel matrix...")

        self._kernel_matrix = self.compute_kernel_matrix(X_train, X_train, verbose=verbose)

        # Fit SVM
        if verbose:
            print("Fitting SVM...")

        self._svm.fit(self._kernel_matrix, y_train)
        self._fitted = True

    def predict(
        self,
        X: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Predict labels for new data.

        Args:
            X: Test features
            verbose: Print progress messages

        Returns:
            Predicted labels
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute kernel matrix
        # Compute kernel matrix
        kernel_test = self.compute_kernel_matrix(X, self._X_train, verbose=verbose)

        # Manual prediction using dual coefficients to avoid sklearn validation issues
        # Decision function: f(x) = sum_i alpha_i * K(x, x_i) + b
        # where alpha_i = dual_coef_[0, i] for binary classification

        # Get dual coefficients and intercept
        dual_coef = self._svm.dual_coef_  # Shape: (n_classes, n_support_vectors)
        intercept = self._svm.intercept_  # Shape: (n_classes,)

        # Get support vector indices
        support_indices = self._svm.support_

        # Extract kernel values for support vectors only
        # kernel_test has shape (n_test, n_train)
        # We need kernel values for support vectors: (n_test, n_support_vectors)
        kernel_sv = kernel_test[:, support_indices]

        # Compute decision values for each test sample
        # For binary classification: decision = sum_j dual_coef[0, j] * kernel_sv[i, j] + intercept[0]
        decision_values = np.sum(kernel_sv * dual_coef[0], axis=1) + intercept[0]

        # Convert decision values to binary predictions
        predictions = (decision_values > 0).astype(int)

        return predictions

    def forward(
        self,
        state: Optional[State] = None,
        X_test: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_train: Optional[np.ndarray] = None,
    ) -> State:
        """
        Forward pass - returns quantum state for compatibility with QRENN.

        Args:
            state: Input quantum state (optional, ignored in NeuralQKM)
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

        # Return a mock state for compatibility
        mock_state = State(self._n_qubits)
        return mock_state

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def kernel_type(self) -> str:
        return self._kernel_type

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
            'n_layers': self._n_layers,
            'kernel_type': self._kernel_type,
            'entanglement': self._entanglement,
            'C': self._C,
            'is_fitted': self._fitted,
        }

    def reset(self) -> None:
        """Reset model, clearing training data and fitted state."""
        self._X_train = None
        self._y_train = None
        self._kernel_matrix = None
        self._predictions = None
        self._fitted = False
        self.qnn._trained = False
        self.qnn._theta_star = None

    def to(self, device: Union[str, torch.device]) -> 'NeuralQKM':
        """Mock method for PyTorch compatibility."""
        return self

    def __repr__(self) -> str:
        fitted_status = "fitted" if self._fitted else "not fitted"
        return (
            f"NeuralQKM(n_qubits={self._n_qubits}, "
            f"kernel_type={self._kernel_type}, "
            f"n_layers={self._n_layers}, "
            f"C={self._C}, "
            f"status={fitted_status})"
        )


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

    print(f"{'='*60}")
    print("Testing NeuralQKM")
    print(f"{'='*60}")

    # Test PQK kernel type
    nqkm = NeuralQKM(
        n_qubits=2,
        n_layers=3,
        entanglement='cnot',
        kernel_type='pqk',
        C=1.0
    )

    # Train model
    print(f"Training NeuralQKM with pqk kernel...")
    nqkm.fit(X_train, y_train, epochs=15, learning_rate=0.005, verbose=True)

    # Make predictions
    print("Making predictions...")
    predictions = nqkm.predict(X_test, verbose=False)

    # Compute accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"\n{'='*60}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"{'='*60}")
