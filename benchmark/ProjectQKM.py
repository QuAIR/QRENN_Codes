"""
Projected Quantum Kernel Model (ProjectQKM)
=========================================
A QuAIRKit implementation of the Projected Quantum Kernel (PQK) model.

This module provides a flexible implementation of the Projected Quantum Kernel Model
based on the TensorFlow Quantum tutorial "Quantum data".

The PQK approach uses:
1. A single qubit rotation wall (X, Y, Z rotations on each qubit)
2. Parametrized entanglement V(theta) using Pauli strings
3. Trotterized evolution for repeated application
4. 1-RDM (Reduced Density Matrix) feature extraction from expectation values

Reference: TensorFlow Quantum "Quantum data" tutorial
https://www.tensorflow.org/quantum/tutorials/quantum_data

Author: QuAIRKit implementation
"""

import numpy as np
import torch
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

from quairkit import Circuit, State
from quairkit.database import *


# =============================================================================
# Single Qubit Rotation Wall
# =============================================================================

class SingleQubitRotationWall:
    """
    Single qubit rotation wall that applies X, Y, Z rotations to each qubit.

    For each qubit i, applies: R_x(r_i[0]) * R_y(r_i[1]) * R_z(r_i[2])
    where r_i are the rotation angles for that qubit.
    """

    def __init__(self, n_qubits: int, random_range: Tuple[float, float] = (-2, 2)):
        """
        Initialize the single qubit rotation wall.

        Args:
            n_qubits: Number of qubits
            random_range: Range for random rotation angles (min, max)
        """
        self.n_qubits = n_qubits
        self.random_range = random_range
        self._rotations = None

    def generate_random_rotations(self) -> np.ndarray:
        """Generate random rotation angles for each qubit."""
        return np.random.uniform(
            self.random_range[0],
            self.random_range[1],
            size=(self.n_qubits, 3)
        )

    def set_rotations(self, rotations: Optional[np.ndarray] = None) -> 'SingleQubitRotationWall':
        """
        Set rotation angles for the wall.

        Args:
            rotations: Rotation angles of shape (n_qubits, 3). If None, generates random angles.

        Returns:
            self (for chain calling)
        """
        if rotations is None:
            self._rotations = self.generate_random_rotations()
        else:
            if rotations.shape != (self.n_qubits, 3):
                raise ValueError(f"Rotations must have shape ({self.n_qubits}, 3), got {rotations.shape}")
            self._rotations = rotations
        return self

    def apply(self, circuit: Circuit, qubits_idx: List[int]) -> Circuit:
        """
        Apply the rotation wall to a circuit.

        Args:
            circuit: QuAIRKit Circuit to modify
            qubits_idx: List of qubit indices to apply rotations to

        Returns:
            The modified circuit
        """
        if self._rotations is None:
            self.set_rotations()

        for i, qubit in enumerate(qubits_idx):
            # Apply X, Y, Z rotations
            circuit.rx(qubits_idx=[qubit], param=self._rotations[i, 0])
            circuit.ry(qubits_idx=[qubit], param=self._rotations[i, 1])
            circuit.rz(qubits_idx=[qubit], param=self._rotations[i, 2])

        return circuit


# =============================================================================
# V(theta) Entanglement
# =============================================================================

class VThetaEvolution:
    """
    V(theta) entanglement using Pauli strings.

    For each pair of adjacent qubits (i, i+1), applies:
    exp(-i * theta * (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}))

    This is a Heisenberg-type coupling between neighboring qubits.
    """

    def __init__(self, n_qubits: int):
        """
        Initialize V(theta) evolution.

        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits
        self._theta_values = None

    def set_theta_values(self, theta_values: Optional[np.ndarray] = None,
                        use_classical_data: bool = False,
                        classical_data: Optional[np.ndarray] = None) -> 'VThetaEvolution':
        """
        Set theta values for the evolution.

        Args:
            theta_values: Explicit theta values for each qubit pair
            use_classical_data: Whether to use classical data for theta values
            classical_data: Classical data to convert to theta values

        Returns:
            self (for chain calling)
        """
        if use_classical_data and classical_data is not None:
            # Convert classical data to theta values
            # Use a subset of the data as theta values
            n_pairs = self.n_qubits - 1
            theta_values = classical_data[:n_pairs] if len(classical_data) >= n_pairs else \
                         np.pad(classical_data, (0, n_pairs - len(classical_data)))
            self._theta_values = theta_values
        elif theta_values is not None:
            if len(theta_values) != self.n_qubits - 1:
                raise ValueError(f"Theta values must have length {self.n_qubits - 1}, got {len(theta_values)}")
            self._theta_values = theta_values
        else:
            # Default: use small values
            self._theta_values = np.ones(self.n_qubits - 1) * 0.1

        return self

    def apply(self, circuit: Circuit, qubits_idx: List[int],
             theta_values: Optional[np.ndarray] = None) -> Circuit:
        """
        Apply V(theta) evolution to a circuit.

        Args:
            circuit: QuAIRKit Circuit to modify
            qubits_idx: List of qubit indices
            theta_values: Theta values to use (overrides stored values if provided)

        Returns:
            The modified circuit
        """
        if theta_values is None:
            theta_values = self._theta_values

        if theta_values is None:
            raise ValueError("Theta values not set. Call set_theta_values() first or pass theta_values.")

        # Apply Heisenberg-type coupling between adjacent qubits
        # Since quairkit doesn't have direct exp(i*P*P) gates, we decompose
        # into simpler operations
        for i in range(len(qubits_idx) - 1):
            theta = theta_values[i]
            q0, q1 = qubits_idx[i], qubits_idx[i + 1]

            # Decompose exp(-i * theta * (XX + YY + ZZ))
            # This is equivalent to -i * exp(-i * theta) SWAP up to a phase
            # For simplicity, we'll use RXX, RYY, RZZ gates if available
            # Otherwise, we approximate with CNOT rotations

            # Apply RXX rotation
            circuit.h(qubits_idx=[q0])
            circuit.cx(qubits_idx=[q0, q1])
            circuit.rz(qubits_idx=[q1], param=2 * theta)
            circuit.cx(qubits_idx=[q0, q1])
            circuit.h(qubits_idx=[q0])

            # Apply RYY rotation
            circuit.rx(qubits_idx=[q0], param=-np.pi / 2)
            circuit.rx(qubits_idx=[q1], param=-np.pi / 2)
            circuit.cx(qubits_idx=[q0, q1])
            circuit.rz(qubits_idx=[q1], param=2 * theta)
            circuit.cx(qubits_idx=[q0, q1])
            circuit.rx(qubits_idx=[q0], param=np.pi / 2)
            circuit.rx(qubits_idx=[q1], param=np.pi / 2)

            # Apply RZZ rotation
            circuit.cx(qubits_idx=[q0, q1])
            circuit.rz(qubits_idx=[q1], param=2 * theta)
            circuit.cx(qubits_idx=[q0, q1])

        return circuit


# =============================================================================
# PQK Feature Circuit
# =============================================================================

class PQKFeatureCircuit:
    """
    PQK (Projected Quantum Kernel) feature circuit.

    Builds the circuit:
    1. Apply single qubit rotation wall
    2. Apply V(theta) evolution repeated n_trotter times

    The final state can be used to extract 1-RDM features.
    """

    def __init__(
        self,
        n_qubits: int,
        n_trotter: int = 10,
        random_range: Tuple[float, float] = (-2, 2),
        use_classical_data: bool = False
    ):
        """
        Initialize PQK feature circuit.

        Args:
            n_qubits: Number of qubits
            n_trotter: Number of Trotter steps for V(theta) evolution
            random_range: Range for random rotation angles
            use_classical_data: Whether to use classical data for theta values
        """
        self.n_qubits = n_qubits
        self.n_trotter = n_trotter
        self.random_range = random_range
        self.use_classical_data = use_classical_data

        # Initialize components
        self.rotation_wall = SingleQubitRotationWall(n_qubits, random_range)
        self.v_theta = VThetaEvolution(n_qubits)
        self._classical_data = None

    def set_classical_data(self, data: np.ndarray) -> 'PQKFeatureCircuit':
        """
        Set classical data for encoding.

        Args:
            data: Classical data to encode

        Returns:
            self (for chain calling)
        """
        self._classical_data = data
        if self.use_classical_data:
            self.v_theta.set_theta_values(
                use_classical_data=True,
                classical_data=data
            )
        return self

    def build_circuit(self, classical_data: Optional[np.ndarray] = None) -> Circuit:
        """
        Build the PQK feature circuit.

        Args:
            classical_data: Classical data to encode (if not already set)

        Returns:
            QuAIRKit Circuit
        """
        if classical_data is not None:
            self.set_classical_data(classical_data)

        circuit = Circuit(self.n_qubits)
        qubits_idx = list(range(self.n_qubits))

        # Apply single qubit rotation wall
        self.rotation_wall.apply(circuit, qubits_idx)

        # Apply V(theta) evolution for n_trotter steps
        for _ in range(self.n_trotter):
            self.v_theta.apply(circuit, qubits_idx)

        return circuit


# =============================================================================
# 1-RDM Feature Extractor
# =============================================================================

class RDMFeatureExtractor:
    """
    Extract 1-RDM (Reduced Density Matrix) features from quantum states.

    For each qubit, computes the expectation values of X, Y, Z Pauli operators:
    - rdm[i][0] = <psi|X_i|psi>
    - rdm[i][1] = <psi|Y_i|psi>
    - rdm[i][2] = <psi|Z_i|psi>

    This gives a feature vector of shape (n_qubits, 3) for each state.
    """

    def __init__(self, n_qubits: int):
        """
        Initialize the RDM feature extractor.

        Args:
            n_qubits: Number of qubits
        """
        self.n_qubits = n_qubits

    def extract_single(self, state: State) -> np.ndarray:
        """
        Extract 1-RDM features from a single quantum state.

        Args:
            state: QuAIRKit State object

        Returns:
            RDM features of shape (n_qubits, 3)
        """
        features = np.zeros((self.n_qubits, 3))

        # Get the state vector
        ket = state.ket
        if isinstance(ket, torch.Tensor):
            ket_np = ket.detach().cpu().numpy()
        else:
            ket_np = np.array(ket)

        # For each qubit, compute X, Y, Z expectation values
        for i in range(self.n_qubits):
            # Build Pauli matrices for this qubit
            pauli_ops = self._build_pauli_kron(self.n_qubits, i)

            # Compute expectation values
            for p_idx, pauli in enumerate([pauli_ops[0], pauli_ops[1], pauli_ops[2]]):
                # <psi|P|psi> = psi^H @ P @ psi
                exp_val = np.vdot(ket_np, pauli @ ket_np)
                features[i, p_idx] = np.real(exp_val)

        return features

    def extract_batch(self, states: List[State]) -> np.ndarray:
        """
        Extract 1-RDM features from a batch of quantum states.

        Args:
            states: List of QuAIRKit State objects

        Returns:
            RDM features of shape (batch_size, n_qubits, 3)
        """
        batch_size = len(states)
        features = np.zeros((batch_size, self.n_qubits, 3))

        for i, state in enumerate(states):
            features[i] = self.extract_single(state)

        return features

    def _build_pauli_kron(self, n_qubits: int, target_qubit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the Kronecker product of Pauli matrices.

        Args:
            n_qubits: Total number of qubits
            target_qubit: Qubit to apply Pauli operator to

        Returns:
            Tuple of (X_op, Y_op, Z_op) matrices
        """
        # Pauli matrices
        X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
        I = np.eye(2, dtype=np.complex64)

        # Build operators for each qubit
        x_list, y_list, z_list = [], [], []
        for i in range(n_qubits):
            if i == target_qubit:
                x_list.append(X)
                y_list.append(Y)
                z_list.append(Z)
            else:
                x_list.append(I)
                y_list.append(I)
                z_list.append(I)

        # Kronecker product
        X_op = x_list[0]
        Y_op = y_list[0]
        Z_op = z_list[0]
        for i in range(1, n_qubits):
            X_op = np.kron(X_op, x_list[i])
            Y_op = np.kron(Y_op, y_list[i])
            Z_op = np.kron(Z_op, z_list[i])

        return X_op, Y_op, Z_op


# =============================================================================
# ProjectQKM Model
# =============================================================================

class ProjectQKM(torch.nn.Module):
    """
    Projected Quantum Kernel Model (ProjectQKM).

    This model implements the Projected Quantum Kernel (PQK) approach:
    1. Encode classical data using PQK feature circuits
    2. Extract 1-RDM features from the quantum states
    3. Use these features as input to a classical classifier

    The PQK circuit consists of:
    - Single qubit rotation wall (X, Y, Z rotations)
    - V(theta) evolution with Trotterization
    - Feature extraction via 1-RDM expectation values

    Usage:
        >>> pqkm = ProjectQKM(n_qubits=4, n_trotter=10)
        >>> features = pqkm.compute_features(X_train)
        >>> # Use features with classical classifier
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_trotter: int = 10,
        random_range: Tuple[float, float] = (-2, 2),
        use_classical_data: bool = False,
        n_classes: int = 2
    ):
        """
        Initialize ProjectQKM model.

        Args:
            n_qubits: Number of qubits
            n_trotter: Number of Trotter steps for V(theta) evolution
            random_range: Range for random rotation angles
            use_classical_data: Whether to use classical data for theta values
            n_classes: Number of output classes
        """
        super().__init__()

        self._n_qubits = n_qubits
        self._n_trotter = n_trotter
        self._random_range = random_range
        self._use_classical_data = use_classical_data
        self._n_classes = n_classes

        # Initialize PQK feature circuit
        self.pqk_circuit = PQKFeatureCircuit(
            n_qubits=n_qubits,
            n_trotter=n_trotter,
            random_range=random_range,
            use_classical_data=use_classical_data
        )

        # Initialize RDM feature extractor
        self.rdm_extractor = RDMFeatureExtractor(n_qubits=n_qubits)

        # Feature dimension (calculated, not stored as separate property)
        # feature_dim is a read-only property: n_qubits * 3

        # Build classical classifier (neural network)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(n_qubits * 3, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 16),
            torch.nn.Sigmoid(),
            torch.nn.Linear(16, n_classes)
        )

        # Storage for computed features
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._train_features: Optional[np.ndarray] = None
        self._predictions: Optional[np.ndarray] = None
        self._fitted = False

    def compute_single_feature(self, x: np.ndarray) -> np.ndarray:
        """
        Compute PQK features for a single data point.

        Args:
            x: Single data point

        Returns:
            1-RDM features of shape (n_qubits, 3)
        """
        # Build circuit with this data point
        circuit = self.pqk_circuit.build_circuit(classical_data=x)

        # Execute circuit
        state = circuit()

        # Extract features
        features = self.rdm_extractor.extract_single(state)

        return features

    def compute_features(self, X: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Compute PQK features for a batch of data points.

        Args:
            X: Batch of data points (n_samples, n_features)
            verbose: Print progress messages

        Returns:
            1-RDM features of shape (n_samples, n_qubits, 3)
        """
        n_samples = X.shape[0]
        features = np.zeros((n_samples, self._n_qubits, 3))

        if verbose:
            print(f"Computing PQK features for {n_samples} samples...")

        for i in range(n_samples):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{n_samples}")

            features[i] = self.compute_single_feature(X[i])

        return features

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.003, verbose: bool = True) -> None:
        """
        Train the classical classifier on PQK features.

        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            verbose: Print progress messages
        """
        # Store training data
        self._X_train = X_train
        self._y_train = y_train

        # Compute PQK features
        if verbose:
            print("Computing PQK features for training data...")
        train_features = self.compute_features(X_train, verbose=verbose)

        # Flatten features
        train_features_flat = train_features.reshape(train_features.shape[0], -1)
        self._train_features = train_features_flat

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(train_features_flat, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        # Define loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)

        # Training loop
        if verbose:
            print(f"\nTraining classifier for {epochs} epochs...")

        self.train()  # Set to training mode
        for epoch in range(epochs):
            # Forward pass
            outputs = self.classifier(X_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        self._fitted = True
        if verbose:
            print("Training complete!")

    def predict(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Predict labels for test data.

        Args:
            X_test: Test features
            verbose: Print progress messages

        Returns:
            Predicted labels
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute PQK features for test data
        if verbose:
            print("Computing PQK features for test data...")
        test_features = self.compute_features(X_test, verbose=verbose)

        # Flatten features
        test_features_flat = test_features.reshape(test_features.shape[0], -1)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(test_features_flat, dtype=torch.float32)

        # Get predictions
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            outputs = self.classifier(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        self._predictions = predicted.numpy()
        return self._predictions

    def predict_proba(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Predict class probabilities for test data.

        Args:
            X_test: Test features
            verbose: Print progress messages

        Returns:
            Class probabilities
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute PQK features for test data
        test_features = self.compute_features(X_test, verbose=verbose)

        # Flatten features
        test_features_flat = test_features.reshape(test_features.shape[0], -1)

        # Convert to PyTorch tensor
        X_tensor = torch.tensor(test_features_flat, dtype=torch.float32)

        # Get predictions
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            outputs = self.classifier(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.numpy()

    # PyTorch Module interface
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            X: Input features (must be numpy array for PQK computation)

        Returns:
            Output logits
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.array(X)

        # Compute PQK features
        features = self.compute_features(X_np, verbose=False)

        # Flatten and convert to tensor
        features_flat = features.reshape(features.shape[0], -1)
        X_tensor = torch.tensor(features_flat, dtype=torch.float32)

        # Pass through classifier
        return self.classifier(X_tensor)

    # Properties
    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_trotter(self) -> int:
        return self._n_trotter

    @property
    def feature_dim(self) -> int:
        return self._n_qubits * 3

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def predictions(self) -> Optional[np.ndarray]:
        return self._predictions

    def get_info(self) -> Dict[str, Any]:
        """Get model configuration info."""
        return {
            'n_qubits': self._n_qubits,
            'n_trotter': self._n_trotter,
            'feature_dim': self.feature_dim,
            'n_classes': self._n_classes,
            'is_fitted': self._fitted,
        }

    def reset(self) -> None:
        """Reset model, clearing training data and fitted state."""
        self._X_train = None
        self._y_train = None
        self._train_features = None
        self._predictions = None
        self._fitted = False

    def __repr__(self) -> str:
        fitted_status = "fitted" if self._fitted else "not fitted"
        return (
            f"ProjectQKM(n_qubits={self._n_qubits}, "
            f"n_trotter={self._n_trotter}, "
            f"feature_dim={self.feature_dim}, "
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
    print("Testing ProjectQKM")
    print(f"{'='*60}")

    # Initialize ProjectQKM
    eqkm = ProjectQKM(n_qubits=2, n_trotter=5, use_classical_data=True)

    print(f"\nModel: {eqkm}")
    print(f"Feature dimension: {eqkm.feature_dim}")

    # Train model
    print(f"\n{'='*60}")
    print("Training ProjectQKM...")
    print(f"{'='*60}")
    eqkm.fit(X_train, y_train, epochs=50, verbose=True)

    # Make predictions
    print(f"\n{'='*60}")
    print("Making predictions...")
    print(f"{'='*60}")
    predictions = eqkm.predict(X_test, verbose=True)

    # Compute accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"\n{'='*60}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"{'='*60}")
