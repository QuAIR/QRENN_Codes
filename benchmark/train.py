import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any, List, Union
from quairkit.database import z
from quairkit.qinfo import NKron


class Trainer:
    """
    A general trainer class for Quantum Neural Networks (QNNs) for binary classification.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        batch_size: Optional[int] = None,
        iterations: int = 2000,
        device: str = 'cpu',
        optimizer: str = 'adam',
        scheduler: Optional[str] = 'plateau',
        scheduler_params: Optional[Dict[str, Any]] = None,
        threshold: float = 0.0,
        trainable_qubits: int = 1,
        ancilla_qubits: int = 7,
        measurement: Optional[torch.Tensor] = None,
    ):
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._iterations = iterations
        self._device = torch.device(device) if torch.cuda.is_available() or 'cpu' in device else torch.device('cpu')
        self._optimizer_type = optimizer.lower()
        self._scheduler_type = scheduler
        self._scheduler_params = scheduler_params or {'factor': 0.5, 'mode': 'min'}
        self._threshold = threshold
        self._trainable_qubits = trainable_qubits
        self._ancilla_qubits = ancilla_qubits
        self._measurement = measurement
        self._history: Dict[str, List[float]] = {}
    
    # Properties
    @property
    def learning_rate(self) -> float:
        return self._learning_rate
    
    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size
    
    @property
    def iterations(self) -> int:
        return self._iterations
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def threshold(self) -> float:
        return self._threshold
    
    @property
    def trainable_qubits(self) -> int:
        return self._trainable_qubits
    
    @property
    def ancilla_qubits(self) -> int:
        return self._ancilla_qubits
    
    @property
    def measurement(self) -> Optional[torch.Tensor]:
        return self._measurement
    
    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history.copy()
    
    def _default_measurement(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Default measurement operator based on Z operators.
        
        M = |0><0| ⊗ I  for label 1
        M = |1><1| ⊗ I  for label 0
        
        Args:
            labels: Binary labels tensor
            
        Returns:
            Measurement operator tensor
        """
        
        z_tensor = NKron(*[z() for _ in range(self._trainable_qubits)])
        identity_train = torch.eye(2**self._trainable_qubits, dtype=torch.complex64, device=self._device)
        identity_ancilla = torch.eye(2**self._ancilla_qubits, dtype=torch.complex64, device=self._device)
        
        proj_0 = (identity_train + z_tensor) / 2  # |0><0| for label 1
        proj_1 = (identity_train - z_tensor) / 2  # |1><1| for label 0
        
        M = torch.stack([proj_0 if b == 1 else proj_1 for b in labels])
        M = NKron(M, identity_ancilla)
        
        return M
    
    def get_measurement(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get measurement operator for given labels.
        Uses custom measurement if set, otherwise uses default.
        
        Args:
            labels: Binary labels tensor
            
        Returns:
            Measurement operator tensor
        """
        if self._measurement is not None:
            return self._measurement
        return self._default_measurement(labels)
    
    def _create_optimizer(self, params) -> torch.optim.Optimizer:
        if self._optimizer_type == 'adam':
            return torch.optim.Adam(params, lr=self._learning_rate)
        elif self._optimizer_type == 'adamw':
            return torch.optim.AdamW(params, lr=self._learning_rate)
        elif self._optimizer_type == 'sgd':
            return torch.optim.SGD(params, lr=self._learning_rate)
        else:
            return torch.optim.Adam(params, lr=self._learning_rate)
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        if self._scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=self._scheduler_params.get('mode', 'min'),
                factor=self._scheduler_params.get('factor', 0.5)
            )
        elif self._scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self._scheduler_params.get('step_size', 10),
                gamma=self._scheduler_params.get('gamma', 0.1)
            )
        elif self._scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self._scheduler_params.get('T_max', self._iterations)
            )
        return None
    
    def _default_loss(self, model, labels: torch.Tensor) -> torch.Tensor:
        """
        Default loss function: 1 - mean(Tr(rho @ M)) 
        """
        from quairkit.qinfo import trace
        
        M = self.get_measurement(labels)
        rho = model().density_matrix
        return 1 - torch.mean(trace(rho @ M).real)
    
    def _default_predict(self, model) -> torch.Tensor:
        """
        Default prediction function using Z expectation value.
        """
        import quairkit as qkit
        
        z0 = qkit.Hamiltonian([[1.0, ",".join([f"Z{i}" for i in range(self._trainable_qubits)])]])
        return model().expec_val(z0).real
    
    def train(
        self,
        model,
        labels: torch.Tensor,
        loss_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
        params: Optional[list] = None,
        lr_threshold: float = 2e-8,
        print_interval: int = 100,
    ) -> torch.Tensor:
        """
        Train the model for binary classification.
        
        Args:
            model: The QNN model/circuit to train
            labels: Binary labels (0 or 1)
            loss_fn: Custom loss function with signature loss_fn(model, labels) -> loss tensor
            predict_fn: Custom prediction function with signature predict_fn(model) -> predictions
            params: Model parameters. If None, uses model.parameters()
            lr_threshold: Stop training when learning rate falls below this
            print_interval: Print loss every N iterations
            
        Returns:
            Trained model parameters
        """
        if params is None:
            params = model.parameters()
        
        if loss_fn is None:
            loss_fn = self._default_loss
        
        if predict_fn is None:
            predict_fn = self._default_predict
        
        opt = self._create_optimizer(params)
        scheduler = self._create_scheduler(opt)
        
        self._history = {'loss': [], 'lr': [], 'accuracy': []}
        
        iterations = self._iterations
        
        print("Training:")
        i = 0
        while i < iterations:
            opt.zero_grad()
            
            loss = loss_fn(model, labels)
            loss.backward()
            opt.step()
            
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                else:
                    scheduler.step()
            
            loss_val = loss.item()
            current_lr = opt.param_groups[0]['lr']
            
            self._history['loss'].append(loss_val)
            self._history['lr'].append(current_lr)
            
            if current_lr < lr_threshold:
                print(f"Stopping: learning rate {current_lr:.2E} below threshold {lr_threshold:.2E}")
                break
            
            if i > 5000:
                print(f"Stopping: iteration {i} exceeds threshold of 5000")
                break
            
            if i == iterations - 100 and current_lr >= lr_threshold:
                iterations += 1000
            
            if i % print_interval == 0 or i == iterations - 1:
                accuracy = self.compute_accuracy(model, labels, predict_fn)
                self._history['accuracy'].append(accuracy)
                print(f"iter: {i}, loss: {loss_val:.8f}, accuracy: {accuracy:.4f}, lr: {current_lr:.2E}")
            
            i += 1
        
        final_accuracy = self.compute_accuracy(model, labels, predict_fn)
        print(f"Training accuracy: {final_accuracy:.4f}")

        # Return trained parameters as concatenated tensor
        return torch.cat([p.flatten() for p in model.parameters()]).detach()
    
    def compute_accuracy(
        self,
        model,
        labels: torch.Tensor,
        predict_fn: Optional[Callable] = None,
    ) -> float:
        """
        Compute binary classification accuracy.
        """
        if predict_fn is None:
            predict_fn = self._default_predict
        
        predictions = self.predict(model, predict_fn)
        pred_labels = torch.zeros_like(predictions)
        pred_labels[predictions >= self._threshold] = 1
        accuracy = 1 - torch.mean(torch.abs(labels - pred_labels).float())
        return accuracy.item()
    
    def predict(
        self,
        model,
        predict_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Generate predictions.
        """
        if predict_fn is None:
            predict_fn = self._default_predict
        
        with torch.no_grad():
            return predict_fn(model)
    
    def predict_labels(
        self,
        model,
        predict_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Generate predicted binary labels.
        """
        predictions = self.predict(model, predict_fn)
        pred_labels = torch.zeros_like(predictions)
        pred_labels[predictions >= self._threshold] = 1
        return pred_labels
    
    def evaluate(
        self,
        model,
        labels: torch.Tensor,
        predict_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        """
        if predict_fn is None:
            predict_fn = self._default_predict
        
        predictions = self.predict(model, predict_fn)
        pred_labels = torch.zeros_like(predictions)
        pred_labels[predictions >= self._threshold] = 1
        accuracy = 1 - torch.mean(torch.abs(labels - pred_labels).float())
        
        return {
            'accuracy': accuracy.item(),
            'predictions': predictions,
            'pred_labels': pred_labels
        }
    
    def save_checkpoint(self, data: Dict[str, Any], path: str) -> None:
        """Save checkpoint to file."""
        data['history'] = self._history
        torch.save(data, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint from file."""
        data = torch.load(path, map_location=self._device)
        if 'history' in data:
            self._history = data['history']
        return data
    
    def __repr__(self) -> str:
        return (
            f"Trainer(learning_rate={self._learning_rate}, "
            f"iterations={self._iterations}, "
            f"device='{self._device}', "
            f"trainable_qubits={self._trainable_qubits}, "
            f"ancilla_qubits={self._ancilla_qubits}, "
            f"threshold={self._threshold})"
        )