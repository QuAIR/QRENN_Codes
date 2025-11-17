import time
import torch
import math
import numpy as np
import quairkit as qkit

from typing import List, Tuple
from quairkit.database.state import zero_state, ghz_state
from quairkit.database.matrix import *
from quairkit.database import haar_orthogonal, haar_unitary, pauli_group, random_clifford
from quairkit.core.state import to_state
from quairkit.circuit import Circuit
from quairkit.qinfo import dagger, NKron

qkit.set_dtype('complex128')
NUMPY_DTYPE = 'complex128'

# %%
def qrenn_cir(data_generators:torch.Tensor,
             trainable_qubits:int, 
             V_number_qubits:int, 
             layers:int, 
             is_conj:bool = False) -> Circuit:
    
    cir = Circuit(trainable_qubits + V_number_qubits)
    
    cir.h(qubits_idx=list(range(trainable_qubits, trainable_qubits+V_number_qubits)))
    for _ in range(layers):
        cir.universal_qudits(qubits_idx=list(range(trainable_qubits)))
        if is_conj:
            if _ % 2 == 0:
                cir.control_oracle(data_generators, system_idx = [list(range(trainable_qubits))] + 
                                    list(range(trainable_qubits, V_number_qubits +trainable_qubits)))
            else:
                cir.control_oracle(dagger(data_generators), system_idx = [list(range(trainable_qubits))] + 
                                    list(range(trainable_qubits, V_number_qubits +trainable_qubits)))
        else:
            cir.control_oracle(data_generators, system_idx = [list(range(trainable_qubits))] + 
                                    list(range(trainable_qubits, V_number_qubits +trainable_qubits)))
    # cir.u3(qubits_idx=list(range(trainable_qubits)))
    cir.universal_qudits(qubits_idx=list(range(trainable_qubits)))

    return cir


# %%
def mse_loss(v:qkit.State, num_labels:int, m:torch.Tensor):
    return torch.norm((v.bra @ m @ v.ket).squeeze() - num_labels, p=2) / len(num_labels)

# Training
def train_model_convergence(u:torch.Tensor,
                y_labels:torch.Tensor,
                trainable_qubits:int, 
                ITR:int = 100, 
                LR:float = 0.1, 
                slot:int = 10):
    
    assert u.shape[0] == y_labels.shape[0], "Number of Hams and y_labels should be the same"
    
    dim_h = u[0].shape[0]
    V_number_qubits = int(np.log2(dim_h))
    
    h_train = NKron(*[z() for _ in range(trainable_qubits)])
    h_data = NKron(*[torch.eye(2, dtype=torch.complex128) for _ in range(V_number_qubits)])
    m = NKron(h_train, h_data)
    
    loss_list, time_list = [], []
    
    # initialize the model
    cir = qrenn_cir(u, trainable_qubits, V_number_qubits, layers=slot)
    
    # cir is a Circuit type
    opt = torch.optim.Adam(lr=LR, params=cir.parameters())

    # activate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5)

    _ = 0
    while _ < ITR:

        start_time = time.time()
        opt.zero_grad()

        loss = mse_loss(cir(), y_labels, m)

        loss.backward()  # compute gradients
        opt.step()  # update parameters
        scheduler.step(loss)  # activate scheduler

        loss = loss.item()
        loss_list.append(loss)
        time_list.append(time.time() - start_time)

        if scheduler.get_last_lr()[0]<2e-8:
            print(
                f"iter: {_}, loss: {loss:.8f}, lr: {scheduler.get_last_lr()[0]:.2E}, avg_time: {np.mean(time_list):.4f}s"
            )
            time_list = []
            break

        if _ % 500 == 0 or _ == ITR - 1:
            print(
                f"iter: {_}, loss: {loss:.8f}, lr: {scheduler.get_last_lr()[0]:.2E}, avg_time: {np.mean(time_list):.4f}s"
            )
            time_list = []
        _ += 1
    return cir, loss_list


def predict_model(u:torch.Tensor,
                  trainable_qubits:int, 
                  params:torch.Tensor, 
                  slot:int,
                  scale:float=1.0):
    
    z0 = qkit.Hamiltonian([[scale, ",".join([f"Z{i}" for i in range(trainable_qubits)])]])
    
    dim_h = u[0].shape[0]
    V_number_qubits = int(np.log2(dim_h))
    
    # initialize the model
    cir = qrenn_cir(u, trainable_qubits, V_number_qubits, layers=slot)
    cir.update_param(params)

    return cir().expec_val(z0)

# %%
# Data generation
def orthogonal_data_generation(dim:int, num_samples:int) -> Tuple[torch.Tensor, torch.Tensor]:
    sour = np.random.randint(0, 2, num_samples)
    y_labels = np.where(sour == 0, -1, 1)
    data_U = np.zeros((num_samples, dim, dim), dtype=NUMPY_DTYPE)
    for y in range(num_samples):
        if sour[y]:
            data_U[y, :, :] = haar_orthogonal(dim)
        else:
            data_U[y, :, :] = haar_unitary(dim)
    
    return torch.tensor(y_labels), torch.tensor(data_U)


def uhermitian_data_generation(dim:int, num_samples:int) -> Tuple[torch.Tensor, torch.Tensor]:
    sour = np.random.randint(0, 2, num_samples)
    y_labels = np.where(sour == 0, -1, 1)
    data_U = np.zeros((num_samples, dim, dim), dtype=NUMPY_DTYPE)
    for y in range(num_samples):
        if sour[y]:
             # Randomly choose eigenvalues of +1 or -1
            eigenvalues = np.random.choice([1.0, -1.0], size=dim)
            D = np.diag(eigenvalues)
            
            # Generate a random unitary matrix U
            U = haar_unitary(dim)
            data_U[y, :, :] = U @ D @ U.conj().T

        else:
            data_U[y, :, :] = haar_unitary(dim)
    
    return torch.tensor(y_labels), torch.tensor(data_U)


def random_diag_data_generation(dim:int, num_samples:int) -> Tuple[torch.Tensor, torch.Tensor]:
    sour = np.random.randint(0, 2, num_samples)
    y_labels = np.where(sour == 0, -1, 1)
    data_U = np.zeros((num_samples, dim, dim), dtype=NUMPY_DTYPE)
    for y in range(num_samples):
        if sour[y]:
             # Randomly choose eigenvalues of +1 or -1
            eigenphase = 1*np.pi*np.random.random(size=dim)
            data_U[y, :, :] = np.diag(np.exp(1j*eigenphase))
        else:
            data_U[y, :, :] = haar_unitary(dim)
    
    return torch.tensor(y_labels), torch.tensor(data_U)


def random_pauli_data_generation(dim:int, num_samples:int) -> Tuple[torch.Tensor, torch.Tensor]:
    sour = np.random.randint(0, 2, num_samples)
    n = int(np.log2(dim))
    y_labels = np.where(sour == 0, -1, 1)
    data_U = np.zeros((num_samples, dim, dim), dtype=NUMPY_DTYPE)
    for y in range(num_samples):
        if sour[y]:
            data_U[y, :, :] = pauli_group(n)[np.random.randint(1, 4**n)]
        else:
            data_U[y, :, :] = haar_unitary(dim)
    
    return torch.tensor(y_labels), torch.tensor(data_U)


def random_clifford_data_generation(dim:int, num_samples:int) -> Tuple[torch.Tensor, torch.Tensor]:
    sour = np.random.randint(0, 2, num_samples)
    y_labels = np.where(sour == 0, -1, 1)
    data_U = np.zeros((num_samples, dim, dim), dtype=NUMPY_DTYPE)
    for y in range(num_samples):
        if sour[y]:
            data_U[y, :, :] = random_clifford(int(np.log2(dim)))
        else:
            data_U[y, :, :] = haar_unitary(dim)
    
    return torch.tensor(y_labels), torch.tensor(data_U)


def random_local_data_generation(dim:int, num_samples:int) -> Tuple[torch.Tensor, torch.Tensor]:
    sour = np.random.randint(0, 2, num_samples)
    y_labels = np.where(sour == 0, -1, 1)
    data_U = np.zeros((num_samples, dim, dim), dtype=NUMPY_DTYPE)
    for y in range(num_samples):
        if sour[y]:
            data_U[y, :, :] = NKron(*[haar_unitary(2) for _ in range(int(np.log2(dim)))])
        else:
            data_U[y, :, :] = haar_unitary(dim)
    
    return torch.tensor(y_labels), torch.tensor(data_U)

# %%
trainable_qubits = 2
V_number_qubits = 7

# data generation
train_samples = 100
test_samples = 500

# labels, data_raw = orthogonal_data_generation(2**V_number_qubits, train_samples+test_samples)
# labels, data_raw = random_pauli_data_generation(2**V_number_qubits, train_samples+test_samples)
# labels, data_raw = uhermitian_data_generation(2**V_number_qubits, train_samples+test_samples)
# labels, data_raw = random_clifford_data_generation(2**V_number_qubits, train_samples+test_samples)
# labels, data_raw = random_local_data_generation(2**V_number_qubits, train_samples+test_samples)
labels, data_raw = random_diag_data_generation(2**V_number_qubits, train_samples+test_samples)

# %%
slot = 2
LR = 0.1
ITR = 2000
num_exp = 10

results = []
for exp in range(num_exp):
    
    # Create a shuffled copy of psi states
    shuffled_indices = np.random.permutation(labels.shape[0])
    data_raw = data_raw[shuffled_indices]
    labels = labels[shuffled_indices]

    y_train, x_train = labels[:train_samples], data_raw[:train_samples]
    y_test, x_test = labels[train_samples:], data_raw[train_samples:]
    
    cir, loss_ = train_model_convergence(u=x_train,
                                y_labels=y_train,
                                trainable_qubits=trainable_qubits, 
                                ITR = ITR, 
                                LR = LR, 
                                slot = slot)

    y_predict = predict_model(x_test,
                                trainable_qubits=trainable_qubits, 
                                params=cir.param, 
                                slot=slot)
    diff = np.where(y_predict.detach().numpy() >= 0, 1, -1) - y_test.numpy()
    results.append(np.sum(diff == 0) / diff.shape[0])
    
print(f'Average accuracy: {np.mean(results)}, std: {np.std(results)}')
    

