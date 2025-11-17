import time
import math
import torch
import gc
import os
import numpy as np
from datetime import date

import quairkit as qkit
from tqdm import tqdm
from typing import List, Tuple
from quairkit.circuit import Circuit
from quairkit.core.state.backend import Hamiltonian
from quairkit.database import *
from quairkit.qinfo import NKron, trace

qkit.set_dtype('complex128')
NUMPY_DTYPE = 'complex128'


# def generate_hermitian(n:int, H_j:torch.Tensor, H_h:torch.Tensor) -> torch.Tensor:

#     return sum([NKron(torch.eye(2**i), H_j, torch.eye(2 ** (n - i - 3))) for i in range(n - 2)]
#             ), sum([NKron(torch.eye(2**i), H_h, torch.eye(2 ** (n - i - 2))) for i in range(n - 1)])


def save_array(array, folder, filename):
    """
    Save a NumPy array to a specified folder with a given filename.
    If the folder does not exist, it is created.
    If the filename already exists, a counter is appended to the filename.

    Parameters:
    - array (np.ndarray): The NumPy array to save.
    - folder (str): The name of the target folder.
    - filename (str): The desired filename (with extension, e.g., 'data.npy').

    Returns:
    - saved_path (str): The path where the array was saved.
    """
    # Ensure the folder exists; if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created.")

    # Split filename into name and extension
    name, ext = os.path.splitext(filename)
    if not ext:
        # Default to .npy if no extension provided
        ext = '.npy'
        filename = name + ext

    # Initialize counter
    counter = 1
    # Construct the full path
    full_path = os.path.join(folder, filename)

    # Check if file exists and append counter if necessary
    while os.path.exists(full_path):
        new_filename = f"{name}_({counter}){ext}"
        full_path = os.path.join(folder, new_filename)
        counter += 1

    # Save the array
    try:
        np.save(full_path, array)
        print(f"Array saved to '{full_path}'.")
    except Exception as e:
        print(f"An error occurred while saving the array: {e}")
        return None

    return full_path


def qrenn_cir(data_generators:torch.Tensor,
             trainable_qubits:int,
             V_number_qubits:int,
             layers:int) -> Circuit:

    cir = Circuit(trainable_qubits + V_number_qubits)
    
    for _ in range(layers):

        cir.u3(qubits_idx=list(range(trainable_qubits)))
        
        cir.control_oracle(data_generators, system_idx = [list(range(trainable_qubits))] + 
                list(range(trainable_qubits, V_number_qubits + trainable_qubits)))

    cir.u3(qubits_idx=list(range(trainable_qubits)))

    return cir


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


# def lossfunction(v:qkit.State, num_labels:int, m:torch.Tensor):
#     return torch.mean( - num_labels * torch.abs(trace(m @ v.density_matrix)))

# def lossfunction(v:qkit.State, num_labels:int, m:torch.Tensor):
#     return torch.norm(trace(m @ v.density_matrix).squeeze() - num_labels, p=2) / len(num_labels)


def lossfunction(v:qkit.State, num_labels:int):
    rho = v.trace(list(range(trainable_qubits, trainable_qubits+V_number_qubits)))
    rho = rho.density_matrix
    loss = 0.0
    for j in range(len(num_labels)):
        if num_labels[j] == -1:
            loss += torch.trace(rho[j] @ projector_minus)
        else:
            loss += torch.trace(rho[j] @ projector_plus)
    return -torch.real(loss) / trained_data


def gradient_sampling(n_sample:int,
                u:torch.Tensor,
                y_labels:torch.Tensor,
                trainable_qubits:int,
                V_number_qubits:int,
                slot:int):

    current_date = date.today()

    h_train = NKron(*[z() for _ in range(trainable_qubits)])
    h_data = NKron(*[torch.eye(2, dtype=torch.complex128) for _ in range(V_number_qubits)])
    m = NKron(h_train, h_data)

    loss_list, grad_list = [], []
    
    # initialize the model
    for _ in range(n_sample):
        cir = qrenn_cir(u, trainable_qubits, V_number_qubits, layers=slot)
        opt = torch.optim.Adam(lr=0.1, params=cir.parameters())
        opt.zero_grad()
        loss = lossfunction(cir(), y_labels)
        loss.backward()  # compute gradients
        opt.step()
        loss = loss.detach()
        grad = torch.from_numpy(cir.grad).detach()
        loss_list.append(loss)
        grad_list.append(grad)
        del cir, loss, grad
        gc.collect()
        torch.cuda.empty_cache()
        if _ % 100 == 0:
            print(f"{_}th iteration.")
    
    # save_array(np.array(loss_list), f'grad_exp@{current_date}_diag', 
    #                  f'exp_value_n_process_{n_process}_n_data_{n_data}_slot_{t_slot}' 
    #                  )
    save_array(np.array(grad_list), f'grad_exp@{current_date}_invol', 
                     f'grad_value_n_process_{trainable_qubits}_n_data_{V_number_qubits}_slot_{slot}'
                     )

    # torch.save({'var_a': torch.stack(loss_list).detach(),'var_b': torch.stack(grad_list).detach()},
    #            f'./QRENN_grad_exp/grad_trainable_qubits_{trainable_qubits}_V_number_qubits_{V_number_qubits}_layers_{slot}.pt')
    
    return 


LR = 0.1
sample = 500
trainable_qubits = 1
V_number_qubits = 2
layers = 20
trained_data = 100
Z_tensor = NKron(*[z() for _ in range(trainable_qubits)])
projector_plus = 0.5 * (torch.eye(2 ** trainable_qubits) + Z_tensor)
projector_minus = 0.5 * (torch.eye(2 ** trainable_qubits) - Z_tensor)

labels, data_raw = uhermitian_data_generation(2**V_number_qubits, trained_data)

gc.collect()
torch.cuda.empty_cache()

gradient_sampling(n_sample = sample,
                u = data_raw,
                y_labels= labels,
                trainable_qubits = trainable_qubits,
                V_number_qubits = V_number_qubits,
                slot = layers)
    