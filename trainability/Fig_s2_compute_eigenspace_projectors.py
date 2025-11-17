import torch
import numpy as np
import gc
from tqdm import tqdm
import time
import quairkit as qkit

from quairkit.circuit import Circuit



qkit.set_dtype('complex128')


def compute_eigenspace_projectors(H:np.ndarray, tol=1e-6):
    """
    Compute the projection operators onto the eigenspaces of a Hermitian matrix H.

    Parameters:
    - H (ndarray): A Hermitian matrix.
    - tol (float): Tolerance for grouping nearly equal eigenvalues.

    Returns:
    - projectors (dict): A dictionary mapping each distinct eigenvalue
                         to its corresponding projector (ndarray).
    """

    # Diagonalize H; np.linalg.eigh is appropriate for Hermitian matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Group indices of eigenvectors corresponding to (nearly) the same eigenvalue.
    projectors = {}
    # We'll keep track of "unique" eigenvalues up to the tolerance tol.
    unique_eigvals = []
    eig_indices = {}

    for idx, eig in enumerate(eigenvalues):
        matched = False
        for unique in unique_eigvals:
            if abs(eig - unique) < tol:
                matched = True
                eig_indices[unique].append(idx)
                break
        if not matched:
            unique_eigvals.append(eig)
            eig_indices[eig] = [idx]

    # Compute the projector for each unique eigenvalue.
    for eig in unique_eigvals:
        indices = eig_indices[eig]
        # Extract the eigenvectors (each column is an eigenvector)
        vecs = eigenvectors[:, indices]
        # Construct the projector onto the subspace: P = sum_i |v_i><v_i|
        projector = vecs @ vecs.conj().T
        projectors[eig] = projector

    return projectors

def joint_eigen_overlap(H:np.ndarray, rho_in:np.ndarray):
    '''Compute the joint eigenspace overlap of each individual tensor be given
    '''
    res = []
    for j in tqdm(range(H.shape[0])):
        val = 0
        for proj in compute_eigenspace_projectors(H[j]).values():
            val += np.trace(proj @ rho_in).real ** 2
        res.append(val)
    return res


test_data = 100


for V_number_qubits in range(3,9):
    data = torch.load(
        f'./SPT_exp/V_number_qubits{V_number_qubits}_data{test_data}.pt',
        map_location=torch.device('cpu')
    )
    Hams = data["var_a"].numpy()
    lambda_ = data["var_b"].numpy()

    cir = Circuit(V_number_qubits)
    cir.ry(qubits_idx=list(range(V_number_qubits)),
            param=torch.tensor([0]*(V_number_qubits))*torch.pi)
    rho_in = cir().density_matrix.detach().numpy()

    del data, cir
    gc.collect()
    torch.cuda.empty_cache()


    # labels = data["var_b"].numpy()

    result = joint_eigen_overlap(Hams, rho_in)

    # print(np.mean(result), result)

    np.savez(f'./SPT_exp/compute_eigenspace_projector_sreuslt_V_number_qubits{V_number_qubits}.npz',
         var_a=result,
         var_b=lambda_)
