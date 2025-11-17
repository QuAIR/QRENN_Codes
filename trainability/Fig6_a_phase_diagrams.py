import torch
import numpy as np
import quairkit as qkit
import matplotlib.pyplot as plt
from quairkit.database.matrix import *
from quairkit.database import x,z
from quairkit.qinfo import dagger, NKron

qkit.set_dtype('complex128')


def generate_hermitian(n:int, h_j:torch.Tensor, h_h1:torch.Tensor, h_h2:torch.Tensor,) -> torch.Tensor:
    
    return sum([NKron(torch.eye(2**i), h_j, torch.eye(2 ** (n - i - 3))) for i in range(n - 2)]
            ), sum([NKron(torch.eye(2**i), h_h1, torch.eye(2 ** (n - i - 1))) for i in range(n)]
                    ), sum([NKron(torch.eye(2**i), h_h2, torch.eye(2 ** (n - i - 2))) for i in range(n - 1)])
I = torch.eye(2, dtype=qkit.get_dtype())

n = 7
parity = 1
proj_rep = z()
uni_rep = x()
data = 64

alt_tensor = (torch.arange(n) + parity) % 2
ops = NKron(*[(uni_rep if val == 0 else I) for val in alt_tensor.tolist()])

I_n_1 = I
for _ in range(n - 2):
    I_n_1 = NKron(I_n_1, I) 

O = NKron(proj_rep, I_n_1) @ ops @ NKron(I_n_1, proj_rep)

h1 = torch.linspace(0, 1.6, data).repeat(data)
h2 = torch.linspace(1.6, -1.6, data).repeat_interleave(data)

h_j = NKron(z(), x(), z())
h_h1 = x()
h_h2 = NKron(x(), x())

H_J,H_H1,H_H2 = generate_hermitian(n,h_j,h_h1,h_h2) 

eig_vec = []
for _ in range(len(h1)):
    eig_vec.append(torch.linalg.eigh(-H_J - h1[_] * H_H1 - h2[_] * H_H2)[1][:, 0].unsqueeze(-1))
eig_vec = torch.stack(eig_vec)

result = (dagger(eig_vec) @ torch.stack([O] * len(h1)) @ eig_vec).real

torch.save({'var_a': result.detach()}, f'./SPT_exp/Fig6_phase_diagrams_n{n}.pt')