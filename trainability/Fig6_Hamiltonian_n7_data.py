import torch
import quairkit as qkit
import gc

from quairkit.qinfo import NKron,dagger
from quairkit.database import z,x,y


qkit.set_dtype('complex128')
qkit.set_device('cuda:0')

    
def generate_hermitian(n:int, h_j:torch.Tensor, h_h1:torch.Tensor, h_h2:torch.Tensor,) -> torch.Tensor:
    
    return sum([NKron(torch.eye(2**i), h_j, torch.eye(2 ** (n - i - 3))) for i in range(n - 2)]
            ), sum([NKron(torch.eye(2**i), h_h1, torch.eye(2 ** (n - i - 1))) for i in range(n)]
                    ), sum([NKron(torch.eye(2**i), h_h2, torch.eye(2 ** (n - i - 2))) for i in range(n - 1)])
            
            
def main():
    
    n = 7
    parity = 1
    train_data = 100
    test_data = 200
    
    proj_rep = z()
    uni_rep = x()


    I = torch.eye(2, dtype= qkit.get_dtype())
    alt_tensor = (torch.arange(n) + parity) % 2
    ops = NKron(*[(uni_rep if val == 0 else I) for val in alt_tensor.tolist()])
    
    I_n_1 = I
    for _ in range(n - 2):
        I_n_1 = NKron(I_n_1, I) 

    O = NKron(proj_rep, I_n_1) @ ops @ NKron(I_n_1, proj_rep)
    

    h_j = NKron(z(), x(), z())
    h_h1 = x()
    h_h2 = NKron(x(), x())

    H_J,H_H1,H_H2 = generate_hermitian(n,h_j,h_h1,h_h2) 
    
    
    for h1 in [0.8,1.2]:
        h2 = torch.linspace(1.6, -1.6, train_data).view(train_data, 1, 1)
        
        H_J_batch  = -H_J.unsqueeze(0).expand(train_data, 2**n, 2**n)         # (train_data, train_data, train_data)
        H_H1_batch = -H_H1.unsqueeze(0).expand(train_data, 2**n, 2**n)        # (train_data, train_data, train_data)
        H_H2_batch = -H_H2.unsqueeze(0).expand(train_data, 2**n, 2**n)        # (train_data, train_data, train_data)
        
        Hams = H_J_batch + h1 * H_H1_batch + h2 * H_H2_batch
        eig_vec = torch.linalg.eigh(Hams)[1][:, :, 0].unsqueeze(-1)

        result = (dagger(eig_vec) @ O.expand(train_data, 2**n, 2**n) @ eig_vec).real.squeeze()

        labels = torch.zeros(len(result))
        labels[result > 0.3] = 1
        
        torch.save({
        'var_a': Hams.detach(),
        'var_b': labels.detach(),
        'var_c': result.detach()
        }, f'./SPT_exp/Fig6_train_n{n}_data{train_data}_h1{h1}.pt')

        del Hams, labels,eig_vec
        gc.collect()
        torch.cuda.empty_cache()
        
        h2_ = torch.linspace(1.6, -1.6, test_data)
        h2 = h2_.view(test_data, 1, 1)
        H_J_batch  = -H_J.unsqueeze(0).expand(test_data, 2**n, 2**n)         # (test_data, train_data, train_data)
        H_H1_batch = -H_H1.unsqueeze(0).expand(test_data, 2**n, 2**n)        # (test_data, train_data, train_data)
        H_H2_batch = -H_H2.unsqueeze(0).expand(test_data, 2**n, 2**n)        # (test_data, train_data, train_data)
        
        test_Hams = H_J_batch + h1 * H_H1_batch + h2 * H_H2_batch
        eig_vec = torch.linalg.eigh(test_Hams)[1][:, :, 0].unsqueeze(-1)

        result = (dagger(eig_vec) @ O.expand(test_data, 2**n, 2**n) @ eig_vec).real.squeeze()
        
        labels = torch.zeros(len(result))
        labels[result > 0.3] = 1
        
        torch.save({
        'var_a': test_Hams.detach(),
        'var_b': labels.detach(),
        'var_c': result.detach()
        }, f'./SPT_exp/Fig6_test_n{n}_data{test_data}_h1{h1}.pt')
    
    
if __name__ == '__main__':
    main()
