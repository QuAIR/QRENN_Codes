import torch
import quairkit as qkit
import gc

from quairkit.qinfo import NKron
from quairkit.database import z,x,y


qkit.set_dtype('complex128')
qkit.set_device('cuda:0')
            
def generate_hermitian(n:int, H_j:torch.Tensor, H_h:torch.Tensor) -> torch.Tensor:

    return sum([NKron(torch.eye(2**i), H_j, torch.eye(2 ** (n - i - 3))) for i in range(n - 2)]
            ), sum([NKron(torch.eye(2**i), H_h, torch.eye(2 ** (n - i - 2))) for i in range(n - 1)])
            
            
def main():
    
    n = 8
    
    h_j = NKron(x(), z(), x())
    h_h = NKron(y(), y())

    H_J, H_h = generate_hermitian(n,h_j,h_h)
    H_J += NKron(x(), torch.eye(2**(n-3)), x(), z()) + NKron(z(), x(), torch.eye(2**(n-3)), x()) 
    H_h += NKron(y(), torch.eye(2**(n-2)), y())
    
    total = 600
    lambda_train_without_one = torch.sort(torch.rand(total)*2)[0]
    perm = torch.randperm(total)
    
    train_data = 100
        
    sampled = torch.sort(lambda_train_without_one[perm[:train_data]])[0]

    Hams = []
    for h in sampled:
        Hams.append( - H_J + h * H_h)
        
    del h_j,h_h
    gc.collect()
    torch.cuda.empty_cache()
    
    labels = torch.zeros(train_data)
    labels[sampled >=1 ] = 1 
    
    torch.save({
        'var_a': torch.stack(Hams).detach(),
        'var_b': labels.detach()
        }, f'./SPT_exp/Fig5_train_n_{n}_data{train_data}.pt')
    
    del Hams
    gc.collect()
    torch.cuda.empty_cache()
        
    remaining_lambda = torch.sort(lambda_train_without_one[perm[train_data:]])[0]
        
    test_Hams =[]
    for h in remaining_lambda:
        test_Hams.append( - H_J + h * H_h)
    
    theory_labels = torch.zeros(total - train_data)
    theory_labels[remaining_lambda >= 1] = 1
    
    torch.save({
        'var_a': torch.stack(test_Hams).detach(),
        'var_b': theory_labels.detach(),
        'var_c': remaining_lambda.detach(),
        }, f'./SPT_exp/Fig5_test_n_{n}_data{total - train_data}.pt')
    
if __name__ == '__main__':
    main()