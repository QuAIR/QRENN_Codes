import torch
import gc
import quairkit as qkit

from quairkit.circuit import Circuit
from quairkit.database import *
from quairkit.qinfo import trace, NKron
from quairkit.database import z


qkit.set_dtype('complex128')
qkit.set_device('cuda:0')


def QRENN(hamiltonian:torch.Tensor,
            m:int, 
            n:int, 
            layers:int 
            ) -> Circuit:
    
    cir = Circuit(m + n)
    
    U_data = torch.linalg.matrix_exp(-1j*hamiltonian)
    
    cir.ry(qubits_idx =list(range(m, m+ n)), param = torch.tensor([3/2, 3/2, -1/2,3/2,3/2,3/2,1/2,3/2,1/2])*torch.pi)
    
    for _ in range(layers):
        cir.universal_qudits(qubits_idx=list(range(m)))
        
        cir.control_oracle(U_data, system_idx = [list(range(m))] + 
                list(range(m, n +m)))

    cir.universal_qudits(qubits_idx=list(range(m)))

    return cir


def train_model(Hams:torch.Tensor,
                m:int,
                n:int, 
                ITR:int, 
                slot:int,
                labels
                ):
    
    # initialize the model
    cir = QRENN(Hams, m, n, layers=slot)
    
    # cir is a Circuit type
    opt = torch.optim.Adam(lr=0.1, params=cir.parameters())

    # activate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5)
    
    M = torch.stack([ (torch.eye(2**m) + NKron(*[z() for _ in range(m)]))/2 if b == 1 
                     else  (torch.eye(2**m) - NKron(*[z() for _ in range(m)]))/2 for b in labels])
    M = NKron(M, torch.eye(2**n))  
    
    _ = 0
    while _ < ITR:
        
        opt.zero_grad()

        loss = 1-torch.mean(trace(cir().density_matrix @ M).real)
        
        loss.backward()  # compute gradients
        opt.step()  # update parameters
        scheduler.step(loss)  # activate scheduler

        loss = loss.item()
        
        if _ == ITR - 100 and scheduler.get_last_lr()[0]>=2e-8:
            ITR += 1000
        

        if scheduler.get_last_lr()[0]<2e-8 or _>1e3:
            break

            
        _ += 1

    return cir.param.detach()


def main():
    

    m = 1
    n = 9
    
    
    slot = 60
    
    ITR = 2000
    
    train_data = 100
    test_data = 200
    
    
    z0 = qkit.Hamiltonian([[1.0, ",".join([f"Z{i}" for i in range(m)])]])
    
    h1 = 0.4

    data = torch.load(f'./SPT_exp/Fig6_train_n{n}_data{train_data}_h1{h1}.pt')
    Hams = data["var_a"]
    labels = data["var_b"]

    del data
    gc.collect()
    torch.cuda.empty_cache()
    
    data = torch.load(f'./SPT_exp/Fig6_test_n{n}_data{test_data}_h1{h1}.pt')
    Hams_test = data["var_a"]
    theory_labels_test = data["var_b"]

    del data
    gc.collect()
    torch.cuda.empty_cache()    
    

    seed = 2023395290
    qkit.set_seed(seed)
    
    cir_param = train_model(Hams = Hams,
                        m = m,
                        n = n,
                        ITR = ITR,
                        slot = slot,
                        labels = labels
                        )

    test_cir = QRENN(Hams_test, m, n, layers=slot)
    test_cir.update_param(cir_param)
    
    result = torch.real(test_cir().expec_val(z0)).detach()
    test_labels = torch.zeros(len(result))
    test_labels[result >= 0] = 1

    acc = 1-torch.mean(torch.abs(theory_labels_test-test_labels))
    print("test accuracy:", acc)
        
    torch.save({
    'var_a': result,
    'var_b': cir_param,
    }, f'./SPT_exp/Fig6_m{m}_n{n}_h1_{h1}_slot{slot}_seed{seed}_acc{acc}%.pt')

        
    del cir_param
    gc.collect()
    torch.cuda.empty_cache()

        
if __name__ == '__main__':
    main()