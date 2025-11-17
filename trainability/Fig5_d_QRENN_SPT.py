import torch
import gc
import quairkit as qkit

from quairkit.circuit import Circuit
from quairkit.core.state.backend import Hamiltonian
from quairkit.qinfo import trace, NKron, is_hermitian
from quairkit.database import zero_state,one_state,random_hermitian,z


qkit.set_dtype('complex128')
qkit.set_device('cuda:0')


def QRENN(hamiltonian:torch.Tensor,
            m:int, 
            n:int, 
            layers:int 
            ) -> Circuit:
    
    cir = Circuit(m + n)

    U_data = torch.linalg.matrix_exp(-1j * hamiltonian)
    
    cir.h(qubits_idx=list(range(m, m+ n)))
    
    for _ in range(layers):

        cir.u3(qubits_idx=list(range(m)))
        
        cir.control_oracle(U_data, system_idx = [list(range(m))] + 
                list(range(m, n +m)))
        
    cir.u3(qubits_idx=list(range(m)))

    return cir


def train_model(Hams:torch.Tensor,
                m:int,
                n:int, 
                ITR:int, 
                slot:int,
                z0: Hamiltonian,
                labels
                ):
    
    # initialize the model
    cir = QRENN(Hams, m, n, layers=slot)
    
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
        
        if scheduler.get_last_lr()[0]<2e-8 or _>2e4 :
            break
        _ += 1   
        
    return cir.param.detach()

def main():
    

    m = 1
    n = 8
    
    ITR = 10000
    
    slot = 10
    
    total = 600
    
    z0 = qkit.Hamiltonian([[1.0, ",".join([f"Z{i}" for i in range(m)])]])
    
    list_accuracy = []
    list_result = []
    
        
    for train_data in range(10,110, 10):   
        print("train_data:", train_data, "n:", n, "slot:", slot, '\n' + '-' * 100 + '\n')
        
        sample = total - train_data
        data = torch.load(f'./SPT_exp/Fig5_train_n_{n}_data{train_data}.pt')
        Hams = data["var_a"]
        labels = data["var_b"]
        print(labels)
        del data
        gc.collect()
        torch.cuda.empty_cache()
        
        data = torch.load(f'./SPT_exp/Fig5_test_n_{n}_data{total - train_data}.pt')
        Hams_test = data["var_a"]
        theory_labels_test = data["var_b"]

        del data
        gc.collect()
        torch.cuda.empty_cache()
        
        for k in range(20):
            seed = torch.randint(0, 2**32, (1,)).item()
            qkit.set_seed(seed)
            print('k=',k, 'seed=',qkit.get_seed())
            

            cir_param = train_model(Hams = Hams,
                                m = m,
                                n = n,
                                ITR = ITR,
                                slot = slot,
                                z0 = z0,
                                labels = labels
                                )
            
            test_cir = QRENN(Hams_test, m, n, layers=slot)
            test_cir.update_param(cir_param)
            result = torch.real(test_cir().expec_val(z0)).detach()
            list_result.append(result)
            test_labels = torch.zeros(sample)
            test_labels[result >= 0] = 1
            
            print(1-torch.mean(torch.abs(theory_labels_test-test_labels)))
            list_accuracy.append(1-torch.mean(torch.abs(theory_labels_test-test_labels)))
            
            del test_cir, result, cir_param, test_labels
            gc.collect()
            torch.cuda.empty_cache()
            
        del Hams, labels, Hams_test, theory_labels_test
        gc.collect()
        torch.cuda.empty_cache()
    
    torch.save({'var_a': torch.stack(list_accuracy).detach(),'var_b': list_result},
               f'./SPT_exp/Fig5_d_test_#{slot}slot_m_{m}_n_{n}.pt')
    
if __name__ == '__main__':
    main()