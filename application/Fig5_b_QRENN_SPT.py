import torch
import gc
import quairkit as qkit

from quairkit.circuit import Circuit
from quairkit.core.state.backend import Hamiltonian
from quairkit.database import *
from quairkit.qinfo import trace, NKron
from quairkit.database import z


qkit.set_dtype('complex128')
qkit.set_device('cuda:0')


def QRENN(hamiltonian:torch.Tensor,
            trainable_qubits:int, 
            V_number_qubits:int, 
            layers:int 
            ) -> Circuit:
    
    cir = Circuit(trainable_qubits + V_number_qubits)
    
    U_data = torch.linalg.matrix_exp(-1j*hamiltonian)
    
    for _ in range(layers):

        cir.u3(qubits_idx=list(range(trainable_qubits)))
        
        cir.control_oracle(U_data, system_idx = [list(range(trainable_qubits))] + 
                list(range(trainable_qubits, V_number_qubits +trainable_qubits)))

    cir.u3(qubits_idx=list(range(trainable_qubits)))

    return cir


def train_model(Hams:torch.Tensor,
                trainable_qubits:int,
                V_number_qubits:int, 
                ITR:int, 
                slot:int,
                labels
                ):
    
    # initialize the model
    cir = QRENN(Hams, trainable_qubits, V_number_qubits, layers=slot)
    
    # cir is a Circuit type
    opt = torch.optim.Adam(lr=0.1, params=cir.parameters())

    # activate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5)
    
    M = torch.stack([ (torch.eye(2**trainable_qubits) + NKron(*[z() for _ in range(trainable_qubits)]))/2 if b == 1 
                     else  (torch.eye(2**trainable_qubits) - NKron(*[z() for _ in range(trainable_qubits)]))/2 for b in labels])
    M = NKron(M, torch.eye(2**V_number_qubits))  
    
    print("Training:")
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
        
        if scheduler.get_last_lr()[0]<2e-8 or _>3e4:
            break
        
        if _ % 100 == 0 or _ == ITR - 1:
            print(
                f"iter: {_}, loss: {loss:.8f}, lr: {scheduler.get_last_lr()[0]:.2E}"
                )
            
        _ += 1   
        
    return cir.param.detach()


def main():
    

    m = 1
    n = 8
    
    
    slot = 10
    
    ITR = 2000
    
    total = 600
    
    train_data = 40

    z0 = qkit.Hamiltonian([[1.0, ",".join([f"Z{i}" for i in range(m)])]])
    
    
    data = torch.load(f'./SPT_exp/train_n_{n}_data{train_data}.pt')
    Hams = data["var_a"]
    labels = data["var_b"]

    del data
    gc.collect()
    torch.cuda.empty_cache()
    
    data = torch.load(f'./SPT_exp/test_n_{n}_data{total - train_data}.pt')
    Hams_test = data["var_a"]
    theory_labels_test = data["var_b"]
    lambda_test = data["var_c"]
    
    del data
    gc.collect()
    torch.cuda.empty_cache()    


    seed = torch.randint(0, 2**32, (1,)).item()
    qkit.set_seed(seed)
    
    

    cir_param = train_model(Hams = Hams,
                        trainable_qubits = m,
                        V_number_qubits = n,
                        ITR = ITR,
                        slot = slot,
                        labels = labels
                        )

    test_cir = QRENN(Hams_test, m, n, layers=slot)
    test_cir.update_param(cir_param)
    result = torch.real(test_cir().expec_val(z0)).detach()
    test_labels = torch.zeros(total - train_data)
    test_labels[result >= 0] = 1

    print(1-torch.mean(torch.abs(theory_labels_test-test_labels)))
    
    torch.save({
                'var_a': result,
                'var_b': lambda_test,
                'var_c': cir_param,
                'var_d': seed,
                }, f'./SPT_exp/Fig5_b_test_#{slot}slot_m_{m}_n_{n}_seed{seed}.pt')
        
        
if __name__ == '__main__':
    main()