import torch


def minimal_shift_to_pd(A):
    A_sym = 0.5 * (A + A.T)
    eigvals = torch.linalg.eigvalsh(A_sym)  # symmetric eigenvalues
    lambda_min = torch.min(eigvals)
    delta = torch.clamp(-lambda_min, min=0.0)
    return delta.item()


def svd(weight, rank):

    weight_shape = weight.size()
    if weight_shape[0]>weight_shape[1]:
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        U_r = U[:, :rank]  
        S_r = S[:rank]  
        Vt_r = Vt[:rank, :]  
        U_final = U_r * torch.sqrt(S_r).unsqueeze(0)  # Equivalent to U * sqrt(Sigma)
        V_final = torch.sqrt(S_r).unsqueeze(1) * Vt_r 
    else:
        U, S, Vt = torch.linalg.svd(weight.t(), full_matrices=False)
        U_r = U[:, :rank]  
        S_r = S[:rank]  
        Vt_r = Vt[:rank, :]  
        U_trans = U_r * torch.sqrt(S_r).unsqueeze(0)  # Equivalent to U * sqrt(Sigma)
        V_trans = torch.sqrt(S_r).unsqueeze(1) * Vt_r

        U_final = V_trans.t()
        V_final = U_trans.t()

    return U_final, V_final
