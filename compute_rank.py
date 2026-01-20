#%%
# llama-2-7b
import matplotlib.pyplot as plt
import numpy as np
import torch

def llama_2_7b():
    bi_score_path = 'ranks/wikitext2/llama-2-7b/bi_score.pt'
    bi_score_angular = torch.load(bi_score_path)
    dq = 4096 * 4096
    do = 4096 * 4096
    dk = 4096 * 4096
    dv = 4096 * 4096
    dmlp = 4096 * 11008
    bi_score_angular = torch.tensor(bi_score_angular) / 4096 / 128
    return bi_score_angular, 32, np.array([dq, dk, dv, do, dmlp])

def llama_2_13b():
    bi_score_path = 'ranks/wikitext2/llama-2-13b/bi_score.pt'
    bi_score_angular = torch.load(bi_score_path)
    dq = 5120 * 5120
    do = 5120 * 5120
    dk = 5120 * 5120
    dv = 5120 * 5120
    dmlp = 5120 * 13824
    bi_score_angular = torch.tensor(bi_score_angular) / 5120 / 128
    return bi_score_angular, 40, np.array([dq, dk, dv, do, dmlp])

def mistral_7b():
    bi_score_path = 'ranks/wikitext2/mistral-7b/bi_score.pt'
    bi_score_angular = torch.load(bi_score_path)
    dq = 4096 * 4096
    do = 4096 * 4096
    dk = 4096 * 1024
    dv = 4096 * 1024
    dmlp = 4096 * 14336
    bi_score_angular = torch.tensor(bi_score_angular) / 4096 / 128
    return bi_score_angular, 32, np.array([dq, dk, dv, do, dmlp])

def llama_2_70b():
    bi_score_path = 'ranks/wikitext2/llama-2-70b/bi_score.pt'
    bi_score_angular = torch.load(bi_score_path)
    dq = 8192 * 8192
    do = 8192 * 8192
    dk = 8192 * 1024
    dv = 8192 * 1024
    dmlp = 8192 * 28672
    bi_score_angular = torch.tensor(bi_score_angular) / 4096 / 128
    return bi_score_angular, 80, np.array([dq, dk, dv, do, dmlp])

def llama_3_8b():
    bi_score_path = 'ranks/wikitext2/llama-3-8b/bi_score.pt'
    bi_score_angular = torch.load(bi_score_path)
    dq = 4096 * 4096
    do = 4096 * 4096
    dk = 4096 * 1024
    dv = 4096 * 1024
    dmlp = 4096 * 14336
    bi_score_angular = torch.tensor(bi_score_angular) / 4096 / 128
    return bi_score_angular, 32, np.array([dq, dk, dv, do, dmlp])

bi_score_angular, N, sizes = llama_2_7b()
print(bi_score_angular.sum(), min(bi_score_angular), max(bi_score_angular))
bi_score_angular, N, sizes = llama_2_13b()
print(bi_score_angular.sum(), min(bi_score_angular), max(bi_score_angular))
bi_score_angular, N, sizes = mistral_7b()
print(bi_score_angular.sum(), min(bi_score_angular), max(bi_score_angular))
bi_score_angular, N, sizes = llama_3_8b()
print(bi_score_angular.sum(), min(bi_score_angular), max(bi_score_angular))

# bi_score_angular, N, sizes = llama_2_70b()
# print(bi_score_angular.sum())

#%%
import numpy as np

def proportional_allocation_with_cap(t, C):
    N = len(t)
    w = np.zeros_like(t)
    remaining = np.arange(N)
    while True:
        t_sub = t[remaining]
        C_sub = C
        w_sub = t_sub / t_sub.sum() * C_sub
        clipped = w_sub > 1
        # print(clipped)
        if not torch.any(clipped):
            w[remaining] = w_sub
            break
        # fix the clipped values at 1
        fixed = remaining[clipped]
        w[fixed] = 1.0
        C -= len(fixed)
        remaining = remaining[~clipped]
        if len(remaining) == 0:
            break
    return w

def compute_optimal_sparsity_torch(s, epsilon=0.1, target_avg_sparsity=0.5): # modegpt
    """
    Compute optimal sparsity using the softmax-based closed-form solution (in PyTorch).

    Args:
        s: 1D torch.Tensor of importance scores (length L)
        epsilon: entropy smoothing parameter
        target_avg_sparsity: desired global sparsity (e.g., 0.5 for 50%)

    Returns:
        1D torch.Tensor of per-layer sparsity scores (phi)
    """
    s = s.to(torch.float32)
    L = s.shape[0]
    
    softmax_weights = torch.softmax(-s / epsilon, dim=0)
    phi = L * target_avg_sparsity * softmax_weights
    
    return 1-phi

import matplotlib.colors as mcolors

### choose the model you want to analyze
dataset = 'wikitext2'
model = "llama-2-7b"  # Change this to the model you are analyzing
bi_score_angular, N, sizes = llama_2_7b()
# bi_score_angular, N, sizes = llama_2_13b()
# bi_score_angular, N, sizes = mistral_7b()
# bi_score_angular, N, sizes = llama_2_70b()
# bi_score_angular, N, sizes = llama_3_8b()

import os
os.makedirs(f"ranks/{dataset}/{model}", exist_ok=True)
colormap = plt.get_cmap('rainbow')
fig, ax = plt.subplots()
plt.plot(range(len(bi_score_angular)), bi_score_angular, label='importance', color='gray', linestyle='--')
for target in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    r = 1 - target
    dq, dk, dv, do, dmlp = sizes
    # r = (r * (dmlp * 3 + (dq + dk + dv + do)) - (dq + dk)) / (dmlp * 3 + (dv + do)) # uncomment it if do not compress QK
    print(f'total remained ratio: {(1-target)*100:.2f} %, remained rank ratio (V,O,MLP):{r*100:.2f} %')
    # C = (1 - target) * N
    C = r * N
    phi = proportional_allocation_with_cap(bi_score_angular, C)
    print(f'max remained ratio {phi.max()*100:.2f} %, min remained ratio {phi.min() * 100:.2f} %')
    # torch.save(phi, f"ranks/{dataset}/{model}/sparsity_score_{int(r*100)}%.pt")
    plt.plot(range(N), phi, label=f'reamined_ratio={target * 100}%', color=colormap(1 - target))
sm = plt.cm.ScalarMappable(cmap=colormap, norm=mcolors.Normalize(vmin=0, vmax=100))
cbar = plt.colorbar(sm, ax=ax, label='Remained Ratio (%)')
cbar.ax.tick_params(labelsize=10)

plt.grid()
plt.xlabel('Layer index')
plt.ylabel('Remained parameter ratio')
plt.show()