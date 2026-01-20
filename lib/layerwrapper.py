import torch
import torch.nn as nn
import math
import numpy as np
from typing import List, Optional

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, args, config, layer_id=0, layer_name="none", device='cuda'):
        self.layer = layer
        self.dev = device
        self.args = args
        self.config = config
        self.d_head_kv = self.config.num_key_value_heads
        self.d_head = self.config.num_attention_heads

        if type(layer) == nn.Linear:
            self.rows = layer.weight.data.shape[0]
            self.columns = layer.weight.data.shape[1]
            self.cov = torch.zeros((self.columns, self.columns), device=self.dev)
            self.cov_v = torch.zeros((self.d_head_kv, self.rows // self.d_head_kv, self.rows // self.d_head_kv), device=self.dev)

        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def compute_eig(self, cov, eig_vec_attr, eig_num_attr, eig_val_attr, score=None):
        # check data
        if torch.isnan(cov).any() or torch.isinf(cov).any():
            print("X contains NaN or Inf values!")
            cov = torch.nan_to_num(cov)  # Replace NaNs with 0
            print(cov)

        # damp = 0.01 * torch.mean(torch.diag(cov))
        # diag = torch.arange(cov.shape[-1]).to(device=self.dev)
        # cov[diag, diag] += damp

        eig_val, eig_vec = torch.linalg.eigh(cov)
        index = torch.argsort(eig_val, descending=True)
        eig_val = eig_val[index]
        max_eig_val = eig_val[0].clone()

        setattr(self, eig_val_attr, eig_val)

        eig_val /= max_eig_val
        eig_vec = eig_vec[:, index]
        setattr(self, eig_vec_attr, eig_vec)
        
        # if score is not None:
        #     eig_val *= score
        #     eig_num = torch.sum(eig_val > tol).item()
        # else:
        #     eig_num = torch.sum(eig_val > tol).item()
        total = np.sum(eig_val.cpu().numpy())
        cumulative = np.cumsum(eig_val.cpu().numpy()) / total
        print(cumulative)
        eig_num = np.searchsorted(cumulative, self.args.tol) + 1
        eig_num = torch.tensor(eig_num, device=self.dev).item()

        setattr(self, eig_num_attr, eig_num)

        # Save eig_num to a text file in append mode
        if(eig_num_attr == 'eig_num_r'):
            print('eig_num_r: ', eig_num)
            with open("eig_values/eig_values_r_" + str(self.args.n_tensorized_modes) + '_' + str(self.args.tol) + '_' + str(self.args.sparsity_ratio) + ".txt", "a") as f:
                f.write(f"{eig_num}\n")
        elif(eig_num_attr == 'eig_num_l'):
            print('eig_num_l: ', eig_num)
            with open("eig_values/eig_values_l_" + str(self.args.n_tensorized_modes) + '_' + str(self.args.tol) + '_' + str(self.args.sparsity_ratio) + ".txt", "a") as f:
                f.write(f"{eig_num}\n")
        else:
            print('eig_num: ', eig_num)
            with open("eig_values/eig_values_" + str(self.args.n_tensorized_modes) + '_' + str(self.args.tol) + '_' + str(self.args.sparsity_ratio) + ".txt", "a") as f:
                f.write(f"{eig_num}\n")

        del eig_vec, eig_val, index

    def compute_eig_3d(self, cov, eig_vec_attr, eig_num_attr, eig_val_attr, score=None):
        # check data
        if torch.isnan(cov).any() or torch.isinf(cov).any():
            print("X contains NaN or Inf values!")
            cov = torch.nan_to_num(cov)  # Replace NaNs with 0
            print(cov)

        # damp = 0.01 * torch.mean(torch.diagonal(cov, dim1=-2, dim2=-1), dim=-1)
        # diag = torch.arange(cov.shape[-1]).to(device=self.dev)
        # cov[:, diag, diag] += damp.unsqueeze(-1)

        eig_val, eig_vec = torch.linalg.eigh(cov)
        index = torch.argsort(eig_val, descending=True, dim=-1)
        # eig_val = torch.gather(eig_val, -1, index)
        # max_eig_val = eig_val[:, 0].clone()

        setattr(self, eig_val_attr, eig_val)

        # eig_val = eig_val / max_eig_val.unsqueeze(-1)

        eig_vec = torch.stack([
            eig_vec[i, :, index[i]] for i in range(eig_vec.shape[0])
        ], dim=0) 

        setattr(self, eig_vec_attr, eig_vec)

        # cumulative = torch.cumsum(eig_val, dim=-1) / torch.sum(eig_val, dim=-1, keepdim=True)
        # print(cumulative)

        # eig_num = torch.sum(cumulative < self.args.tol, dim=-1) + 1
        
        # setattr(self, eig_num_attr, eig_num)

        # print('eig_num: ', eig_num)
        del eig_val, eig_vec, index

    def compute_eig_norm(self, cov, cov_attr, score=None):
        # check data
        if torch.isnan(cov).any() or torch.isinf(cov).any():
            print("X contains NaN or Inf values!")
            cov = torch.nan_to_num(cov)  # Replace NaNs with 0
            print(cov)

        eig_val, eig_vec = torch.linalg.eigh(cov)

        eig_val = torch.sqrt(eig_val)
        cov_sqrt = eig_vec @ torch.diag_embed(eig_val) @ eig_vec.transpose(-2, -1)

        setattr(self, cov_attr, cov_sqrt)

        del eig_val, eig_vec, cov_sqrt

    def add_batch(self, inp, out, flag=0):
        if type(out) == torch.Tensor:
            out = out.clone().to(self.dev)
            if len(out.shape) == 2:
                out = out.unsqueeze(0)
            tmp = out.shape[0]
            if len(out.shape) == 3:
                out = out.reshape(-1, out.shape[-1])

        if type(inp) == torch.Tensor:
            inp = inp.clone().to(self.dev)
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if len(inp.shape) == 3:
                inp = inp.reshape(-1, inp.shape[-1])

        if not flag:
            x_tensor = inp # .reshape(-1, self.columns)
            x_tensor = x_tensor.to(torch.double)
            self.cov += x_tensor.T @ x_tensor # / (self.rank[self.d//2 + 1] - 1)
            del x_tensor
        else:
            if len(out.shape) == 4: # [B, Nh, S, dh]
                out = out.transpose(1, 2) # [B, S, Nh, dh]
            x_tensor = out.reshape(-1, self.d_head_kv, self.rows // self.d_head_kv).transpose(0, 1) # [n_head, seqlen, dh]
            x_tensor = x_tensor.to(torch.double)
            self.cov_v += torch.bmm(x_tensor.transpose(1,2), x_tensor)
            if (self.nsamples == self.args.nsamples - 1):
                self.compute_eig_3d(self.cov_v, 'eig_vec', 'eig_num', 'eig_val', None)
            del x_tensor
        # inp = inp.t()

        # self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        # inp = inp.type(torch.float32)
        # self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

    def add_batch_from_saved(self, saved_inputs, saved_outputs, flag=0):
        """
        Add batch using pre-saved activations instead of collecting during forward pass.
        saved_inputs: tensor of shape [nsamples, seq_len, hidden_dim] OR path to .pt file
        saved_outputs: tensor of shape [nsamples, seq_len, hidden_dim] OR path to .pt file
        """
        # Load from disk if paths are provided
        if isinstance(saved_inputs, str):
            saved_inputs = torch.load(saved_inputs, weights_only=True)
        if isinstance(saved_outputs, str):
            saved_outputs = torch.load(saved_outputs, weights_only=True)
            
        if not flag:
            # Process inputs for covariance matrix (for down_proj)
            x_tensor = saved_inputs.reshape(-1, saved_inputs.shape[-1]).to(self.dev)
            x_tensor = x_tensor.to(torch.double)
            self.cov += x_tensor.T @ x_tensor
            del x_tensor
        else:
            # Process outputs for attention (for v_proj)  
            out = saved_outputs.to(self.dev)
            if len(out.shape) == 4: # [B, Nh, S, dh]
                out = out.transpose(1, 2) # [B, S, Nh, dh]
            x_tensor = out.reshape(-1, self.d_head_kv, self.rows // self.d_head_kv).transpose(0, 1) # [n_head, seqlen, dh]
            x_tensor = x_tensor.to(torch.double)
            self.cov_v += torch.bmm(x_tensor.transpose(1,2), x_tensor)
            self.compute_eig_3d(self.cov_v, 'eig_vec', 'eig_num', 'eig_val', None)
            del x_tensor
            
        self.nsamples += saved_inputs.shape[0]

    def load_precomputed_statistics(self, saved_cov=None, saved_eig_vec=None):
        """
        Load pre-computed statistics instead of computing them from activations.
        saved_cov: covariance matrix (tensor or path to .pt file)
        saved_eig_vec: eigenvectors (tensor or path to .pt file)
        """
        if saved_cov is not None:
            if isinstance(saved_cov, str):
                self.cov = torch.load(saved_cov, weights_only=True).to(self.dev)
            else:
                self.cov = saved_cov.to(self.dev)
                
        if saved_eig_vec is not None:
            if isinstance(saved_eig_vec, str):
                self.eig_vec = torch.load(saved_eig_vec, weights_only=True).to(self.dev)
            else:
                self.eig_vec = saved_eig_vec.to(self.dev)

    def add_batch_qk(self, q, k): # [bsz, n_head, seqlen, dh]
        tmp = q.shape[0]
        dim = q.shape[-1]
        if not hasattr(self, 'cov_q'):
            self.cov_q = torch.zeros((self.d_head, dim, dim), device=self.dev)
        if not hasattr(self, 'cov_k'):
            self.cov_k = torch.zeros((self.d_head_kv, dim, dim), device=self.dev)
        q_tensor = q.squeeze(0).to(torch.double) # [n_head, seqlen, dh]
        k_tensor = k.squeeze(0).to(torch.double) # [n_head, seqlen, dh]
        self.cov_q += torch.bmm(q_tensor.transpose(1,2), q_tensor)
        self.cov_k += torch.bmm(k_tensor.transpose(1,2), k_tensor)
        if (self.nsamples == self.args.nsamples - 1):
            self.compute_eig_norm(self.cov_q, 'cov_sqrt_q', None)
            self.compute_eig_norm(self.cov_k, 'cov_sqrt_k', None)

        del q_tensor, k_tensor            

        self.nsamples += tmp


class WrappedShortGPT:
    def __init__(self, layer, args, config, device='cuda'):
        self.layer = layer
        self.dev = device
        self.args = args
        self.config = config
        self.importances = [0 for _ in range(config.num_hidden_layers)]

    def block_influence(
        self,
        input_hidden_state: torch.Tensor,
        output_hidden_state: torch.Tensor,
        angular: bool = False,
    ):
        """
        input_hidden_state: B, S, D
        output_hidden_state: B, S, D
        """
        _, _, d = input_hidden_state.shape
        input_hidden_state = input_hidden_state.reshape(-1, d)
        output_hidden_state = output_hidden_state.reshape(-1, d)

        norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
        norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

        sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
        sim = sim.diagonal().nan_to_num(nan=0.5)
        sim = torch.clamp(sim, 0, 1)

        if angular:
            print("Your are using angular distance")
            return (torch.arccos(sim) / torch.pi)

        return 1 - sim

    def add_batch(self, inp, out, layer_id, angular=False):
        self.importances[layer_id] += self.block_influence(
            inp,
            out,
            angular=angular
        ).sum().cpu().item()

class ShortLlamaHF:
    def __init__(self, model, args, config, importances, n_prune_layers: Optional[int] = None, device='cuda'):
        self.model = model
        self.dev = device
        self.args = args
        self.config = config
        self.importances = importances
        self.n_prune_layers = int(n_prune_layers)

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = [],
        angular: Optional[bool] = False
    ):
        print(self.n_prune_layers)
        if angular:
            assert self.importances, "Need to compute importances with eval_importance()"
            assert self.n_prune_layers, "Need number of layers to prune, set `n_prune_layers`"
            start_layer = np.argsort(np.array(self.importances[:-self.n_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not layers_to_remove and self.n_prune_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.model.model.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove