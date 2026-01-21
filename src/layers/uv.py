import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
from src.layers.templates import UVTemplate
from src.layers import utils
import time
class SVDW(UVTemplate):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(SVDW, self).__init__()
        
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize U and V matrices
        self.U = nn.Parameter(torch.randn(in_features, rank))
        self.V = nn.Parameter(torch.randn(rank, out_features))

        # Initialize bias if needed
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        

    @classmethod
    def from_teacher(cls, teacher_layer, config):
        if config.mode!='teacher':
            raise ValueError('config mode must be teacher')
        if not isinstance(teacher_layer, nn.Linear):
            raise ValueError('The teacher_layer must be a Linear layer')

        device = teacher_layer.weight.device

        weight=teacher_layer.weight.detach()#.to(device)
        if teacher_layer.bias!=None:
            bias=teacher_layer.bias.detach()#.to(device)
        else:
            bias=None
        weight=weight.t()
        if bias==None:
            use_bias=False
        else:
            use_bias=True

        # start = time.time()
        U_new, V_new = utils.svd(weight, config.rank)
        # end = time.time()
        

        # print(f"Elapsed time: {end - start:.4f} seconds")
        instance = cls(teacher_layer.in_features, teacher_layer.out_features, config.rank, bias=use_bias)

        instance.U = nn.Parameter(U_new.to(device))
        instance.V = nn.Parameter(V_new.to(device))

        if use_bias:
            instance.bias = nn.Parameter(bias.to(device))
        return instance



class SVD_LLM(UVTemplate):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(SVD_LLM, self).__init__()
        
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize U and V matrices
        self.U = nn.Parameter(torch.randn(in_features, rank))
        self.V = nn.Parameter(torch.randn(rank, out_features))

        # Initialize bias if needed
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    

    @classmethod
    def from_teacher(cls, teacher_layer, config):
        if config.mode!='teacher':
            raise ValueError('config mode must be teacher')
        if not isinstance(teacher_layer, nn.Linear):
            raise ValueError('The teacher_layer must be a Linear layer')

        device = teacher_layer.weight.device

        weight=teacher_layer.weight.detach()#.to(device)
        if teacher_layer.bias!=None:
            bias=teacher_layer.bias.detach()#.to(device)
        else:
            bias=None

        if bias==None:
            use_bias=False
        else:
            use_bias=True

        emp_cov_mat = config.covariance_matrix

        delta = utils.minimal_shift_to_pd(emp_cov_mat)
        adj_cov_mat = emp_cov_mat + 1.1 * delta * torch.eye(emp_cov_mat.shape[0], device=device, dtype=emp_cov_mat.dtype)    

        try:
            S = torch.linalg.cholesky(adj_cov_mat)
        except RuntimeError as e:
            raise RuntimeError(f"Cholesky decomposition failed for layer {config.layer_name}: {e}")
        try:
            S_inv = torch.linalg.inv(S)
        except RuntimeError as e:
            raise RuntimeError(f"S inverse failed for layer {config.layer_name}: {e}")
        
        W_transformed = weight @ S
        U_new, V_new = utils.svd(W_transformed, config.rank)
        V_new = V_new @ S_inv
        instance = cls(teacher_layer.in_features, teacher_layer.out_features, config.rank, bias=use_bias)

        instance.U = nn.Parameter(V_new.t().to(device))
        instance.V = nn.Parameter(U_new.t().to(device))

        if use_bias:
            instance.bias = nn.Parameter(bias.to(device))

        return instance

