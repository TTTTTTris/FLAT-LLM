import torch
import torch.nn as nn


class UVTemplate(nn.Module):
    def __init__(self):
        super(UVTemplate, self).__init__()
        
    def update_params_dict(self):
        total_count, ratio = self.count_params()
        self.__params_dict={
            'u': self.U,
            'v': self.V,
            'bias': self.bias,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'rank': self.rank,
            'num_params': total_count,
            'params_ratio': ratio
        }

    def count_params(self):
        total_count=self.U.numel()
        total_count+=self.V.numel()
        if self.use_bias:
            total_count+=self.bias.numel()
        base_total=self.in_features*self.out_features
        if self.use_bias:
            base_total+=self.bias.numel()
        ratio=total_count/base_total
        return total_count, ratio

    def get_params_dict(self):
        self.update_params_dict()
        return self.__params_dict


    def extra_repr(self):
        num_params, ratio =self.count_params()
        return f"in_features={self.in_features}, out_features={self.out_features}, num_params={num_params}, ratio={ratio}"

    
    def forward(self, x):
        if self.training:
            W = self.U @ self.V 
            out = x @ W
            if self.use_bias:
                out += self.bias
            return out
        else:
            inter = x @ self.U
            out = inter @ self.V
            if self.use_bias:
                out += self.bias
            return out


    @classmethod
    def from_file(cls, model_params, config):
        params=model_params[config.layer_name]
        bias = params['bias']
        if bias!=None:
            use_bias = True 
        else:
            use_bias = False
        instance=cls(params['in_features'], 
                    params['out_features'], 
                    params['rank'], 
                    use_bias)
        instance.U = nn.Parameter(params['u'].contiguous())
        instance.V = nn.Parameter(params['v'].contiguous())
        if use_bias:
            instance.bias=nn.Parameter(params['bias'].contiguous())
        # instance.update_params_dict()
        return instance
