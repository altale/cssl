import torch
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        _W = torch.empty(out_features, in_features, dtype=dtype, device=device)
        _std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(_W, mean=0, std=_std, a = -3 * _std, b = 3 * _std)
        self.W = torch.nn.Parameter(_W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.W.t())
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        _E = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        torch.nn.init.trunc_normal_(_E, mean=0, std=1, a = -3, b = 3)
        self.E = torch.nn.Parameter(_E)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.E[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        _g = torch.empty(d_model, dtype=dtype, device=device)
        torch.nn.init.ones_(_g)
        self.g = torch.nn.Parameter(_g)
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch_size * sequence_length * d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # rms: batch_size * sequence_length
        rms = x.square().mean(-1).add(self.eps).rsqrt()
        result = rms[:,:,None] * self.g * x
        return result.to(in_dtype)