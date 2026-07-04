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
    
def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x.sigmoid() * x
    
class SwiGLU_FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_1 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.W_2 = Linear(in_features=self.d_ff, out_features=self.d_model, device=device, dtype=dtype)
        self.W_3 = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(SiLU(self.W_1(x)) * self.W_3(x))
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        numerator = torch.arange(end = max_seq_len, device=device, dtype=dtype)
        denominator = torch.pow(theta, torch.arange(end=d_k, step=2, device=device, dtype=dtype).repeat_interleave(2) / d_k)
        angles = numerator[:,None] / denominator    # (max_seq_len, d_k)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x:  (..., seq_len, d_k), token_positions: (..., seq_len)
        x2 = torch.stack((x[..., 1::2], -x[..., 0::2]), dim=-1).flatten(-2)
        # [q2, -q1, q4, -q3]
        # q1 = q1cos + q2sin
        # q2 = q2cos - q1sin
        # Based on pytorch's indexing mechanism, cos[token_position] is (..., seq_len, d_k)
        return x * self.cos[token_positions] + x2 * self.sin[token_positions]
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    expx = torch.exp(x - max_vals)
    return expx / torch.sum(expx, dim=dim, keepdim=True)
    
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    d_k = K.size(-1)
    pre_compute_qk = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        pre_compute_qk.masked_fill(mask.logical_not(), -torch.inf)
    return softmax(pre_compute_qk, dim = -1) @ V

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None, dtype=None):
        # Here we set d_k = d_v = d_model / h, thus QKVO are (d_model, d_model)
        # The following part we assume that d_model % num_heads == 0
        super().__init__()
        self.d_kv = d_model // num_heads
        self.h = num_heads
        self.d_model = d_model
        self.W_K = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_Q = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_V = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.W_O = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.RoPE = RotaryPositionalEmbedding(theta=theta, d_k =self.d_kv ,max_seq_len=max_seq_len, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device).expand(*x.shape[:-1])
        Q = self.W_Q.forward(x).view(*x.shape[:-1], self.h, self.d_kv).transpose(-2, -3)
        #(..., seq_len, h * d_k) -> (..., seq_len, h, d_k) -> (..., h, seq_len, d_k)
        K = self.W_K.forward(x).view(*x.shape[:-1], self.h, self.d_kv).transpose(-2, -3)
        V = self.W_V.forward(x).view(*x.shape[:-1], self.h, self.d_kv).transpose(-2, -3)
        Q = self.RoPE.forward(Q, token_positions)
        K = self.RoPE.forward(K, token_positions)
        mask = torch.ones(seq_len, seq_len, device=x.device).triu(diagonal=1)
        result = scaled_dot_product_attention(Q, K, V, mask).transpose(-2, -3).contiguous().view(*x.shape[:-1], self.d_model)
        # (..., h, seq_len, d_k) -> (..., seq_len, h, d_k) -> (... seq_len * h * d_k) -> (... seq_len, d_model)
        return self.W_O.forward(result)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, device=None, dtype=None):
        self.MHA = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.FFN = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.MHA(RMSNorm(x)) + x
        y = self.FFN(RMSNorm(y)) + y
        return y
    
class Transformer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, device=None, dtype=None):
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = torch.nn.Sequential(
            *(TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, device=device, dtype=dtype) for _ in range(num_layers))
        )
        self.outputEmbedding = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.embedding(x)
        output = self.blocks(output)
        output = RMSNorm(output)
        output = self.outputEmbedding(output)
        return softmax(output, dim=-1)