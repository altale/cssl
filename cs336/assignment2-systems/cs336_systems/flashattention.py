import torch
import math
import timeit
from collections.abc import Callable
import numpy as np
import triton
import triton.language as tl

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    d_k = K.size(-1)
    pre_compute_qk = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    if is_causal:
        n_queries, n_keys = Q.shape[-2], K.shape[-2]
        causal_mask = torch.tril(torch.ones(n_queries, n_keys, device=Q.device, dtype=torch.bool))
        pre_compute_qk = pre_compute_qk.masked_fill(~causal_mask, float("-inf"))
    return torch.softmax(pre_compute_qk, dim = -1) @ V

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), value=-float("inf"), dtype=tl.float32)
    if is_causal:
        q_index = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s = tl.dot(q, tl.trans(k)) * scale # Q_TILE_SIZE * K_TILE_SIZE
        if is_causal:
            k_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_index[:, None] >= k_index[None, :]
            s = tl.where(mask, s, s - 1e6)
        m_new = tl.maximum(m, tl.max(s, axis=-1)) # Q_TILE_SIZE
        correction = tl.exp(m - m_new) # Q_TILE_SIZE
        m = m_new
        p = tl.exp(s - m[:, None]) # Q_TILE_SIZE * K_TILE_SIZE
        l = correction * l + tl.sum(p, axis=-1)
        o = correction[:, None] * o + tl.dot(p.to(v.dtype), v)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
    o = o / l[:, None]
    l = m + tl.log(l)
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, d),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, d),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, d),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dK = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)

    if is_causal:
        k_index = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        Delta = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            q_index = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = q_index[:, None] >= k_index[None, :]
            S = tl.where(mask, S, S - 1e6)
        P = tl.exp(S - L[:, None])
        dV += tl.dot(tl.trans(P).to(dO.dtype), dO)
        dP = tl.dot(dO, tl.trans(V))
        dS = P * (dP - Delta[:, None])
        dK += tl.dot(tl.trans(dS).to(Q.dtype), Q) * scale

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr, dO_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, d),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, d),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    Delta = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    dQ = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)

    if is_causal:
        q_index = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            k_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_index[:, None] >= k_index[None, :]
            S = tl.where(mask, S, S - 1e6)
        P = tl.exp(S - L[:, None])
        dP = tl.dot(dO, tl.trans(V))
        dS = P * (dP - Delta[:, None])
        dQ += tl.dot(dS.to(K.dtype), K) * scale

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))



_Q_TILE_SIZE = 16
_K_TILE_SIZE = 16
    
    
class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        D = K.shape[-1]
        Q_TILE_SIZE = _Q_TILE_SIZE
        K_TILE_SIZE = _K_TILE_SIZE
        scale = 1 / math.sqrt(D)
        
        k_tiles = torch.split(K, K_TILE_SIZE, dim=-2)
        q_tiles = torch.split(Q, Q_TILE_SIZE, dim=-2)
        v_tiles = torch.split(V, K_TILE_SIZE, dim=-2)
        # k_tiles = torch.nn.utils.rnn.pad_sequence(k_tiles, batch_first=True, padding_value=0)
        # q_tiles = torch.nn.utils.rnn.pad_sequence(q_tiles, batch_first=True, padding_value=0)
        kv_tiles = tuple(zip(k_tiles, v_tiles))
        o_tiles = []
        l_tiles = []
        
        for q in q_tiles:
            m = torch.full(q.shape[:-1], -torch.inf, device=Q.device, dtype=Q.dtype)
            l = torch.full(q.shape[:-1], 0, device=Q.device, dtype=torch.float32)
            o = torch.zeros(q.shape, device=Q.device, dtype=Q.dtype)
            for k, v in kv_tiles:
                s = q @ k.transpose(-1, -2) * scale
                m_new = torch.max(s, dim=-1).values
                correction = torch.exp(m - m_new)
                m = m_new
                p = torch.exp(s - m[..., None])
                l = correction * l + torch.sum(p, dim=-1)
                o = correction[..., None] * o + p @ v
            o = o / l[..., None]
            l = m + torch.log(l)
            o_tiles.append(o)
            l_tiles.append(l)
        
        O = torch.cat(o_tiles, dim = -2)
        L = torch.cat(l_tiles, dim = -1)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return O
                 
    
    @staticmethod
    @torch.compile
    def backward(ctx: torch.autograd.function.FunctionCtx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        scale = ctx.scale
        D = torch.sum(O * dO, dim = -1)
        S = Q @ K.transpose(-1, -2) * scale
        P = torch.exp(S - L[..., None])
        dV = P.transpose(-1, -2) @ dO
        dP = dO @ V.transpose(-1, -2)
        dS = P * (dP - D[..., None])
        dQ = dS @ K * scale
        dK = dS.transpose(-1, -2) @ Q * scale
        return dQ, dK, dV, None

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        batch_size = K.shape[0]
        N_KEYS = K.shape[-2]
        D = K.shape[-1]
        N_QUERIES = Q.shape[-2]
        Q_TILE_SIZE = _Q_TILE_SIZE
        K_TILE_SIZE = _K_TILE_SIZE
        O = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
        L = torch.empty(Q.shape[:-1], device=Q.device, dtype=torch.float32)
        flash_fwd_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            1 / math.sqrt(D),
            D,
            Q_TILE_SIZE, K_TILE_SIZE,
            is_causal
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch_size = Q.shape[0]
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        d = Q.shape[-1]
        scale = 1 / math.sqrt(d)
        Q_TILE_SIZE = _Q_TILE_SIZE
        K_TILE_SIZE = _K_TILE_SIZE

        Delta = torch.sum(O * dO, dim=-1)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        flash_bwd_dkdv_kernel[(triton.cdiv(N_KEYS, K_TILE_SIZE), batch_size)](
            Q, K, V,
            L, Delta, dO,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            Delta.stride(0), Delta.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            is_causal,
        )

        flash_bwd_dq_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)](
            Q, K, V,
            L, Delta, dO,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            Delta.stride(0), Delta.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            d,
            Q_TILE_SIZE, K_TILE_SIZE,
            is_causal,
        )

        return dQ, dK, dV, None

# batch_size = 8
# d_models = {16, 32, 64, 128}
# seq_lens = {256, 1024, 4096, 8192, 16384}

# def benchmark(d_model: int, seq_len: int, attn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
#     Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
#     K = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
#     V = torch.randn(batch_size, seq_len, d_model, requires_grad=True, device="cuda")
#     for _ in range(5):
#         O = attn(Q, K, V)
#         O.sum().backward()
#         torch.cuda.synchronize()
#     forward_times = []
#     backward_times = []
#     memory_before_backward = []
#     torch.cuda.reset_peak_memory_stats()
#     for _ in range(100):
#         start = timeit.default_timer()
#         O = attn(Q, K, V)
#         torch.cuda.synchronize()
#         forward_times.append(timeit.default_timer() - start)
#         memory_before_backward.append(torch.cuda.memory_allocated())
#         start = timeit.default_timer()
#         O.sum().backward()
#         torch.cuda.synchronize()
#         backward_times.append(timeit.default_timer() - start)
#     print(f'Forward time: {np.mean(forward_times)}')
#     print(f'Backward time: {np.mean(backward_times)}')
#     print(f'Memory before backward: {np.mean(memory_before_backward)}')

# if __name__ == '__main__':
#     for d_model in d_models:
#         for seq_len in seq_lens:
#             print(f'Benchmarking for d_model: {d_model}, seq_len: {seq_len}')
#             try:
#                 benchmark(d_model, seq_len, scaled_dot_product_attention)
#             except Exception as e:
#                 torch.cuda.empty_cache()
#                 print(f'OOM!: {e.__class__}')
#                 continue
