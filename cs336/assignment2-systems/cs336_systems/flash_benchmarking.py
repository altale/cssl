import argparse
import gc
import itertools

import pandas as pd
import torch
import triton

import cs336_systems.flashattention as fa
from cs336_systems.flashattention import FlashAttentionTriton, scaled_dot_product_attention

SEQ_LENS = [2 ** i for i in range(7, 17)]
D_MODELS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]


def pick_tile_size(seq_len: int, d_model: int) -> int:
    tile = min(128, seq_len)
    while tile * d_model > 128 * 16 and tile > 16:
        tile //= 2
    return tile


def make_inputs(seq_len: int, d_model: int, dtype: torch.dtype, device: str = "cuda"):
    shape = (1, seq_len, d_model)
    Q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V


def clear_grads(*tensors):
    for t in tensors:
        t.grad = None


def bench_forward(fn, Q, K, V, is_causal):
    return triton.testing.do_bench(lambda: fn(Q, K, V, is_causal))


def bench_backward(fn, Q, K, V, is_causal):
    O = fn(Q, K, V, is_causal)
    dO = torch.randn_like(O)

    def run():
        clear_grads(Q, K, V)
        out = fn(Q, K, V, is_causal)
        out.backward(dO)

    return triton.testing.do_bench(run)


def bench_end_to_end(fn, Q, K, V, is_causal):
    def run():
        clear_grads(Q, K, V)
        out = fn(Q, K, V, is_causal)
        out.sum().backward()

    return triton.testing.do_bench(run)


def benchmark_impl(name, fn, seq_len, d_model, dtype):
    try:
        Q, K, V = make_inputs(seq_len, d_model, dtype)
        fwd_ms = bench_forward(fn, Q, K, V, True)
        bwd_ms = bench_backward(fn, Q, K, V, True)
        e2e_ms = bench_end_to_end(fn, Q, K, V, True)
        return fwd_ms, bwd_ms, e2e_ms
    except (torch.cuda.OutOfMemoryError, triton.runtime.errors.OutOfResources) as e:
        print(f"[SKIP] {name}: seq_len={seq_len}, d_model={d_model}, dtype={dtype}: {e}")
        return None, None, None
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def run_benchmark():
    rows = []
    for seq_len, d_model, dtype in itertools.product(SEQ_LENS, D_MODELS, DTYPES):
        tile = pick_tile_size(seq_len, d_model)
        fa._Q_TILE_SIZE = tile
        fa._K_TILE_SIZE = tile

        row = {"seq_len": seq_len, "d_model": d_model, "dtype": str(dtype), "tile_size": tile}

        triton_fwd, triton_bwd, triton_e2e = benchmark_impl(
            "triton", FlashAttentionTriton.apply, seq_len, d_model, dtype
        )
        row["triton_fwd_ms"] = triton_fwd
        row["triton_bwd_ms"] = triton_bwd
        row["triton_e2e_ms"] = triton_e2e

        pytorch_fwd, pytorch_bwd, pytorch_e2e = benchmark_impl(
            "pytorch", scaled_dot_product_attention, seq_len, d_model, dtype
        )
        row["pytorch_fwd_ms"] = pytorch_fwd
        row["pytorch_bwd_ms"] = pytorch_bwd
        row["pytorch_e2e_ms"] = pytorch_e2e

        print(row)
        rows.append(row)

    return pd.DataFrame(rows)


def get_args():
    parser = argparse.ArgumentParser(description="FlashAttention-2 vs PyTorch benchmarking")
    parser.add_argument("--output", type=str, default="flash_benchmark_results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    assert torch.cuda.is_available()

    df = run_benchmark()
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    print(df.to_markdown(index=False))
