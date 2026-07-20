import argparse
import torch
import timeit
import numpy as np
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

def get_args():
    parser = argparse.ArgumentParser(description="Transformer benchmarking script")

    parser.add_argument("--model-size", type=str, default="small",
                         choices=["small", "medium", "large"])
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10000)

    parser.add_argument("--mode", type=str, default="forward_backward",
                         choices=["forward", "forward_backward", "full_step"])

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)

    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args


def run_step(model: BasicsTransformerLM,
             data: torch.Tensor, 
             target: torch.Tensor, 
             optimizer: torch.optim.Optimizer, 
             do_backward: bool,
             do_optimizer_step: bool):
    if do_optimizer_step:
        optimizer.zero_grad(set_to_none=True)

    logits = model(data)

    if do_backward:
        loss = cross_entropy(logits, target)
        loss.backward()

    if do_optimizer_step:
        optimizer.step()

if __name__ == "__main__":
    args = get_args()

    MODEL_PRESETS = {
        "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
        "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
        "large": dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    }

    if args.model_size in MODEL_PRESETS:
        preset = MODEL_PRESETS[args.model_size]
        args.d_model = args.d_model or preset["d_model"]
        args.d_ff = args.d_ff or preset["d_ff"]
        args.num_layers = args.num_layers or preset["num_layers"]
        args.num_heads = args.num_heads or preset["num_heads"]

    do_backward = args.mode in ("forward_backward", "full_step")
    do_optimizer_step = args.mode == "full_step"
    
    print(args)
    
    torch.manual_seed(args.seed)
    model = BasicsTransformerLM(vocab_size=args.vocab_size, 
                                context_length=args.context_length,
                                d_model=args.d_model,
                                num_layers=args.num_layers,
                                num_heads=args.num_heads,
                                d_ff=args.d_ff
                            ).to(args.device)
    data = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    
    if do_optimizer_step:
        optimizer = AdamW(model.parameters())
    else:
        optimizer = None
        
    if do_backward:
        target = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    else:
        target = None
    
    # warm-up
    for _ in range(args.warmup_steps):
        run_step(model, data, target, optimizer, do_backward, do_optimizer_step)
        torch.cuda.synchronize()
        
    # ---- measure ----
    timings = []
    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        run_step(model, data, target, optimizer, do_backward, do_optimizer_step)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        timings.append(end - start)
    
    print(np.mean(timings), np.std(timings))