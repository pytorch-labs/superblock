#  Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torchvision.models as models
import torch.utils.benchmark as benchmark
import torch._dynamo
from supermask import SupermaskLinear, apply_supermask
import argparse

def annotate_layers_with_names(model):
    for name, module in model.named_modules():
        if isinstance(module, SupermaskLinear):
            module.layer_name = name

def load_model_with_supermask(model_name, checkpoint_path, device='cuda', verbose=False, **supermask_args):
    model = models.__dict__.get(model_name)()
    if not model:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    apply_supermask(model, device=device, verbose=verbose, **supermask_args)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    annotate_layers_with_names(model)
    return model

def benchmark_in_us(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"f": f, "args": args, "kwargs": kwargs})
    return int(t0.blocked_autorange().mean * 1e6)

def benchmark_model_with_supermask(model_name, checkpoint_path, batch_size=256, device='cuda', verbose=False, **supermask_args):
    model = load_model_with_supermask(model_name, checkpoint_path, device=device, verbose=verbose, **supermask_args)
    input_tensor = torch.rand(batch_size, 3, 224, 224, device=device)
    return benchmark_in_us(model, input_tensor)

def main():
    parser = argparse.ArgumentParser(description='Load and benchmark a model with supermask applied')
    parser.add_argument('model', type=str, help='Name of the model to load')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for benchmarking')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model onto')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    # Supermask related arguments
    parser.add_argument('--sparsity-linear', type=float, default=0.0, help='Sparsity for linear layers')
    parser.add_argument('--sp-linear-tile-size', type=int, default=1, help='Tile size for linear layer sparsity')
    parser.add_argument('--conv1x1-sparsity', type=float, default=0.0, help='Sparsity for 1x1 convolution layers')
    parser.add_argument('--sp-conv1x1-tile-size', type=int, default=1, help='Tile size for 1x1 convolution layer sparsity')
    parser.add_argument('--conv-sparsity', type=float, default=0.0, help='Sparsity for convolution layers')
    parser.add_argument('--sp-conv-tile-size', type=int, default=1, help='Tile size for convolution layer sparsity')

    args = parser.parse_args()

    time_us = benchmark_model_with_supermask(args.model, args.checkpoint_path, args.batch_size, args.device, args.verbose,
                                             linear_sparsity=args.sparsity_linear, linear_sp_tilesize=args.sp_linear_tile_size,
                                             conv1x1_sparsity=args.conv1x1_sparsity, conv1x1_sp_tilesize=args.sp_conv1x1_tile_size,
                                             conv_sparsity=args.conv_sparsity, conv_sp_tilesize=args.sp_conv_tile_size)

    print(f"Model: {args.model}, Batch Size: {args.batch_size}, "
          f"Execution Time: {time_us} microseconds")

if __name__ == "__main__":
    main()
