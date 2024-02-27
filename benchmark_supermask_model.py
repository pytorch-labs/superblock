import torch
import torchvision.models as models
import torch.utils.benchmark as benchmark
import torch._dynamo
from supermask import apply_supermask, SupermaskLinear
import argparse


def to_bsr(tensor, blocksize):
    if tensor.ndim != 2:
        raise ValueError("to_bsr expects 2D tensor")
    if tensor.size(0) % blocksize or tensor.size(1) % blocksize:
        raise ValueError("Tensor dimensions must be divisible by blocksize")
    return tensor.to_sparse_bsr(blocksize)


def verify_sparsity(model, verbose=False):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_weights = module.weight.numel()
            sparse_weights = (module.weight == 0).sum().item()
            sparsity_percentage = (sparse_weights / total_weights) * 100
            if verbose:
                print(f"Sparsity verified in layer {name}: {sparsity_percentage:.2f}%")


def load_model_with_supermask(model_name, checkpoint_path, bsr_blocksize=None, apply_sparsity=True, device='cuda', torch_compile=False, verbose=False, **supermask_args):
    model = models.__dict__.get(model_name)
    if not model:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    model = model()
    apply_supermask(model, device=device, verbose=verbose, **supermask_args)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    input_tensor = torch.rand(32, 3, 224, 224, device=device)

    if apply_sparsity:
        for module in model.modules():
            if isinstance(module, SupermaskLinear):
                module.compile_weight()
        verify_sparsity(model, verbose)
    elif bsr_blocksize is not None:
        raise ValueError("BSR conversion requires sparsity to be applied. Use --apply-sparsity or remove --bsr argument.")

    output_before_bsr = model(input_tensor)

    if bsr_blocksize:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    module.weight = torch.nn.Parameter(to_bsr(module.weight.data, bsr_blocksize))
                except ValueError as e:
                    if verbose:
                        print(f"Unable to convert weight of {name} to bsr format: {e}")

    if torch_compile:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        model = torch.compile(model)

    if verbose:
        bsr_info = f" and BSR format applied with block size {bsr_blocksize}" if bsr_blocksize else ""
        print(f"Model loaded successfully with supermask applied{bsr_info}.")

    output_after_bsr = model(input_tensor)
    assert torch.allclose(output_before_bsr, output_after_bsr, rtol=1e-3, atol=1e-5), "Output discrepancy after BSR conversion"

    return model


def benchmark_in_us(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt="f(*args, **kwargs)", globals={"f": f, "args": args, "kwargs": kwargs})
    return int(t0.blocked_autorange().mean * 1e6)


def benchmark_model_with_supermask(model_name, checkpoint_path, batch_size=256, bsr_blocksize=None, apply_sparsity=True, device='cuda', verbose=False, **supermask_args):
    model = load_model_with_supermask(model_name, checkpoint_path, bsr_blocksize, apply_sparsity, verbose=verbose, device=device, **supermask_args)
    input_tensor = torch.rand(batch_size, 3, 224, 224, device=device)
    return benchmark_in_us(model, input_tensor)


def main():
    parser = argparse.ArgumentParser(description='Load a model with optional BSR conversion')
    parser.add_argument('model', type=str, help='Name of the model to load')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size of input')
    parser.add_argument('--bsr', type=int, nargs='?', const=4, default=None, help='Optional BSR block size (default 4)')
    parser.add_argument('--apply-sparsity', dest='apply_sparsity', action='store_true', help='Apply sparsity to the model')
    parser.add_argument('--no-sparsity', dest='apply_sparsity', action='store_false', help='Do not apply sparsity to the model')
    parser.set_defaults(apply_sparsity=True)
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model and inputs to')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose log')
    parser.add_argument('--sparsity-linear', type=float, default=0.0, help='Sparsity for linear layers')
    parser.add_argument('--sp-linear-tile-size', type=int, default=1, help='Tile size for linear layer sparsity')
    parser.add_argument('--conv1x1-sparsity', type=float, default=0.0, help='Sparsity for 1x1 convolution layers')
    parser.add_argument('--sp-conv1x1-tile-size', type=int, default=1, help='Tile size for 1x1 convolution layer sparsity')
    parser.add_argument('--conv-sparsity', type=float, default=0.0, help='Sparsity for convolution layers')
    parser.add_argument('--sp-conv-tile-size', type=int, default=1, help='Tile size for convolution layer sparsity')
    
    args = parser.parse_args()

    time_us = benchmark_model_with_supermask(args.model, args.checkpoint_path, args.batch_size, args.bsr, args.apply_sparsity, args.device, args.verbose,
                                            linear_sparsity=args.sparsity_linear, linear_sp_tilesize=args.sp_linear_tile_size,
                                            conv1x1_sparsity=args.conv1x1_sparsity, conv1x1_sp_tilesize=args.sp_conv1x1_tile_size,
                                            conv_sparsity=args.conv_sparsity, conv_sp_tilesize=args.sp_conv_tile_size)

    print(f"Model: {args.model}, Batch Size: {args.batch_size}, Sparsity Applied: {args.apply_sparsity}, "
      f"BSR Block Size: {'None' if args.bsr is None else args.bsr}, "
      f"Execution Time: {time_us} microseconds")



if __name__ == "__main__":
    main()
