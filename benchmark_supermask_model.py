import torch
import torchvision.models as models
import torch.utils.benchmark as benchmark
import torch._dynamo
from supermask import apply_supermask, SupermaskLinear
import argparse

def to_bsr(tensor, blocksize):
    # Convert dense tensor to BSR format
    if len(tensor.shape) != 2:
        raise ValueError("to_bsr expects 2D tensor")
    if tensor.shape[0] % blocksize != 0 or tensor.shape[1] % blocksize != 0:
        raise ValueError("Tensor dimensions must be divisible by blocksize")
    return tensor.to_sparse_bsr(blocksize)

def load_model_with_supermask(model_name, checkpoint_path, bsr_blocksize=None, device='cuda', torch_compile=False, verbose=False, **supermask_args):
    # Check if model is available in torchvision
    if model_name not in models.__dict__:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    # Load base model
    model = models.__dict__[model_name]()

    # Apply supermask
    apply_supermask(model, device=device, verbose=verbose, **supermask_args)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.to(device)

    # get output
    input = torch.rand(32, 3, 224, 224).to(device)
    output0 = model(input)

    # Convert weights to BSR format if bsr_blocksize is specified
    if bsr_blocksize is not None:
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, SupermaskLinear):
                count+=1
                module.compile_weight()
        
        output1 = model(input)

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    module.weight = torch.nn.Parameter(to_bsr(module.weight.data, bsr_blocksize))
                except:
                    if verbose:
                        print(f"Unable to convert weight of {name} with shape {module.weight.shape} to bsr format")

    if torch_compile:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        model = torch.compile(model)

    if verbose:
        print("Model loaded successfully with supermask applied", end='')
        if bsr_blocksize is not None:
            print(f" and BSR format applied with block size {bsr_blocksize}.")
        else:
            print(".")

    # verify correctness
    output2 = model(input)
    print(output0)
    print(output1)
    print(output2)
    assert(torch.allclose(output0, output1, rtol=1e-3, atol=1e-5))

    return model


def benchmark_in_us(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return int(t0.blocked_autorange().mean * 1e6)

def benchmark_model_with_supermask(model_name, checkpoint_path, batch_size=256, bsr_blocksize=None, device='cuda', verbose=False, **supermask_args):
    # Load the model with supermask
    model = load_model_with_supermask(
        model_name,
        checkpoint_path,
        bsr_blocksize=bsr_blocksize,
        verbose=verbose,
        **supermask_args
    )

    if model:
        input = torch.rand(batch_size, 3, 224, 224).to(device)
        time = benchmark_in_us(model, input)
        return(time)
    else:
        raise RuntimeError("Failed to create model")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load a model with optional BSR conversion')
    parser.add_argument('model', type=str, help='Name of the model to load')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size of input')
    parser.add_argument('--bsr', type=int, nargs='?', const=4, default=None, help='Optional BSR block size (default 4)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model and inputs to')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose log')
    parser.add_argument('--sparsity-linear', type=float, default=0.0, help='Sparsity for linear layers')
    parser.add_argument('--sp-linear-tile-size', type=int, default=1, help='Tile size for linear layer sparsity')
    parser.add_argument('--conv1x1-sparsity', type=float, default=0.0, help='Sparsity for 1x1 convolution layers')
    parser.add_argument('--sp-conv1x1-tile-size', type=int, default=1, help='Tile size for 1x1 convolution layer sparsity')
    parser.add_argument('--conv-sparsity', type=float, default=0.0, help='Sparsity for convolution layers')
    parser.add_argument('--sp-conv-tile-size', type=int, default=1, help='Tile size for convolution layer sparsity')
    args = parser.parse_args()

    # Load the model with supermask
    time = benchmark_model_with_supermask(
        args.model,
        args.checkpoint_path,
        batch_size=args.batch_size,
        bsr_blocksize=args.bsr,
        verbose=args.verbose,
        device=args.device,
        linear_sparsity=args.sparsity_linear,
        linear_sp_tilesize=args.sp_linear_tile_size,
        conv1x1_sparsity=args.conv1x1_sparsity,
        conv1x1_sp_tilesize=args.sp_conv1x1_tile_size,
        conv_sparsity=args.conv_sparsity,
        conv_sp_tilesize=args.sp_conv_tile_size
    )

    print(f"{time} us")

if __name__ == "__main__":
    main()
