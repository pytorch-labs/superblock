import os
import time
import sys
import warnings
import hashlib
import torchvision

import presets
import torch
import torch.utils.data
import utils
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from supermask import apply_supermask, SupermaskLinear
from supermask_ts import apply_supermask_ts, SupermaskTensor, apply_bsr_ts, verify_sparsity_ts, verify_sparsity_ts_bsr


def apply_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, SupermaskLinear) and "mlp" in name:
            module.sparsify_offline()


def apply_bsr(model):
    for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "mlp" in name:
                try:
                    module.weight = torch.nn.Parameter(to_bsr(module.weight.data, args.bsr))
                    print(f"Converted {name} to bsr format.")
                except ValueError as e:
                    print(f"Unable to convert weight of {name} to bsr format: {e}")


def to_bsr(tensor, blocksize):
    if tensor.ndim != 2:
        raise ValueError("to_bsr expects 2D tensor")
    if tensor.size(0) % blocksize or tensor.size(1) % blocksize:
        raise ValueError("Tensor dimensions must be divisible by blocksize")
    return tensor.to_sparse_bsr(blocksize)


def verify_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_weights = module.weight.numel()
            sparse_weights = (module.weight == 0).sum().item()
            sparsity_percentage = (sparse_weights / total_weights) * 100
            print(f"Sparsity verified in layer {name}: {sparsity_percentage:.2f}%")


def benchmark_in_ms(warmup, iters, f, *args, **kwargs):
    for _ in range(warmup):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(iters):
        f(*args, **kwargs)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / float(iters)


def main(args):
    print(args)
    device = torch.device(args.device)

    # We disable the cudnn benchmarking because it can noticeably affect the accuracy
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    num_classes = 1000

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    if not args.use_ts:
        apply_supermask(
            model,
            linear_sparsity=args.sparsity_linear,
            linear_sp_tilesize=args.sp_linear_tile_size,
            conv1x1_sparsity=args.sparsity_conv1x1,
            conv1x1_sp_tilesize=args.sp_conv1x1_tile_size,
            conv_sparsity=args.sparsity_conv,
            conv_sp_tilesize=args.sp_conv_tile_size,
            skip_last_layer_sparsity=args.skip_last_layer_sparsity,
            skip_first_transformer_sparsity=args.skip_first_transformer_sparsity,
            device=device,
            verbose=True,
        )
        model.to(device)
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        model_without_ddp = model
        if args.bfloat16:
            print("Using bfloat16")
            model = model.to(torch.bfloat16)
    else:
        model.to(device)
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
        model_without_ddp = model
        if args.bfloat16:
            print("Using bfloat16")
            model = model.to(torch.bfloat16)
        assert args.sparsity_conv1x1 == 0
        assert args.sparsity_conv == 0
        apply_supermask_ts(
            model,
            linear_sparsity=args.sparsity_linear,
            linear_sp_tilesize=args.sp_linear_tile_size,
            skip_last_layer_sparsity=args.skip_last_layer_sparsity,
            skip_first_transformer_sparsity=args.skip_first_transformer_sparsity,
            verbose=True,
        )

    if args.bsr and not args.sparsify_weights:
        raise ValueError("--bsr can only be used when --sparsify_weights is also specified.")
    if not args.use_ts:
        if args.sparsify_weights:
            apply_sparsity(model)
            verify_sparsity(model)
            if args.bsr:
                apply_bsr(model)
    else:
        if args.sparsify_weights:
            verify_sparsity_ts(model)
            if args.bsr:
                print("0 ---")
                apply_bsr_ts(model, args.bsr)
                print("1 ---")
                apply_bsr_ts(model, args.bsr)
                print("2 ---")
                verify_sparsity_ts_bsr(model)
                print("3 ---")
    image = torch.empty(args.batch_size, 3, args.val_crop_size, args.val_crop_size, dtype=torch.bfloat16 if args.bfloat16 else None, device=device)
    with torch.no_grad():
        # model = torch.compile(model, mode='max-autotune')
        ms = benchmark_in_ms(10, 100, model, image)
        max_memory_allocated_bytes = torch.cuda.max_memory_allocated()
        _, total_memory = torch.cuda.mem_get_info()
        max_memory_allocated_percentage = int(100 * (max_memory_allocated_bytes / total_memory))
        max_memory_allocated_bytes = max_memory_allocated_bytes >> 20
        print(f"{ms}ms {max_memory_allocated_bytes}MB of RAM, {max_memory_allocated_percentage}% of RAM", file=sys.stderr)
        return


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    # NOTE: sparsity args
    parser.add_argument("--sparsity-linear", type=float, default=0.0)
    parser.add_argument("--sp-linear-tile-size", type=int, default=1)
    parser.add_argument("--sparsity-conv1x1", type=float, default=0.0)
    parser.add_argument("--sp-conv1x1-tile-size", type=int, default=1)
    parser.add_argument("--sparsity-conv", type=float, default=0.0)
    parser.add_argument("--sp-conv-tile-size", type=int, default=1)
    parser.add_argument("--skip-last-layer-sparsity", action="store_true", help="Skip applying sparsity to the last linear layer (for vit only)")
    parser.add_argument("--skip-first-transformer-sparsity", action="store_true", help="Skip applying sparsity to the first transformer layer (for vit only)")
    parser.add_argument('--sparsify-weights', action='store_true', help='Apply weight sparsification in evaluation mode')
    parser.add_argument('--bsr', type=int, nargs='?', const=256, default=None, help='Convert sparsified weights to BSR format with optional block size (default: 256)')
    parser.add_argument("--bfloat16", action="store_true", help="Use bfloat16")
    parser.add_argument("--use-ts", action="store_true", help="Use Tensor subclass")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
