import torch.nn as nn
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.linalg import hadamard
import numpy as np

def set_parameter(model, name, param):
    if '.' in name:
        names = name.split('.')
        set_parameter(getattr(model, names[0]), '.'.join(names[1:]), param)
    else:
        setattr(model, name, param)


def apply_bsr_ts(model, blocksize):
    for name, param in model.named_parameters():
        if isinstance(param, SupermaskTensor):
            try:
                set_parameter(model, name, torch.nn.Parameter(param.data.to_sparse_bsr(blocksize)))
                print(f"Converted SupermaskTensor {name} to bsr format.")
            except RuntimeError:
                # Fall back to  strided
                set_parameter(model, name, torch.nn.Parameter(param.data.to_strided()))
                print(f"Converted SupermaskTensor {name} to strided format.")

def verify_sparsity_ts(model):
    for name, param in model.named_parameters():
        if isinstance(param, SupermaskTensor):
            total_weights = param.to_strided().numel()
            sparse_weights = (param.to_strided() == 0).sum().item()
            sparsity_percentage = (sparse_weights / total_weights) * 100
            print(f"Sparsity verified in layer {name}: {sparsity_percentage:.2f}%")

def verify_sparsity_ts_bsr(model):
    for name, param in model.named_parameters():
        if param.layout == torch.sparse_bsr:
            print(f"ratio: {param.values().numel() / param.numel()}")

def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model

def swap_conv2d_1x1_to_linear(model, filter_fn=None):
    """
    Changes all conv2d 1x1 modules to equivalent linear modules so that they can then be quantized.
    """

    class PermuteSandwich(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *args):
            return self.mod(args[0].permute(0, 2, 3, 1)).permute(-0, 3, 1, 2)

    def replace_conv2d_1x1(conv):
        assert conv.kernel_size == (1, 1)
        lin = torch.nn.Linear(
            conv.in_channels, conv.out_channels, bias=(conv.bias is None)
        )
        lin.weight = torch.nn.Parameter(conv.weight.squeeze(-1, -2))
        lin.bias = conv.bias
        return PermuteSandwich(lin)

    if filter_fn is None:
        filter_fn = lambda mod, *args: isinstance(
            mod, torch.nn.Conv2d
        ) and mod.kernel_size == (1, 1)

    _replace_with_custom_fn_if_matches_filter(
        model, replace_conv2d_1x1, filter_fn=filter_fn
    )


# original supermask
scores_min=None
scores_max=9e9
uniform_init_01 = False

# adjusted supermask, initialize scores with uniform distribution in [0,1], clamp scores in each step in [0,1]
# scores_min=0.
# scores_max=1.
# uniform_init_01 = True

def percentile(t, q):
    """Return the value that is larger than q% of t"""
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


def to_bsr(tensor, blocksize=256):
    if tensor.ndim != 2:
        print("Tensor is not 2D, skipping BSR conversion.")
        return tensor  
    
    if tensor.size(0) % blocksize or tensor.size(1) % blocksize:
        print("Tensor dimensions are not divisible by blocksize, skipping BSR conversion.")
        return tensor  
    
    try:
        converted_tensor = tensor.to_sparse_bsr(blocksize=blocksize)
        print(f"Converted tensor to BSR format with blocksize: {blocksize}")
        return converted_tensor 
    except ValueError as e:
        print(f"Unable to convert tensor to BSR format: {e}")
        return tensor 


class GetSubnet(torch.autograd.Function):
    """Supermask STE function"""
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        scores.clamp_(min=scores_min,max=scores_max)
        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

from typing import Dict, Tuple, Any
SUPERMASK_OPS_TABLE: Dict[Any, Any] = {}

def implements(aten_ops):
    """Use this decorator to implement a function for an aten op in __torch_dispatch__"""

    def decorator(func):
        for op in aten_ops:
            SUPERMASK_OPS_TABLE[op] = func
        return func

    return decorator

@implements([torch.ops.aten.detach.default, torch.ops.aten.detach])
def noop_detach(func, *args, **kwargs):
    return args[0][0]


# weight, scores, shift, scale should be parameters
# that can be trained
class SupermaskTensor(torch.Tensor):

    def __new__(
        cls,
        weight: torch.Tensor,
        scores: torch.Tensor,
        sparsity: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
        tile_size: int):
        supermask_tensor = torch.Tensor._make_wrapper_subclass(
                cls,
                weight.shape,
                weight.stride(),
                weight.storage_offset(),
                dtype=weight.dtype,
                device=weight.device,
                requires_grad=weight.requires_grad,
                )
        return supermask_tensor

    def __init__(
        self,
        weight: torch.Tensor,
        scores: torch.Tensor,
        sparsity: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
        tile_size: int):
        self.weight = weight
        self.scores = scores
        self.sparsity = sparsity
        self.scale = scale
        self.shift = shift
        self.tile_size = tile_size

    def get_mask(self):
        subnet = GetSubnet.apply(self.scores,
                                 torch.zeros_like(self.scores),
                                 torch.ones_like(self.scores),
                                 self.sparsity)

        if self.tile_size != 1:
            for i, k in enumerate(self.weight.shape):
                subnet = subnet.repeat_interleave(self.tile_size, dim=i)
                subnet = torch.narrow(subnet, i, 0, k)

        return subnet

    def to_strided(self):
        subnet = self.get_mask()
        return (self.weight*self.scale+self.shift) * subnet

    def to_sparse_bsr(self, blocksize):
        return self.to_strided().to_sparse_bsr(blocksize)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func in SUPERMASK_OPS_TABLE:
            return SUPERMASK_OPS_TABLE[func](func, args, kwargs)
        print("func: ", func)
        return NotImplemented

def to_supermask_tensor(weight, sparsity, fixed_mask, fixed_weight, bitwidth, transform, fixed_transform, tile_size):
    # initialize the scores
    max_sparsity = 1 - (1 / math.prod([math.ceil(k / tile_size) for k in weight.size()]))
    if sparsity > max_sparsity:
        print(
            f"reducing sparsity from {sparsity} to {max_sparsity}",
            f"(maximum sparsity for layer with shape {weight.size()} and tile size {tile_size})"
        )
        sparsity = max_sparsity
    scores = torch.empty([max(1, int(math.ceil(wn / tile_size))) for wn in weight.size()], device=weight.device, dtype=weight.dtype)
    nn.init.uniform_(scores) if uniform_init_01 else nn.init.kaiming_uniform_(scores, a=math.sqrt(5))

    # the shift and the scale are transformation parameters 
    # the actually used weights = self.weight*self.scale+self.shift
    # the transformation is activated only for quantized weights
    shift = torch.tensor([0.], requires_grad=False, device=weight.device, dtype=weight.dtype)
    scale = torch.tensor([1.], requires_grad=False, device=weight.device, dtype=weight.dtype)

    assert bitwidth is None

    # self.weight.requires_grad = not fixed_weight

    return SupermaskTensor(weight,
            scores,
            sparsity,
            scale,
            shift,
            tile_size)

def apply_supermask_ts(
    model,
    linear_sparsity=0.0,
    linear_sp_tilesize=1,
    skip_last_layer_sparsity=False,
    skip_first_transformer_sparsity=False,
    verbose=False,
):
    swap_conv2d_1x1_to_linear(model)
    for n, m in model.named_modules():
        # check conditions for skipping sparsity
        if skip_last_layer_sparsity and n == "heads.head":
            continue
        if skip_first_transformer_sparsity and "encoder.layers.encoder_layer_0" in n:
            continue
        if linear_sparsity != 0.0 and isinstance(m, torch.nn.Linear):
            m.weight = torch.nn.Parameter(to_supermask_tensor(m.weight,
                linear_sparsity,
                False,
                False,
                None,
                None,
                None,
                linear_sp_tilesize))
        if linear_sparsity != 0.0 and isinstance(m, torch.nn.MultiheadAttention):
            assert m._qkv_same_embed_dim
            m.in_proj_weight = torch.nn.Parameter(to_supermask_tensor(m.in_proj_weight,
                linear_sparsity,
                False,
                False,
                None,
                None,
                None,
                linear_sp_tilesize))
