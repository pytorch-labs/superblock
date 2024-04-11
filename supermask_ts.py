import torch.nn as nn
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.linalg import hadamard
import numpy as np

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



class SupermaskLinear(nn.Linear):
    """Supermask class for Linear layer"""
    def __init__(self, sparsity, fixed_mask, fixed_weight, bitwidth, transform, fixed_transform, *args, **kwargs):
        tile_size = kwargs.pop("tile_size", 1)
        super(SupermaskLinear, self).__init__(*args, **kwargs)
        # initialize the scores
        max_sparsity = 1 - (1 / math.prod([math.ceil(k / tile_size) for k in self.weight.size()]))
        self.sparsity = sparsity
        if self.sparsity > max_sparsity:
            print(
                f"reducing sparsity from {self.sparsity} to {max_sparsity}",
                f"(maximum sparsity for layer with shape {self.weight.size()} and tile size {tile_size})"
            )
            self.sparsity = max_sparsity
        self.tile_size = tile_size
        self.sparsify_weights = False
        self.scores = nn.Parameter(
            torch.empty(
                [max(1, int(math.ceil(wn / tile_size))) for wn in self.weight.size()]
            ),
            requires_grad=not fixed_mask,
        )
        nn.init.uniform_(self.scores) if uniform_init_01 else nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # the shift and the scale are transformation parameters 
        # the actually used weights = self.weight*self.scale+self.shift
        # the transformation is activated only for quantized weights
        self.shift=nn.Parameter(torch.Tensor(1).fill_(0.), requires_grad=False)
        self.scale=nn.Parameter(torch.Tensor(1).fill_(1.), requires_grad=False)
        
        with torch.no_grad():
            # if bitwidth is None, then use floating point values in self.weight
            # if bitwidth is not None, then quantize self.weight into k-bit (k=bitwidth)
            # quantized values are -2^(k-1), -2^(k-1)+1, ..., 0, 1, ..., 2^(k-1)-1 
            # these quantized values are uniformly distributed
            if bitwidth is not None:
                weights_max = torch.max(self.weight).item()
                weights_min = torch.min(self.weight).item()
                least_step = (weights_max-weights_min)/pow(2,bitwidth)
                left_bound = weights_min-1e-6
                right_bound = weights_min+least_step+1e-6
                # self.shift=nn.Parameter(torch.Tensor(1).fill_( (weights_min+(pow(2,bitwidth-1)+0.5)*least_step) if transform[0] is None else transform[0] ), requires_grad=not fixed_transform[0])
                # self.scale=nn.Parameter(torch.Tensor(1).fill_( least_step if transform[1] is None else transform[1] ), requires_grad=not fixed_transform[1])
                # for example, if using binary weights (k=1) with -a, +a, set transform = [a,2a]; if using binary weights (k=1) with a, 0, set transform = [0,-a];
                self.shift=nn.Parameter(torch.Tensor(1).fill_( 0. if transform[0] is None else transform[0] ), requires_grad=not fixed_transform[0])
                self.scale=nn.Parameter(torch.Tensor(1).fill_( 1. if transform[1] is None else transform[1] ), requires_grad=not fixed_transform[1])
                for i in range(-int(pow(2,bitwidth-1)),int(pow(2,bitwidth-1))):
                    self.weight[torch.logical_and(self.weight>left_bound, self.weight<=right_bound)] = i                 
                    left_bound = right_bound
                    right_bound += least_step

        self.weight.requires_grad = not fixed_weight

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
    
    def sparsify_offline(self):
        subnet = self.get_mask()
        self.weight.data = (self.weight*self.scale+self.shift) * subnet
        self.sparsify_weights = True

    def forward(self, x):
        if not self.sparsify_weights:
            subnet = self.get_mask()
            w = (self.weight*self.scale+self.shift) * subnet
        else:
            w = self.weight
        return F.linear(x, w, self.bias)
    

class SupermaskConv2d(nn.Conv2d):
    """Supermask class for Conv2d layer"""
    def __init__(self, sparsity, fixed_mask, fixed_weight, bitwidth, transform, fixed_transform, *args, **kwargs):
        tile_size = kwargs.pop("tile_size", 1)
        super(SupermaskConv2d, self).__init__(*args, **kwargs)
        # initialize the scores
        max_sparsity = 1 - (1 / math.prod([math.ceil(k / tile_size) for k in self.weight.size()]))
        self.sparsity = sparsity
        if self.sparsity > max_sparsity:
            print(
                f"reducing sparsity from {self.sparsity} to {max_sparsity}",
                f"(maximum sparsity for layer with shape {self.weight.size()} and tile size {tile_size})"
            )
            self.sparsity = max_sparsity
        self.tile_size = tile_size
        self.scores = nn.Parameter(
            torch.empty(
                [max(1, int(math.ceil(wn / tile_size))) for wn in self.weight.size()]
            ),
            requires_grad=not fixed_mask,
        )
        nn.init.uniform_(self.scores) if uniform_init_01 else nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # the shift and the scale are transformation parameters 
        # the actually used weights = self.weight*self.scale+self.shift
        # the transformation is activated only for quantized weights
        self.shift=nn.Parameter(torch.Tensor(1).fill_(0.), requires_grad=False)
        self.scale=nn.Parameter(torch.Tensor(1).fill_(1.), requires_grad=False)

        with torch.no_grad():
            # if bitwidth is None, then use floating point values in self.weight
            # if bitwidth is not None, then quantize self.weight into k-bit (k=bitwidth)
            # quantized values are -2^(k-1), -2^(k-1)+1, ..., 0, 1, ..., 2^(k-1)-1 
            # these quantized values are uniformly distributed
            if bitwidth is not None:
                weights_max = torch.max(self.weight).item()
                weights_min = torch.min(self.weight).item()
                least_step = (weights_max-weights_min)/pow(2,bitwidth)
                left_bound = weights_min-1e-6
                right_bound = weights_min+least_step+1e-6
                # self.shift=nn.Parameter(torch.Tensor(1).fill_( (weights_min+(pow(2,bitwidth-1)+0.5)*least_step) if transform[0] is None else transform[0] ), requires_grad=not fixed_transform[0])
                # self.scale=nn.Parameter(torch.Tensor(1).fill_( least_step if transform[1] is None else transform[1]), requires_grad=not fixed_transform[1])
                # for example, if using binary weights (k=1) with -a, +a, set transform = [a,2a]; if using binary weights (k=1) with a, 0, set transform = [0,-a];
                self.shift=nn.Parameter(torch.Tensor(1).fill_( 0. if transform[0] is None else transform[0] ), requires_grad=not fixed_transform[0])
                self.scale=nn.Parameter(torch.Tensor(1).fill_( 1. if transform[1] is None else transform[1] ), requires_grad=not fixed_transform[1])
                for i in range(-int(pow(2,bitwidth-1)),int(pow(2,bitwidth-1))):
                    self.weight[torch.logical_and(self.weight>left_bound, self.weight<=right_bound)] = i                 
                    left_bound = right_bound
                    right_bound += least_step

        self.weight.requires_grad = not fixed_weight

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores,
                                 torch.zeros_like(self.scores),
                                 torch.ones_like(self.scores),
                                 self.sparsity)
    
        if self.tile_size != 1:
            for i, k in enumerate(self.weight.shape):
                # if k == 1: continue
                subnet = subnet.repeat_interleave(self.tile_size, dim=i)
                subnet = torch.narrow(subnet, i, 0, k)

        w = (self.weight*self.scale+self.shift) * subnet
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

@torch.no_grad()
def set_sparsity(modules, sparsity):
    """Set the sparsity for supermask layers"""
    sm_idx = 0
    for mod in modules:
        if isinstance(mod, (SupermaskLinear, SupermaskConv2d)):
            mod.sparsity=sparsity[sm_idx]
            sm_idx += 1
            print(mod)
            print('Sparsity: ', mod.sparsity)


def apply_supermask(
    model,
    linear_sparsity=0.0,
    linear_sp_tilesize=1,
    conv1x1_sparsity=0.0,
    conv1x1_sp_tilesize=1,
    conv_sparsity=0.0,
    conv_sp_tilesize=1,
    skip_last_layer_sparsity=False,
    skip_first_transformer_sparsity=False,
    device="cuda",
    verbose=False,
):
    sparsified_modules = {}

    for n, m in model.named_modules():
        # check conditions for skipping sparsity
        if skip_last_layer_sparsity and n == "heads.head":
            continue
        if skip_first_transformer_sparsity and "encoder.layers.encoder_layer_0" in n:
            continue
        
        # convert 1x1 convolutions
        if conv1x1_sparsity != 0.0 and isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1, 1):
            new_m = SupermaskConv2d(
                conv1x1_sparsity, False, False, None, None, None,
                m.in_channels,
                m.out_channels,
                m.kernel_size, 
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=m.bias is not None,
                padding_mode=m.padding_mode,
                device=device,
                tile_size=conv1x1_sp_tilesize,
            )
            new_m.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                new_m.bias.data.copy_(m.bias.data)
            sparsified_modules[n] = new_m
            continue

        # convert all other convolutions (not tested!)
        if conv_sparsity != 0.0 and isinstance(m, torch.nn.Conv2d):
            new_m = SupermaskConv2d(
                conv_sparsity, False, False, None, None, None,
                m.in_channels,
                m.out_channels,
                m.kernel_size, 
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                groups=m.groups,
                bias=m.bias is not None,
                padding_mode=m.padding_mode,
                device=device,
                tile_size=conv_sp_tilesize,
            )
            new_m.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                new_m.bias.data.copy_(m.bias.data)
            sparsified_modules[n] = new_m
            continue

        if linear_sparsity != 0.0 and isinstance(m, torch.nn.Linear):
            new_m = SupermaskLinear(
                linear_sparsity, False, False, None, None, None,
                m.in_features,
                m.out_features,
                bias=m.bias is not None,
                device=device,
                tile_size=linear_sp_tilesize,
            )
            new_m.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                new_m.bias.data.copy_(m.bias.data)
            sparsified_modules[n] = new_m
            continue

    # add modules to model
    for k, v in sparsified_modules.items():
        sm_name, ch_name = k.rsplit(".", 1)
        sm = model.get_submodule(sm_name)
        sm.add_module(ch_name, v)

        if verbose:
            print(f'sparsified module "{k}" with sparsity={v.sparsity}, tile size={v.tile_size}')

    return model
