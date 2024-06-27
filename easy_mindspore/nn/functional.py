from typing import Optional, Tuple, List
import math
import warnings
from mindspore import ops as _ops
from mindspore.ops import Primitive
from ..executor import execute
import easy_mindspore
from easy_mindspore.dtype import typing
from easy_mindspore import BACKEND
from .._tensor import Tensor


def make_tuple(inp):
    if isinstance(inp, tuple):
        return (1, 1, inp[0], inp[1])
    elif isinstance(inp, int):
        return (1, 1, inp, inp)

"""Convolution functions"""
# conv1d


# conv2d
_bias_add = Primitive('BiasAdd')
_bias_add.init_prim_io_names(inputs=['x', 'b'], outputs=['output'])
_bias_add.add_prim_attr('data_format', 'NCHW')
def raw_bias_add(x, y):
    return execute(_bias_add, x, y)

_conv2d = Primitive('Conv2D')
_conv2d.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
_conv2d.add_prim_attr('mode', 1) # only support mode=1
def raw_conv2d(x, w, out_channel, kernel_size, pad_mode="valid", pad=0, stride=1, dilation=1, groups=1, data_format="NCHW"):
    _conv2d.add_prim_attr("out_channel", out_channel)
    _conv2d.add_prim_attr("kernel_size", kernel_size)
    _conv2d.add_prim_attr("pad_mode", pad_mode)
    _conv2d.add_prim_attr("pad", pad)
    _conv2d.add_prim_attr('stride', stride)
    _conv2d.add_prim_attr('dilation', dilation)
    _conv2d.add_prim_attr('group', groups)
    _conv2d.add_prim_attr('groups', groups)
    _conv2d.add_prim_attr('data_format', data_format)
    return execute(_conv2d, x, w)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    weight_shape = weight.shape
    out_channel = weight_shape[0]
    kernel_size = weight_shape[2:4]
    
    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, tuple):
        pad = (padding[0], padding[0], padding[1], padding[1])
    elif isinstance(padding, int):
        pad = (padding,) * 4
    if not isinstance(padding, (int, tuple)):
        pad_mode = padding
        pad = (0,) * 4

    stride = make_tuple(stride)
    dilation = make_tuple(dilation)
    output = raw_conv2d(input, weight, out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
    if bias is not None:
        output = raw_bias_add(output, bias)
    return output

# conv3d


# conv_transpose1d


# conv_transpose2d


# conv_transpose3d


# unfold
_unfold = Primitive('Im2Col')
_unfold.init_prim_io_names(inputs=['x'], outputs=['y'])
def raw_unfold(x, ksizes, strides=1, dilations=1, pads=0):
    _unfold.add_prim_attr('ksizes', ksizes)
    _unfold.add_prim_attr('strides', strides)
    _unfold.add_prim_attr('dilations', dilations)
    _unfold.add_prim_attr('pads', pads)
    _unfold.add_prim_attr('padding_mode', "CALCULATED")
    return execute(_unfold, x)

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    out = raw_unfold(input, kernel_size=kernel_size, dilation=dilation,
                        padding=padding, stride=stride)
    return out.reshape(out.shape[0], -1, out.shape[-1])

# fold

"""Pooling functions"""
# avg_pool1d


# avg_pool2d


# avg_pool3d


# max_pool1d


# max_pool2d
_maxpool = Primitive('MaxPoolWithArgmaxV2')
_maxpool.init_prim_io_names(inputs=["x"], outputs=["output", "argmax"])
_maxpool.add_prim_attr("argmax_type", 4)
def raw_maxpool(x, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
    _maxpool.add_prim_attr("kernel_size", kernel_size)
    _maxpool.add_prim_attr("strides", strides)
    _maxpool.add_prim_attr("pads", pads)
    _maxpool.add_prim_attr("dilation", dilation)
    _maxpool.add_prim_attr("ceil_mode", ceil_mode)
    return execute(_maxpool, x)

def max_pool2d(input, kernel_size, strides=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if strides is None:
        strides = kernel_size
    kernel_size = make_tuple(kernel_size)
    strides = make_tuple(strides)
    pads = make_tuple(padding)
    dilation = make_tuple(dilation)
    out, indices = raw_maxpool(input, kernel_size, strides, pads, dilation, ceil_mode)
    if return_indices:
        return out, indices
    return out

# max_pool3d


# max_unpool1d


# max_unpool2d


# max_unpool3d


# lp_pool1d


# lp_pool2d


# lp_pool3d


# adaptive_max_pool1d


# adaptive_max_pool2d


# adaptive_max_pool3d


# adaptive_avg_pool1d


# adaptive_avg_pool2d


# adaptive_avg_pool3d


# fractional_max_pool2d


# fractional_max_pool3d


"""Attention Mechanisms"""
# scaled_dot_product_attention
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            # proj = linear(q, w, b)
            # # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            # proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            # return proj[0], proj[1], proj[2]
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            # q_proj = linear(q, w_q, b_q)
            # kv_proj = linear(k, w_kv, b_kv)
            # # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            # kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            # return (q_proj, kv_proj[0], kv_proj[1])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched

def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[typing.Type],
        other_name: str,
        target_type: typing.Type,
        check_other: bool = True,
) -> Optional[Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = easy_mindspore.is_floating_point(mask)
        if _mask_dtype != easy_mindspore.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = (
                easy_mindspore.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask

def _none_or_dtype(input: Optional[Tensor]) -> Optional[typing.Type]:
    if input is None:
        return None
    elif isinstance(input, easy_mindspore.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if is_causal:
        # const auto L = query.sym_size(-2), S = key.sym_size(-2);
        # attn_mask = at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
        # attn_mask = convert_boolean_attn_mask(attn_mask, query.dtype());
        pass

    if attn_mask is not None:
        attn = easy_mindspore.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = easy_mindspore.bmm(q, k.transpose(-2, -1))

    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = easy_mindspore.bmm(attn, v)
    return output, attn

"""Non-linear activation functions"""
# threshold


# relu
_relu = Primitive('ReLU')
_relu.init_prim_io_names(inputs=['x'], outputs=['output'])
def relu(x):
    return execute(_relu, x)

# hardtanh


# hardswish


# relu6


# elu


# selu


# celu


# leaky_relu


# prelu


# rrelu


# glu


# gelu
_gelu = Primitive('GeLU')
_gelu.init_prim_io_names(inputs=['x'], outputs=['output'])
def gelu(x, approximate='none'):
    if approximate == 'tanh':
        return execute(_gelu, x)
    else:
        return x * 0.5 * (1.0 + easy_mindspore.ops.erf(x / easy_mindspore.ops.sqrt(2.0)))

# hardshrink


# tanhshrink


# softsign


# softplus


# softmin


# softmax
_softmax = Primitive('Softmax')
_softmax.init_prim_io_names(inputs=['x'], outputs=['output'])
def softmax(input, dim):
    if not isinstance(dim, tuple):
        dim = (dim,)
    _softmax.add_prim_attr('axis', dim)
    return execute(_softmax, input)

# softshrink


# gumbel_softmax


# log_softmax
_log_softmax = Primitive('LogSoftmax')
def log_softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1

    _log_softmax.add_prim_attr('axis', dim)
    out = execute(_log_softmax, input)
    if dtype is not None:
        out = out.to(dtype)
    return out

# tanh

# sigmoid


# hardsigmoid


# silu


# mish


# batch_norm


# group_norm


# instance_norm


# layer_norm
_layer_norm = Primitive("LayerNorm")
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    norm_ndim = len(normalized_shape)
    begin_axis = input.ndim - norm_ndim
    _layer_norm.add_prim_attr("begin_norm_axis", begin_axis)
    _layer_norm.add_prim_attr("begin_params_axis", begin_axis)
    _layer_norm.add_prim_attr("epsilon", eps)

    return execute(_layer_norm, input, weight, bias)

# local_response_norm


# normalize

"""Linear functions"""
# linear
_linear = Primitive('Dense')
_linear.init_prim_io_names(inputs=['x', 'w', 'b'], outputs=["output"])
def linear(input, weight, bias=None):
    if bias is None:
        _linear.add_prim_attr("has_bias", False)
    else:
        _linear.add_prim_attr("has_bias", True)
    return execute(_linear, input, weight, bias)

# bilinear

"""Dropout functions"""
# dropout
_dropout = _ops.Dropout()
def dropout(x: Tensor, p:int=0.5, training:bool=True) -> Tensor:
    """
    http://arxiv.org/abs/1207.0580
    """
    _dropout.add_prim_attr('keep_prob', 1-p)
    if training and p != 0:
        return execute(_dropout, x)[0]
    else:
        return x

# alpha_dropout


# feature_alpha_dropout


# dropout1d


# dropout2d


# dropout3d


"""Sparse functions"""
# embedding

# embedding_bag

# one_hot

"""Distance functions"""

# pairwise_distance


# cosine_similarity


# pdist


"""Loss functions"""
# binary_cross_entropy


# binary_cross_entropy_with_logits


# poisson_nll_loss


# cosine_embedding_loss


# cross_entropy
_softmax_crossentropy = Primitive('SparseSoftmaxCrossEntropyWithLogits')
_softmax_crossentropy.init_prim_io_names(inputs=['features', 'labels'], outputs=['output'])
_softmax_crossentropy.add_prim_attr('sens', 1.0)
def raw_softmax_crossentropy(logits, labels):
    _softmax_crossentropy.add_prim_attr('is_grad', False)
    return execute(_softmax_crossentropy, logits, labels)

_softmax_crossentropy_ascend = Primitive('SparseSoftmaxCrossEntropyWithLogitsV2')
_softmax_crossentropy_ascend.init_prim_io_names(inputs=['features', 'labels'], outputs=['loss', 'backprop'])
def raw_softmax_crossentropy_ascend(logits, labels):
    return execute(_softmax_crossentropy_ascend, logits, labels)

def softmax_cross_entropy(logits, labels):
    if BACKEND == 'Ascend':
        outputs = raw_softmax_crossentropy_ascend(logits, labels)
        return outputs.mean()
    return raw_softmax_crossentropy(logits, labels)


# ctc_loss


# gaussian_nll_loss


# hinge_embedding_loss


# kl_div


# l1_loss


# mse_loss


# margin_ranking_loss


# multilabel_margin_loss


# multilabel_soft_margin_loss


# multi_margin_loss


# nll_loss
_nll_loss = Primitive('NLLLoss')
_nll_loss.init_prim_io_names(inputs=['x', 'target', "weight"], outputs=['loss', 'total_weight'])
def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    if weight is None:
        weight = easy_mindspore.ops.ones(input.shape[-1])
    _nll_loss.add_prim_attr("ignore_index", ignore_index)
    _nll_loss.add_prim_attr("reduction", reduction)
    return execute(_nll_loss, input, target, weight)[0]

# huber_loss


# smooth_l1_loss


# soft_margin_loss


# triplet_margin_loss


# triplet_margin_with_distance_loss


"""Vision functions"""

# pixel_shuffle


# pixel_unshuffle


# pad


# interpolate


# upsample


# upsample_nearest


# upsample_bilinear


# grid_sample


# affine_grid
