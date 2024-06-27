from copy import deepcopy
from mindspore._c_expression import Tensor as MSTensor
from mindspore._c_expression import TensorNode
from mindspore.common._register_for_tensor import tensor_operator_registry
import numpy as np
import easy_mindspore
from . import dtype

class Tensor:
    tensor = None
    stub = None
    def __init__(self, input, dtype=None): # pylint: disable=super-init-not-called
        if isinstance(input, TensorNode):
            self.stub = input
        elif isinstance(input, MSTensor):
            self.tensor = input
        else:
            if isinstance(input, np.ndarray):
                dtype = dtype_infer[input.dtype.type]
            else:
                dtype = dtype_infer[type(input)]
            self.tensor = MSTensor(input, dtype=dtype)

    @property
    def data(self):
        if self.tensor is not None:
            return self.tensor
        return self.stub_sync()

    @data.setter
    def data(self, other):
        if isinstance(other, Tensor):
            self.stub = other.stub
            self.tensor = other.tensor
        elif isinstance(other, MSTensor):
            self.tensor = other.tensor
            self.stub = None

    @property
    def shape(self):
        """shape stub."""
        if self.stub:
            if not hasattr(self, "stub_shape"):
                self.stub_shape = self.stub.get_shape()
            return self.stub_shape
        return tuple(self.tensor.shape)

    @property
    def dtype(self):
        """dtype stub."""
        if self.stub:
            if not hasattr(self, "stub_dtype"):
                self.stub_dtype = self.stub.get_dtype()
            return self.stub_dtype
        return self.tensor.dtype

    def stub_sync(self):
        """sync real tensor."""
        if self.stub:
            val = self.stub.get_value()
            self.tensor = MSTensor(val)
            if hasattr(self, "member_cache"):
                for k, v in self.member_cache.items():
                    setattr(self.tensor, k, v)
            self.stub = None
        return self.tensor

    def __hash__(self):
        return hash(id(self))

    def __len__(self):
        if self.shape == ():
            return 1
        return self.shape[0]

    def set_data(self, data):
        self.copy_(data)

    def __repr__(self) -> str:
        self.data.data_sync(True)
        return self.data.__repr__()

    def __format__(self, format_spec):
        return np.ndarray.__format__(self.numpy(), format_spec)

    def __getitem__(self, slices):
        slices = easy_mindspore.ops.utils.slice_helper(slices)
        return easy_mindspore.ops.array.strided_slice(self, *slices)

    def __setitem__(self, key, value):
        """"""
        # return easy_mindspore.ops.array.setitem(self, key, value)
        tensor_operator_registry.get("__setitem__")(self.data, key, value)

    def __add__(self, other):
        return easy_mindspore.ops.add(self, other)

    def __radd__(self, other):
        return easy_mindspore.ops.add(other, self)

    def __truediv__ (self, other):
        return easy_mindspore.ops.div(self, other)

    def __rtruediv__ (self, other):
        return easy_mindspore.ops.div(other, self)

    def __ne__(self, other):
        return easy_mindspore.ops.ne(self, other)

    def __neg__(self):
        return easy_mindspore.ops.neg(self)

    def __mul__(self, other):
        return easy_mindspore.ops.mul(self, other)

    def __rmul__(self, other):
        return easy_mindspore.ops.mul(other, self)

    def __pow__(self, other):
        return easy_mindspore.ops.pow(self, other)

    def __sub__(self, other):
        return easy_mindspore.ops.sub(self, other)

    def __rsub__(self, other):
        return easy_mindspore.ops.sub(other, self)

    def __eq__(self, other):
        return easy_mindspore.ops.eq(self, other)

    # Tensor.new_tensor
    def new_tensor(self, data, *, dtype=None):
        if dtype is not None:
            dtype = self.dtype
        return Tensor(data, dtype)

    # Tensor.new_full
    def new_full(self, size, fill_value, *, dtype=None):
        return easy_mindspore.ops.full

    # Tensor.new_empty


    # Tensor.new_ones


    # Tensor.new_zeros


    # Tensor.ndim
    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    # Tensor.real


    # Tensor.imag

    # Tensor.nbytes


    # Tensor.itemsize


    # Tensor.abs


    # Tensor.abs_


    # Tensor.absolute


    # Tensor.absolute_


    # Tensor.acos


    # Tensor.acos_


    # Tensor.arccos


    # Tensor.arccos_


    # Tensor.add
    def add(self, other, *, alpha=1):
        return easy_mindspore.ops.add(self, other, alpha=alpha)

    # Tensor.add_
    def add_(self, other, *, alpha=1):
        out = easy_mindspore.ops.add(self, other, alpha=alpha)
        self.data = out

    # Tensor.addbmm


    # Tensor.addbmm_


    # Tensor.addcdiv

    # Tensor.addcdiv_


    # Tensor.addcmul


    # Tensor.addcmul_


    # Tensor.addmm


    # Tensor.addmm_


    # Tensor.sspaddmm


    # Tensor.addmv


    # Tensor.addmv_


    # Tensor.addr


    # Tensor.addr_


    # Tensor.adjoint


    # Tensor.allclose


    # Tensor.amax


    # Tensor.amin


    # Tensor.aminmax


    # Tensor.angle


    # Tensor.apply_


    # Tensor.argmax
    def argmax(self, dim=None, keepdim=False):
        out = easy_mindspore.ops.argmax(self, dim, keepdim)
        return out

    # Tensor.argmin


    # Tensor.argsort


    # Tensor.argwhere


    # Tensor.asin


    # Tensor.asin_


    # Tensor.arcsin


    # Tensor.arcsin_


    # Tensor.as_strided


    # Tensor.atan


    # Tensor.atan_


    # Tensor.arctan


    # Tensor.arctan_


    # Tensor.atan2


    # Tensor.atan2_


    # Tensor.arctan2


    # Tensor.arctan2_


    # Tensor.all


    # Tensor.any

    # Tensor.baddbmm


    # Tensor.baddbmm_


    # Tensor.bernoulli


    # Tensor.bernoulli_


    # Tensor.bfloat16


    # Tensor.bincount


    # Tensor.bitwise_not


    # Tensor.bitwise_not_


    # Tensor.bitwise_and


    # Tensor.bitwise_and_


    # Tensor.bitwise_or


    # Tensor.bitwise_or_


    # Tensor.bitwise_xor


    # Tensor.bitwise_xor_


    # Tensor.bitwise_left_shift


    # Tensor.bitwise_left_shift_


    # Tensor.bitwise_right_shift


    # Tensor.bitwise_right_shift_


    # Tensor.bmm


    # Tensor.bool


    # Tensor.byte


    # Tensor.broadcast_to
    def broadcast_to(self, shape):
        raise NotImplementedError
        # return easy_mindspore.ops.broadcast_to(self, shape)

    # Tensor.cauchy_


    # Tensor.ceil


    # Tensor.ceil_


    # Tensor.char


    # Tensor.cholesky


    # Tensor.cholesky_inverse


    # Tensor.cholesky_solve


    # Tensor.chunk


    # Tensor.clamp


    # Tensor.clamp_


    # Tensor.clip


    # Tensor.clip_


    # Tensor.clone
    def clone(self):
        return deepcopy(self)

    # Tensor.contiguous


    # Tensor.copy_
    def copy_(self, value):
        if isinstance(value, Tensor):
            self.stub = value.stub
            self.tensor = value.tensor
        elif isinstance(value, MSTensor):
            self.stub = None
            self.tensor = value
        else:
            raise ValueError(f'not support type: {type(value)}')

    # Tensor.conj


    # Tensor.conj_physical


    # Tensor.conj_physical_


    # Tensor.resolve_conj


    # Tensor.resolve_neg


    # Tensor.copysign


    # Tensor.copysign_


    # Tensor.cos


    # Tensor.cos_


    # Tensor.cosh


    # Tensor.cosh_


    # Tensor.corrcoef


    # Tensor.count_nonzero


    # Tensor.cov


    # Tensor.acosh


    # Tensor.acosh_


    # Tensor.arccosh


    # Tensor.arccosh_


    # Tensor.cpu


    # Tensor.cross


    # Tensor.cuda


    # Tensor.logcumsumexp


    # Tensor.cummax


    # Tensor.cummin


    # Tensor.cumprod


    # Tensor.cumprod_


    # Tensor.cumsum

    # Tensor.cumsum_


    # Tensor.chalf


    # Tensor.cfloat


    # Tensor.cdouble


    # Tensor.data_ptr


    # Tensor.deg2rad


    # Tensor.dequantize


    # Tensor.det


    # Tensor.dense_dim


    # Tensor.detach


    # Tensor.detach_


    # Tensor.diag


    # Tensor.diag_embed


    # Tensor.diagflat


    # Tensor.diagonal


    # Tensor.diagonal_scatter

    # Tensor.fill_diagonal_


    # Tensor.fmax


    # Tensor.fmin


    # Tensor.diff


    # Tensor.digamma


    # Tensor.digamma_


    # Tensor.dim


    # Tensor.dim_order


    # Tensor.dist


    # Tensor.div


    # Tensor.div_


    # Tensor.divide


    # Tensor.divide_


    # Tensor.dot


    # Tensor.double


    # Tensor.dsplit


    # Tensor.element_size


    # Tensor.eq


    # Tensor.eq_


    # Tensor.equal


    # Tensor.erf


    # Tensor.erf_


    # Tensor.erfc


    # Tensor.erfc_


    # Tensor.erfinv


    # Tensor.erfinv_


    # Tensor.exp


    # Tensor.exp_


    # Tensor.expm1


    # Tensor.expm1_


    # Tensor.expand
    def expand(self, *size):
        if len(size) == 1:
            size = size[0]
        return self.broadcast_to(size)

    # Tensor.expand_as


    # Tensor.exponential_


    # Tensor.fix


    # Tensor.fix_


    # Tensor.fill_


    # Tensor.flatten
    def flatten(self, start_dim=0, end_dim=-1):
        return easy_mindspore.ops.flatten(self, start_dim, end_dim)

    # Tensor.flip


    # Tensor.fliplr


    # Tensor.flipud


    # Tensor.float


    # Tensor.float_power


    # Tensor.float_power_


    # Tensor.floor


    # Tensor.floor_


    # Tensor.floor_divide


    # Tensor.floor_divide_


    # Tensor.fmod


    # Tensor.fmod_


    # Tensor.frac


    # Tensor.frac_


    # Tensor.frexp


    # Tensor.gather


    # Tensor.gcd


    # Tensor.gcd_


    # Tensor.ge


    # Tensor.ge_


    # Tensor.greater_equal


    # Tensor.greater_equal_


    # Tensor.geometric_


    # Tensor.geqrf


    # Tensor.ger


    # Tensor.get_device


    # Tensor.gt


    # Tensor.gt_


    # Tensor.greater


    # Tensor.greater_


    # Tensor.half


    # Tensor.hardshrink


    # Tensor.heaviside


    # Tensor.histc


    # Tensor.histogram


    # Tensor.hsplit


    # Tensor.hypot


    # Tensor.hypot_


    # Tensor.i0


    # Tensor.i0_


    # Tensor.igamma


    # Tensor.igamma_


    # Tensor.igammac


    # Tensor.igammac_


    # Tensor.index_add_


    # Tensor.index_add


    # Tensor.index_copy_


    # Tensor.index_copy


    # Tensor.index_fill_


    # Tensor.index_fill


    # Tensor.index_put_


    # Tensor.index_put


    # Tensor.index_reduce_


    # Tensor.index_reduce

    # Tensor.index_select


    # Tensor.indices


    # Tensor.inner


    # Tensor.int


    # Tensor.int_repr


    # Tensor.inverse


    # Tensor.isclose


    # Tensor.isfinite


    # Tensor.isinf


    # Tensor.isposinf


    # Tensor.isneginf


    # Tensor.isnan


    # Tensor.is_contiguous


    # Tensor.is_complex


    # Tensor.is_conj


    # Tensor.is_floating_point


    # Tensor.is_inference


    # Tensor.is_leaf


    # Tensor.is_pinned


    # Tensor.is_set_to


    # Tensor.is_shared


    # Tensor.is_signed


    # Tensor.is_sparse


    # Tensor.istft


    # Tensor.isreal


    # Tensor.item


    # Tensor.kthvalue


    # Tensor.lcm


    # Tensor.lcm_


    # Tensor.ldexp


    # Tensor.ldexp_


    # Tensor.le


    # Tensor.le_


    # Tensor.less_equal


    # Tensor.less_equal_


    # Tensor.lerp


    # Tensor.lerp_


    # Tensor.lgamma


    # Tensor.lgamma_


    # Tensor.log


    # Tensor.log_


    # Tensor.logdet


    # Tensor.log10


    # Tensor.log10_


    # Tensor.log1p


    # Tensor.log1p_


    # Tensor.log2


    # Tensor.log2_


    # Tensor.log_normal_


    # Tensor.logaddexp


    # Tensor.logaddexp2


    # Tensor.logsumexp


    # Tensor.logical_and


    # Tensor.logical_and_


    # Tensor.logical_not


    # Tensor.logical_not_


    # Tensor.logical_or


    # Tensor.logical_or_


    # Tensor.logical_xor


    # Tensor.logical_xor_


    # Tensor.logit


    # Tensor.logit_


    # Tensor.long


    # Tensor.lt


    # Tensor.lt_


    # Tensor.less


    # Tensor.less_


    # Tensor.lu


    # Tensor.lu_solve


    # Tensor.as_subclass


    # Tensor.map_


    # Tensor.masked_scatter_


    # Tensor.masked_scatter


    # Tensor.masked_fill_


    # Tensor.masked_fill


    # Tensor.masked_select


    # Tensor.matmul


    # Tensor.matrix_power


    # Tensor.matrix_exp


    # Tensor.max


    # Tensor.maximum


    # Tensor.mean


    # Tensor.module_load


    # Tensor.nanmean


    # Tensor.median


    # Tensor.nanmedian


    # Tensor.min


    # Tensor.minimum


    # Tensor.mm


    # Tensor.smm


    # Tensor.mode


    # Tensor.movedim


    # Tensor.moveaxis


    # Tensor.msort


    # Tensor.mul


    # Tensor.mul_


    # Tensor.multiply


    # Tensor.multiply_


    # Tensor.multinomial


    # Tensor.mv


    # Tensor.mvlgamma


    # Tensor.mvlgamma_


    # Tensor.nansum


    # Tensor.narrow


    # Tensor.narrow_copy


    # Tensor.ndimension


    # Tensor.nan_to_num


    # Tensor.nan_to_num_


    # Tensor.ne


    # Tensor.ne_


    # Tensor.not_equal


    # Tensor.not_equal_


    # Tensor.neg


    # Tensor.neg_


    # Tensor.negative


    # Tensor.negative_


    # Tensor.nelement


    # Tensor.nextafter


    # Tensor.nextafter_


    # Tensor.nonzero


    # Tensor.norm


    # Tensor.normal_


    # Tensor.numel


    # Tensor.numpy
    def numpy(self):
        return self.data.asnumpy()

    def asnumpy(self):
        return self.numpy()


    # Tensor.orgqr


    # Tensor.ormqr


    # Tensor.outer


    # Tensor.permute
    def permute(self, *dims):
        return easy_mindspore.ops.permute(self, dims)

    # Tensor.pin_memory


    # Tensor.pinverse


    # Tensor.polygamma


    # Tensor.polygamma_


    # Tensor.positive


    # Tensor.pow


    # Tensor.pow_


    # Tensor.prod


    # Tensor.put_


    # Tensor.qr


    # Tensor.qscheme


    # Tensor.quantile


    # Tensor.nanquantile


    # Tensor.q_scale


    # Tensor.q_zero_point


    # Tensor.q_per_channel_scales


    # Tensor.q_per_channel_zero_points


    # Tensor.q_per_channel_axis


    # Tensor.rad2deg


    # Tensor.random_


    # Tensor.ravel


    # Tensor.reciprocal


    # Tensor.reciprocal_


    # Tensor.record_stream


    # Tensor.register_hook


    # Tensor.register_post_accumulate_grad_hook


    # Tensor.remainder


    # Tensor.remainder_


    # Tensor.renorm


    # Tensor.renorm_


    # Tensor.repeat


    # Tensor.repeat_interleave


    # Tensor.reshape
    def reshape(self, *shape):
        return easy_mindspore.ops.reshape(self, *shape)

    # Tensor.reshape_as


    # Tensor.resize_


    # Tensor.resize_as_


    # Tensor.retain_grad


    # Tensor.retains_grad


    # Tensor.roll


    # Tensor.rot90


    # Tensor.round


    # Tensor.round_


    # Tensor.rsqrt


    # Tensor.rsqrt_


    # Tensor.scatter


    # Tensor.scatter_


    # Tensor.scatter_add_


    # Tensor.scatter_add


    # Tensor.scatter_reduce_


    # Tensor.scatter_reduce


    # Tensor.select


    # Tensor.select_scatter


    # Tensor.set_


    # Tensor.share_memory_


    # Tensor.short


    # Tensor.sigmoid


    # Tensor.sigmoid_


    # Tensor.sign


    # Tensor.sign_


    # Tensor.signbit


    # Tensor.sgn


    # Tensor.sgn_


    # Tensor.sin


    # Tensor.sin_


    # Tensor.sinc


    # Tensor.sinc_


    # Tensor.sinh


    # Tensor.sinh_


    # Tensor.asinh


    # Tensor.asinh_


    # Tensor.arcsinh


    # Tensor.arcsinh_


    # Tensor.shape


    # Tensor.size
    def size(self, dim=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    # Tensor.slogdet


    # Tensor.slice_scatter


    # Tensor.softmax


    # Tensor.sort


    # Tensor.split


    # Tensor.sparse_mask


    # Tensor.sparse_dim


    # Tensor.sqrt


    # Tensor.sqrt_


    # Tensor.square


    # Tensor.square_


    # Tensor.squeeze


    # Tensor.squeeze_


    # Tensor.std


    # Tensor.stft


    # Tensor.storage


    # Tensor.untyped_storage


    # Tensor.storage_offset


    # Tensor.storage_type


    # Tensor.stride


    # Tensor.sub


    # Tensor.sub_


    # Tensor.subtract


    # Tensor.subtract_


    # Tensor.sum
    def sum(self, dim=None, keepdim=False, dtype=None):
        return easy_mindspore.ops.sum(self, dim, keepdim, dtype=dtype)

    # Tensor.sum_to_size


    # Tensor.svd


    # Tensor.swapaxes


    # Tensor.swapdims


    # Tensor.t


    # Tensor.t_


    # Tensor.tensor_split


    # Tensor.tile


    # Tensor.to
    def to(self, dtype):
        if dtype is None:
            return self
        return easy_mindspore.ops.cast(self, dtype)

    # Tensor.take


    # Tensor.take_along_dim


    # Tensor.tan


    # Tensor.tan_


    # Tensor.tanh


    # Tensor.tanh_


    # Tensor.atanh


    # Tensor.atanh_


    # Tensor.arctanh


    # Tensor.arctanh_


    # Tensor.tolist


    # Tensor.topk


    # Tensor.to_dense


    # Tensor.to_sparse


    # Tensor.to_sparse_csr


    # Tensor.to_sparse_csc


    # Tensor.to_sparse_bsr


    # Tensor.to_sparse_bsc


    # Tensor.trace


    # Tensor.transpose
    def transpose(self, dim0, dim1):
        return easy_mindspore.ops.transpose(self, dim0, dim1)

    # Tensor.transpose_


    # Tensor.triangular_solve


    # Tensor.tril


    # Tensor.tril_


    # Tensor.triu


    # Tensor.triu_


    # Tensor.true_divide


    # Tensor.true_divide_


    # Tensor.trunc


    # Tensor.trunc_


    # Tensor.type


    # Tensor.type_as


    # Tensor.unbind


    # Tensor.unflatten
    def unflatten(self, dim, sizes):
        return easy_mindspore.ops.unflatten(self, dim, sizes)

    # Tensor.unfold


    # Tensor.uniform_


    # Tensor.unique


    # Tensor.unique_consecutive


    # Tensor.unsqueeze


    # Tensor.unsqueeze_


    # Tensor.values


    # Tensor.var


    # Tensor.vdot


    # Tensor.view
    def view(self, *shape):
        return self.reshape(*shape)

    # Tensor.view_as


    # Tensor.vsplit


    # Tensor.where


    # Tensor.xlogy


    # Tensor.xlogy_


    # Tensor.zero_

dtype_infer = {
    np.float64: dtype.float32,
    np.float32: dtype.float32,
    np.float16: dtype.float16,
    np.int16: dtype.int16,
    np.int32: dtype.int32,
    np.int64: dtype.int32,
    np.int8: dtype.int8,
    int: dtype.int32,
    float: dtype.float32,
    np.bool_: dtype.bool,
    bool: dtype.bool,
    list: None,
    np.longlong: dtype.long
}

def tensor(data, *, dtype=None):
    return Tensor(data, dtype)

def is_tensor(x):
    return isinstance(x, Tensor)


__all__ = ['tensor', 'is_tensor', 'Tensor']
