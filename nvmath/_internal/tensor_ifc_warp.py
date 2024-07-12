# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Warp array objects.
"""

__all__ = ['WarpTensor']

import warp

from .package_ifc import StreamHolder
from .tensor_ifc import Tensor

class WarpTensor(Tensor):
    """
    Tensor wrapper for Warp arrays.
    """
    name = 'warp'
    module = warp
    name_to_dtype = Tensor.create_name_dtype_map(conversion_function=lambda name: getattr(warp.types, name), exception_type=AttributeError)

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.ptr

    @property
    def device(self):
        return str(self.tensor.device).split(':')[0]

    @property
    def device_id(self):
        return self.tensor.device.ordinal

    @property
    def dtype(self):
        """Name of the data type"""
        s = str(self.tensor.dtype)
        if s.startswith("<class '"):
            s = s[8:-2]
        return s.split('.')[-1]

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def strides(self):
        dtype_size = warp.types.type_size_in_bytes(self.tensor.dtype)
        return [d // dtype_size for d in self.tensor.strides]

    def numpy(self, stream_holder=StreamHolder()):
        return self.tensor.numpy()

    @staticmethod
    def get_device(device_id):
        if device_id is None:
            return warp.get_preferred_device()
        if isinstance(device_id, int):
            if device_id < 0:
                return warp.get_cpu_device()
            return warp.get_cuda_device(device_id)
        return warp.get_device(device_id)

    @classmethod
    def empty(cls, shape, **context):
        """
        Create an empty tensor of the specified shape and data type on the specified device (None, 'cpu', or device id).
        """
        name = context.get('dtype', 'float32')
        dtype = WarpTensor.name_to_dtype[name]
        device = WarpTensor.get_device(context.get('device', None))
        strides = context.get('strides', None)
        
        dtype_size = warp.types.type_size_in_bytes(dtype)
        strides = [d * dtype_size for d in strides] if strides is not None else None
        return warp.empty(shape, dtype=dtype, device=device, strides=strides)

    def to(self, device='cpu', stream_holder=StreamHolder()):
        """
        Create a copy of the tensor on the specified device (integer or
          'cpu'). Copy to  Numpy ndarray if CPU, otherwise return Cupy type.
        """
        if not (device == 'cpu' or isinstance(device, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device}'.")

        with stream_holder.ctx:
            tensor_device = self.tensor.to(device=device)

        return tensor_device

    def copy_(self, src, stream_holder=StreamHolder()):
        """
        Inplace copy of src (copy the data from src into self).
        """
        with stream_holder.ctx:
            warp.copy(self.tensor, src)

        return self.tensor

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, warp.array)

