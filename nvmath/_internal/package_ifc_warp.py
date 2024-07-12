# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to Warp operations.
"""

__all__ = ['WarpPackage']

import warp

from .package_ifc import Package


class WarpPackage(Package):

    @staticmethod
    def get_device(device_id):
        if isinstance(device_id, int):
            if device_id < 0:
                return warp.get_cpu_device()
            return warp.get_cuda_device(device_id)
        return warp.get_device(device_id)

    @staticmethod
    def get_current_stream(device_id):
        return warp.get_stream(WarpPackage.get_device(device_id))

    @staticmethod
    def to_stream_pointer(stream):
        return stream.cuda_stream

    @staticmethod
    def to_stream_context(stream):
        return warp.ScopedStream(stream)

    @classmethod
    def create_external_stream(device_id, stream_ptr):
        return warp.context.Stream(device=WarpPackage.get_device(device_id), cuda_stream=stream_ptr)

    @staticmethod
    def create_stream(device_id):
        stream = warp.context.Stream(device=WarpPackage.get_device(device_id))
        return stream
