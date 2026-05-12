import compiler

from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor, ManagedTensorSlice
from std.utils.index import IndexList

from std.runtime.asyncrt import parallelism_level
from std.algorithm import sync_parallelize, vectorize
from linalg.utils import partition_work

from std.sys.info import simd_width_of

from std.math import sqrt, exp
from std.collections import List
from std.algorithm import parallelize


def matmul(
    output: ManagedTensorSlice[mut = True, ...],
    lhs: ManagedTensorSlice[dtype = output.dtype, rank = 2, ...],
    rhs: ManagedTensorSlice[dtype = output.dtype, rank = 2, ...]
)raises:
    var m = lhs.dim_size(0)
    var k = lhs.dim_size(1)
    var n = rhs.dim_size(1)

    for i in range(m):
        for j in range(n):
            var acc = Scalar[output.dtype](0.0)
            for p in range(k):
                var left_val = lhs.load[1](IndexList[2](i,p))
                var right_val = rhs.load[1](IndexList[2](p,j))
                acc += left_val * right_val
            output.store[1](IndexList[2](i,j), acc)

def top_k[k: Int, dtype: DType](
    output1: ManagedTensorSlice[mut=True, dtype=dtype, rank=2, ...],
    output2: ManagedTensorSlice[mut=True, dtype=DType.int8, rank=2, ...],
    A: ManagedTensorSlice[dtype=dtype, rank=2, ...]
) raises:
    comptime assert dtype.is_floating_point(), "dtype must be floating point"

    var rows = A.dim_size(0)
    var cols = A.dim_size(1)

    for i in range(rows):
        var max1     = Scalar[A.dtype](-1e9)
        var max2     = Scalar[A.dtype](-1e9)
        var max1_idx = 0
        var max2_idx = 0

        for j in range(cols):
            var val = A.load[1](IndexList[2](i, j))
            if val > max1:
                max2     = max1
                max2_idx = max1_idx
                max1     = val
                max1_idx = j
            elif val > max2:
                max2     = val
                max2_idx = j

        var diff     = max2.cast[DType.float32]() - max1.cast[DType.float32]()
        var exponent = exp(diff)
        var new_m1   = Scalar[dtype](1.0 / (1.0 + exponent))
        var new_m2   = Scalar[dtype](exponent / (1.0 + exponent))

        output1.store[1](IndexList[2](i, 0), new_m1)
        output1.store[1](IndexList[2](i, 1), new_m2)
        output2.store[1](IndexList[2](i, 0), Scalar[DType.int8](max1_idx))
        output2.store[1](IndexList[2](i, 1), Scalar[DType.int8](max2_idx))

def mask_experts[dtype: DType](
    output: ManagedTensorSlice[mut=True, dtype=DType.bool, rank=2, ...],
    A: ManagedTensorSlice[dtype=dtype,rank=2, ...]
) raises:
    var r = A.dim_size(0)
    var c = A.dim_size(1)
    
    for i in range(r):
        for j in range(c):
            var expert = Int(A.load[1](IndexList[2](i, j)))
            output.store[1](IndexList[2](expert, i), True)


def silu[dtype: DType](
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=2, ...],
    A: ManagedTensorSlice[dtype=dtype,rank=2, ...]
) raises:
    var r = A.dim_size(0)
    var c = A.dim_size(1)

    for i in range(r):
        for j in range(c):
            var val = A.load[1](IndexList[2](i,j))
            var e = val.cast[DType.float32]()
            var fact = 1.0 + exp(-e)

            var res = val / fact.cast[dtype]()

            output.store[1](IndexList[2](i,j), res)
