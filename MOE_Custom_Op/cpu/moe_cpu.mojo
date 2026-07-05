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
    output: OutputTensor[rank=2, ...],
    lhs: InputTensor[dtype=output.dtype, rank=2, ...],
    rhs: InputTensor[dtype=output.dtype, rank=2, ...]
) raises:
    comptime TILE = 16

    var m = lhs.dim_size(0)
    var k = lhs.dim_size(1)
    var n = rhs.dim_size(1)

    for i0 in range(0, m, TILE):
        var i_end  = min(i0 + TILE, m)
        var i_tile = i_end - i0

        for j0 in range(0, n, TILE):
            var j_end  = min(j0 + TILE, n)
            var j_tile = j_end - j0
            var acc = List[Scalar[output.dtype]](
                length = i_tile * j_tile,
                fill   = Scalar[output.dtype](0)
            )

            for p0 in range(0, k, TILE):
                var p_end = min(p0 + TILE, k)

                for i in range(i0, i_end):
                    var i_local = i - i0
                    for p in range(p0, p_end):
                        var left_val = lhs.load[1](IndexList[2](i, p))

                        for j in range(j0, j_end):
                            acc[i_local * j_tile + (j - j0)] += left_val * rhs.load[1](IndexList[2](p, j))

            for i in range(i0, i_end):
                for j in range(j0, j_end):
                    output.store[1](IndexList[2](i, j), acc[(i - i0) * j_tile + (j - j0)])

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

def expert_ffn_compute(
    output: ManagedTensorSlice[mut=True, dtype=DType.float32, rank=3, ...],
    A: ManagedTensorSlice[dtype=DType.bool,rank=2, ...],
    B: ManagedTensorSlice[dtype=DType.float32,rank=2, ...]
) raises :
    var r = A.dim_size(0)
    var c = A.dim_size(1)
    var emb_dim = B.dim_size(1)

    for i in range(r):
        for j in range(c):
            var flag = A.load[1](IndexList[2](i, j))
            if flag:
                for k in range(emb_dim):
                    var temp = B.load[1](IndexList[2](j, k))
                    output.store[1](IndexList[3](i, j, k), temp)


def silu[dtype: DType](
    output: ManagedTensorSlice[mut=True, dtype=dtype, rank=2, ...],
    A: ManagedTensorSlice[dtype=dtype,rank=2, ...]
) raises:
    var r = A.dim_size(0)
    var c = A.dim_size(1)

    comptime WIDTH = simd_width_of[dtype]()

    for i in range(r):
        var j = 0

        while j + WIDTH <= c:
            var vals = A.load[WIDTH](IndexList[2](i, j))

            var e = vals.cast[DType.float32]()
            var fact = 1.0 + exp(-e)

            var res = vals / fact.cast[dtype]()

            output.store[WIDTH](IndexList[2](i, j), res)

            j += WIDTH
            
        while j < c:
            var val = A.load[1](IndexList[2](i, j))

            var e = val.cast[DType.float32]()
            var fact = 1.0 + exp(-e)

            var res = val / fact.cast[dtype]()

            output.store[1](IndexList[2](i, j), res)
            j += 1

def element_matmul(
    output: ManagedTensorSlice[mut=True, dtype=DType.float32, rank=2, ...],
    A: ManagedTensorSlice[dtype=DType.float32, rank=2, ...],
    B: ManagedTensorSlice[dtype=DType.float32, rank=2, ...],
) raises:
    var r = A.dim_size(0)
    var c = A.dim_size(1)

    comptime WIDTH = simd_width_of[DType.float32]()

    for i in range(r):
        var j = 0

        while j + WIDTH <= c:
            var a = A.load[WIDTH](IndexList[2](i, j))
            var b = B.load[WIDTH](IndexList[2](i, j))

            var res = a * b

            output.store[WIDTH](IndexList[2](i, j), res)

            j += WIDTH

        while j < c:
            var a = A.load[1](IndexList[2](i, j))
            var b = B.load[1](IndexList[2](i, j))

            output.store[1](IndexList[2](i, j), a * b)

            j += 1

def expert_prob_add[expert_idx: Int](
    output: ManagedTensorSlice[mut=True, dtype=DType.float32, rank=2, ...],
    A: ManagedTensorSlice[dtype=DType.float32, rank=2, ...],
    B: ManagedTensorSlice[dtype=DType.float32, rank=2, ...],
    C: ManagedTensorSlice[dtype=DType.int8, rank=2, ...],
) raises:
    
    var r = A.dim_size(0)
    var c = A.dim_size(1)
    var k = B.dim_size(1)

    for i in range(r):
        var is_zero = True
        for j in range(c):
            var val = A.load[1](IndexList[2](i, j))
            if val != 0:
                is_zero = False
                break
        if is_zero:
            continue
        
        var routing_weight: Float32 = 0.0

        for j in range(k):
            var idx = C.load[1](IndexList[2](i, j))

            if Int(idx) == expert_idx:
                routing_weight = B.load[1](IndexList[2](i, j))
                break

        for j in range(c):
            var val = A.load[1](IndexList[2](i, j))
            output.store[1](IndexList[2](i, j), val * routing_weight)

def grouped_matmul[dtype: DType](
    output: OutputTensor[dtype=dtype, rank=2, ...],
    X:      InputTensor[dtype=dtype,  rank=2, ...],
    gate:   InputTensor[dtype=dtype,  rank=2, ...],
    up:     InputTensor[dtype=dtype,  rank=2, ...]
) raises:
    comptime TILE = 16

    var m = X.dim_size(0)
    var k = X.dim_size(1)
    var n = gate.dim_size(1)

    for i0 in range(0, m, TILE):
        var i_end  = min(i0 + TILE, m)
        var i_tile = i_end - i0

        for j0 in range(0, n, TILE):
            var j_end  = min(j0 + TILE, n)
            var j_tile = j_end - j0

            var acc_gate = List[Scalar[dtype]](
                length = i_tile * j_tile,
                fill   = Scalar[dtype](0)
            )
            var acc_up = List[Scalar[dtype]](
                length = i_tile * j_tile,
                fill   = Scalar[dtype](0)
            )

            for p0 in range(0, k, TILE):
                var p_end = min(p0 + TILE, k)

                for i in range(i0, i_end):
                    var i_local = i - i0

                    for p in range(p0, p_end):
                        var x_val = X[i, p]

                        for j in range(j0, j_end):
                            var idx = i_local * j_tile + (j - j0)
                            acc_gate[idx] += x_val * gate[p, j]
                            acc_up[idx]   += x_val * up[p, j]
            for i in range(i0, i_end):
                for j in range(j0, j_end):
                    var idx = (i - i0) * j_tile + (j - j0)
                    output[i, j]     = acc_gate[idx]
                    output[i, j + n] = acc_up[idx]

 
