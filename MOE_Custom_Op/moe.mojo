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

from .cpu.moe_cpu import matmul, top_k, mask_experts, expert_ffn_compute, silu, element_matmul, expert_prob_add


@compiler.register("matmul")
struct MixtureOfExpert:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        A: InputTensor[dtype=output.dtype, rank=2, ...],
        B: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    )raises :

        matmul(
            output,
            A,
            B
        )

@compiler.register("top_k")
struct TopK:
    @staticmethod
    def execute[target: StaticString, k: Int](
        output1: OutputTensor[dtype=DType.float32, rank=2, ...],
        output2: OutputTensor[dtype=DType.int8, rank=2, ...],
        A: InputTensor[dtype=output1.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    )raises :

        top_k[k](
            output1,
            output2,
            A
        )

@compiler.register("mask_of_experts")
struct MaskOfExpert:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.bool, rank=2, ...],
        A: InputTensor[dtype=DType.int8, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:

        mask_experts(
            output,
            A
        )

@compiler.register("expert_ffn_matrix_compute")
struct ExpertFFNMatrix:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=3, ...],
        A: InputTensor[dtype=DType.bool, rank=2, ...],
        B: InputTensor[dtype=DType.float32, rank=2, ...],
    ) raises :
        
        expert_ffn_compute(
            output,
            A,
            B
        )

@compiler.register("Silu")
struct SILU:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        A: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:

        silu(
            output,
            A
        )
        
@compiler.register("elementwise_mul")
struct ElementWiseMatmul:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        A: InputTensor[dtype=DType.float32, rank=2, ...],
        B: InputTensor[dtype=DType.float32, rank=2, ...],
    ) raises:

        element_matmul(
            output,
            A,
            B
        )

@compiler.register("apply_expert_prob")
struct ExpertProbAdd:
    @staticmethod
    def execute[target: StaticString, expert_idx: Int](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        A: InputTensor[dtype=DType.float32, rank=2, ...],
        B: InputTensor[dtype=DType.float32, rank=2, ...],
        C: InputTensor[dtype=DType.int8, rank=2, ...],
    ) raises :
        
        expert_prob_add[
            expert_idx
        ](
            output,
            A,
            B,
            C
        )

@compiler.register("initialize_moe_output")
struct InitializeMoeOutput:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        A: InputTensor[dtype=DType.float32, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
    
        pass