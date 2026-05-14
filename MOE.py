from max.graph import Graph, TensorType, ops, DeviceRef
from max.engine import InferenceSession
from max.experimental.tensor import Tensor
from max.driver import Accelerator, CPU, Buffer
from pathlib import Path
from max.dtype import DType
import time
import math
import numpy as np


#No of tokens
M = 4
# Embedding Dimension of each token
K = 8
# No of Experts that are been available
N = 3
# No of experts need to be selected 
nk = 2
#Hidden dimension size of each expert
D = 2 * K
device = CPU()

mojo_kernel = Path(__file__).parent / "MOE_Custom_Op"

def build_graph():
    x_type = TensorType(
        dtype = DType.float32,
        shape = [M, K],
        device = device
    )

    router_mat_type = TensorType(
        dtype = DType.float32,
        shape = [K, N],
        device = device
    )
    w_gate_type = TensorType(
        dtype = DType.float32,
        shape = [N, K, D],
        device = device
    )
    w_up_type = TensorType(
        dtype = DType.float32,
        shape = [N, K, D],
        device = device
    )
    w_down_type = TensorType(
        dtype = DType.float32,
        shape = [N, D, K],
        device = device 
    )
    t_idx_type = TensorType(
        dtype = DType.int8,
        shape = [M, nk],
        device = device
    )
    t_val_type = TensorType(
        dtype = DType.float32,
        shape = [M, nk],
        device = device
    )
    logits_out_type = TensorType(
        dtype = DType.float32,
        shape = [M, N],
        device = device
    )
    mask_type = TensorType(
        dtype = DType.bool,
        shape = [N, M],
        device = device
    )
    expert_ffn_mat_type = TensorType(
        dtype = DType.float32,
        shape = [N, M, K],
        device = device
    )
    hidden_dim_type = TensorType(
        dtype = DType.float32,
        shape = [M, D],
        device = device
    )
    with Graph(
        "MOE_Graph",
        input_types = [x_type, router_mat_type, w_gate_type, w_up_type, w_down_type],
        custom_extensions = [mojo_kernel]
    ) as graph:
        
        a, b, w_gate, w_up, w_down = graph.inputs

        logits = ops.custom(
            "matmul",
            device = a.device,
            values = [a, b],
            out_types = [logits_out_type]
        )[0].tensor

        topk_val, topk_idx = ops.custom(
            "top_k",
            device = a.device,
            values = [logits],
            out_types = [t_val_type, t_idx_type],
            parameters= {"k": nk},
        )

        mask_matrix = ops.custom(
            "mask_of_experts",
            device = device,
            values = [topk_idx],
            out_types = [mask_type], 
        )[0].tensor

        expert_ffn_matrix = ops.custom(
            "expert_ffn_matrix_compute",
            device = device,
            values = [mask_matrix, a],
            out_types = [expert_ffn_mat_type], 
        )[0].tensor

        Moe_output = ops.custom(
            "initialize_moe_output",
            device = device,
            values = [a],
            out_types = [x_type]
        )[0].tensor

        for i in range(N):
            gate_up_out = ops.custom(
                "matmul",
                device = device,
                values = [expert_ffn_matrix[i, :, :], w_gate[i, :, :]],
                out_types = [hidden_dim_type]
            )[0].tensor

            # print("Gate Up Output")
            # print(gate_up_out)

            silu_out = ops.custom(
                "Silu",
                device = device,
                values = [gate_up_out],
                out_types = [hidden_dim_type]
            )[0].tensor

            # print("Silu Output")
            # print(silu_out) 

            up_proj_out = ops.custom(
                "matmul",
                device = device,
                values = [expert_ffn_matrix[i, :, :], w_up[i, :, :]],
                out_types = [hidden_dim_type]
            )[0].tensor
            # print("Up Projection Output")
            # print(up_proj_out)

            dot_product = ops.custom(
                "elementwise_mul",
                device = device,
                values = [silu_out, up_proj_out],
                out_types = [hidden_dim_type]
            )[0].tensor
            # print("Dot Product Output")
            # print(dot_product)

            gate_down_out = ops.custom(
                "matmul",
                device = device,
                values = [dot_product, w_down[i, :, :]],
                out_types = [x_type]
            )[0].tensor

            # print("Gate Down Output")
            # print(gate_down_out)

            # apply that probability

            expert_output_mat = ops.custom(
                "apply_expert_prob",
                device = device,
                values = [gate_down_out, topk_val, topk_idx],
                out_types = [x_type],
                parameters = {"expert_idx": i}
            )[0].tensor
            
            # print("Expert Output Matrix")
            # print(expert_output_mat)

            # Moe_output = Moe_output + name
            
            Moe_output = Moe_output + expert_output_mat

        # graph.output(logits, topk_val, topk_idx, mask_matrix, expert_ffn_matrix)
        graph.output(Moe_output)
    
    return graph

graph = build_graph()
session = InferenceSession(
    devices = [device],
)
model = session.load(graph)

a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, N).astype(np.float32)

w_gate = np.random.rand(N, K, D).astype(np.float32)
w_up   = np.random.rand(N, K, D).astype(np.float32)
w_down = np.random.rand(N, D, K).astype(np.float32)

A = Buffer.from_numpy(a).to(device)
B = Buffer.from_numpy(b).to(device)
W_GATE = Buffer.from_numpy(w_gate).to(device)
W_UP = Buffer.from_numpy(w_up).to(device)
W_DOWN = Buffer.from_numpy(w_down).to(device)

_ = model.execute(A, B, W_GATE, W_UP, W_DOWN)

st = time.perf_counter()
output = model.execute(A, B, W_GATE, W_UP, W_DOWN)
ed = time.perf_counter()
print(f"Execution time for Build -In Kernel : {(ed - st)* 1e3:.3f} ms")
print()
# matmul = output[0].to(CPU()).to_numpy()
# topk = output[1].to(CPU()).to_numpy()
# topk_idx = output[2].to(CPU()).to_numpy()
# mask_matrix = output[3].to(CPU()).to_numpy()
# expffn = output[4].to(CPU()).to_numpy()

moe_output = output[0].to(CPU()).to_numpy()


# print("Output")
# print(matmul)
# print("Top K Values")
# print(topk)
# print("Top K Indices")
# print(topk_idx)
# print("Masked Matrix")
# print(mask_matrix)
# print("Expert FFN Matrix")
# print(expffn)

print("MOE Output")
print(moe_output)
print()
