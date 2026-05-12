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
    with Graph(
        "MOE_Graph",
        input_types = [x_type, router_mat_type],
        custom_extensions = [mojo_kernel]
    ) as graph:
        
        a, b = graph.inputs

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
            parameters={"k": nk},
        )

        mask_matrix = ops.custom(
            "mask_of_experts",
            device = device,
            values = [topk_idx],
            out_types = [mask_type], 
        )[0].tensor

        graph.output(logits, topk_val, topk_idx, mask_matrix)
    
    return graph

graph = build_graph()
session = InferenceSession(
    devices = [device],
)
model = session.load(graph)

a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, N).astype(np.float32)

A = Buffer.from_numpy(a).to(device)
B = Buffer.from_numpy(b).to(device)

_ = model.execute(A, B)

st = time.perf_counter()
output = model.execute(A, B)
ed = time.perf_counter()
print(f"Execution time for Build -In Kernel : {(ed - st)* 1e3:.3f} ms")
print()
matmul = output[0].to(CPU()).to_numpy()
topk = output[1].to(CPU()).to_numpy()
topk_idx = output[2].to(CPU()).to_numpy()
mask_matrix = output[3].to(CPU()).to_numpy()


print("Output")
print(matmul)
print("Top K Values")
print(topk)
print("Top K Indices")
print(topk_idx)
print("Masked Matrix")
print(mask_matrix)
print()
