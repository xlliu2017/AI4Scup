import torch
import torch_tensorrt
import torchvision.models as models
import time

# ==========================================
# 1. SETUP: Load & Compile
# ==========================================
device = torch.device("cuda")

# Load your model (Example: ResNet50)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().to(device)

# Define static input shape (Critical for TensorRT efficiency)
BATCH_SIZE = 1
input_shape = [BATCH_SIZE, 3, 224, 224]
dummy_input = torch.randn(input_shape).to(device)

print("1. Compiling with TensorRT (FP16)...")
# OPTIMIZATION A: TensorRT
# Fuses layers (Conv+ReLU) and uses FP16 Tensor Cores
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=input_shape,
        opt_shape=input_shape,
        max_shape=input_shape,
        dtype=torch.float32
    )],
    enabled_precisions={torch.half}, # Enable FP16
    truncate_long_and_double=True
)

# ==========================================
# 2. OPTIMIZATION B: CUDA Graphs
# ==========================================
print("2. Capturing CUDA Graph...")

# Create static memory buffers (Graphs require fixed memory addresses)
static_input = torch.randn(input_shape, device=device)
# static_output will be captured from the model's return value

# Warmup (Crucial to initialize internal TRT buffers)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        _ = trt_model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture the Graph
# We record the kernel execution sequence into 'g'
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = trt_model(static_input)

def run_inference(new_data):
    """
    Optimized inference function.
    1. Copies new data to the static memory buffer.
    2. Replays the recorded graph (Instant launch).
    """
    # Fast GPU-to-GPU copy
    static_input.copy_(new_data)
    
    # Replay the graph (This is much faster than calling model(x))
    g.replay()
    
    return static_output

# ==========================================
# 3. BENCHMARKING
# ==========================================
print("\nBenchmarking...")
# Warmup
for _ in range(10): run_inference(dummy_input)

torch.cuda.synchronize()
start = time.time()
iterations = 1000

for _ in range(iterations):
    # In a real app, you would pass your real image tensor here
    output = run_inference(dummy_input)

torch.cuda.synchronize()
avg_time_ms = (time.time() - start) / iterations * 1000

print(f"Average Latency: {avg_time_ms:.3f} ms")
print(f"Throughput:      {1000/avg_time_ms:.1f} FPS")