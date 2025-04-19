import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import contextlib # For conditional autocast
import sys # For checking Python version
import traceback # For better error tracking
from typing import List, Tuple, Optional, Dict, Union # Added Union
from functools import partial # For FNO

# --- Model Definitions from benchMgNO.py ---

class MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, out: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        u, f = out
        u = u + (self.S(f - self.A(u)))
        out = (u, f)
        return out

class MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.S = S

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.S(f)
        return (u, f)

class Restrict(nn.Module):
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi if Pi is not None else nn.Identity()
        self.R = R if R is not None else nn.Identity()
        self.A = A if A is not None else nn.Identity()

    def forward(self, out: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        u, f = out
        if isinstance(self.A, nn.Identity):
             f_res = self.R(f)
        else:
             f_res = self.R(f - self.A(u))
        u_res = self.Pi(u)
        out = (u_res, f_res)
        return out

class MgConv(nn.Module):
    # Add type hints
    resolutions: List[int]
    upsample_kernels: List[int]
    norm_layer_list: nn.ModuleList
    rt_layers: nn.ModuleList
    layers: nn.ModuleList
    post_smooth_layers: nn.ModuleList
    num_levels: int # Add num_levels

    def __init__(self, num_iterations, out_channel, in_channel, resolution=64,
                 padding_mode='zeros', normalization=True, debug=False):
        super(MgConv, self).__init__()
        self.debug = debug
        self.num_iterations = num_iterations
        self.num_levels = len(num_iterations) # Store num_levels

        self.resolutions = self.calculate_downsampling_levels(resolution,
                                                              kernel_sizes=[3] * (self.num_levels - 1),
                                                              padding=1, stride=2)
        self.upsample_kernels = self.calculate_adjusted_upsample_kernels_simple(self.resolutions)

        if self.debug:
            print(f"Calculated Resolutions: {self.resolutions}")
            print(f"Calculated Upsample Kernels: {self.upsample_kernels}")
            if len(self.resolutions) != self.num_levels:
                 print(f"Warning: Length of resolutions ({len(self.resolutions)}) != num_levels ({self.num_levels})")
            if len(self.upsample_kernels) != self.num_levels - 1:
                 print(f"Warning: Length of upsample_kernels ({len(self.upsample_kernels)}) != num_levels-1 ({self.num_levels-1})")

        self.norm_layer_list = nn.ModuleList([
            nn.GroupNorm(1, out_channel, eps=1e-5, affine=True)
            if normalization else nn.Identity()
            for _ in range(self.num_levels - 1)
        ])

        self.rt_layers = nn.ModuleList()
        if len(self.upsample_kernels) != self.num_levels - 1:
             raise ValueError(f"Mismatch between upsample_kernels length ({len(self.upsample_kernels)}) and expected ({self.num_levels - 1})")
        for j in range(self.num_levels - 1):
             self.rt_layers.append(
                 nn.ConvTranspose2d(out_channel, out_channel,
                                    kernel_size=self.upsample_kernels[j],
                                    stride=2, padding=0, bias=False)
             )

        self.layers = nn.ModuleList()
        self.post_smooth_layers = nn.ModuleList()

        for l in range(self.num_levels):
            pre_smooth_sequence = []
            num_pre_smooth = num_iterations[l][0]
            for i in range(num_pre_smooth):
                if l == 0 and i == 0:
                    S = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
                    pre_smooth_sequence.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
                    S = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
                    pre_smooth_sequence.append(MgIte(A, S))

            if l > 0:
                 A_res = nn.Conv2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
                 Pi_res = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                 R_res = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                 restriction_layer = Restrict(Pi_res, R_res, A_res)
                 self.layers.append(nn.Sequential(restriction_layer, *pre_smooth_sequence))
            else:
                 self.layers.append(nn.Sequential(*pre_smooth_sequence))

            post_smooth_sequence = []
            num_post_smooth = num_iterations[l][1]
            if num_post_smooth > 0:
                for _ in range(num_post_smooth):
                    A = nn.Conv2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
                    S = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
                    post_smooth_sequence.append(MgIte(A, S))
                self.post_smooth_layers.append(nn.Sequential(*post_smooth_sequence))
            else:
                self.post_smooth_layers.append(nn.Identity())

    def calculate_downsampling_levels(self, H, kernel_sizes, stride=2, padding=1) -> List[int]:
        image_sizes = [H]
        current_H = H
        for kernel_size in kernel_sizes:
            H_out = (current_H + 2 * padding - kernel_size) // stride + 1
            if H_out < 1:
                 print(f"Warning: Calculated resolution {H_out} < 1 at H_in={current_H}. Stopping downsampling.")
                 break
            image_sizes.append(H_out)
            current_H = H_out
        return image_sizes

    def calculate_adjusted_upsample_kernels_simple(self, downsampling_sizes, stride=2) -> List[int]:
        adjusted_kernel_sizes = []
        for i in range(len(downsampling_sizes) - 1):
            H_in = downsampling_sizes[i+1]
            H_out = downsampling_sizes[i]
            kernel_size = H_out - (H_in - 1) * stride
            if kernel_size < 1:
                 print(f"Warning: Calculated negative/zero upsample kernel size ({kernel_size}) for H_in={H_in}, H_out={H_out}. Clamping to 1.")
                 kernel_size = 1
            adjusted_kernel_sizes.append(kernel_size)
        return adjusted_kernel_sizes

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        out_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_levels
        current_out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        level_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = f
        for l in range(self.num_levels):
            current_out = self.layers[l](level_input)
            out_list[l] = current_out
            level_input = current_out

        if current_out is None:
             raise RuntimeError("Output from the coarsest level (downsampling path) is None.")
        upsample_input_u: torch.Tensor = current_out[0]

        for j in range(self.num_levels - 2, -1, -1):
            fine_level_out = out_list[j]
            if fine_level_out is None:
                 raise RuntimeError(f"Stored output for level {j} is None during upsampling.")
            fine_u, fine_f = fine_level_out
            prolongated_u = self.rt_layers[j](upsample_input_u)
            fine_h, fine_w = fine_u.shape[-2:]
            prolongated_u = F.interpolate(prolongated_u, size=(fine_h, fine_w), mode='bilinear', align_corners=False)
            corrected_u = self.norm_layer_list[j](fine_u + prolongated_u)
            post_smooth_input = (corrected_u, fine_f)
            smoothed_out = self.post_smooth_layers[j](post_smooth_input)
            upsample_input_u = smoothed_out[0]
        return upsample_input_u

# --- FNO SpectralConv2d Definition ---
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, init_scale=2): # Removed out_resolution
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.fourier_weight1 = nn.Parameter(self.scale*
            torch.randn(in_channels, out_channels, modes1, modes2,  dtype=torch.cfloat))
        self.fourier_weight2 = nn.Parameter(self.scale*
            torch.randn(in_channels, out_channels, modes1, modes2,  dtype=torch.cfloat))
        if init_scale:
            nn.init.uniform_(self.fourier_weight1, a=-self.scale*(1/init_scale), b=self.scale*(1/init_scale))
            nn.init.uniform_(self.fourier_weight2, a=-self.scale*(1/init_scale), b=self.scale*(1/init_scale))

    # Complex multiplication
    @staticmethod
    def compl_mul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x): # Removed out_resolution argument
        batch_size = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm='forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.fourier_weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.fourier_weight2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='forward')
        return x


# --- Benchmarking Function ---
# (Keep existing benchmark_inference function as it is generic)
def benchmark_inference(model_name, model, device, input_size, in_channels, batch_size=10, num_runs=100, dtype=torch.float32, use_amp=False):
    """Benchmarks the inference time of a given model (takes single tensor 'f' as input)."""
    try:
        model.eval() # Set model to evaluation mode
        model.to(device)

        # Create dummy input 'f'
        f = torch.randn((batch_size, in_channels, input_size, input_size), device=device, dtype=dtype)

        # Warm-up runs
        print(f"[{model_name}] Warming up...")
        amp_context = torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp and dtype==torch.float32 and device.type=='cuda' else contextlib.nullcontext()
        with torch.no_grad():
             with amp_context:
                for _ in range(10):
                    _ = model(f)

        # Timing
        print(f"[{model_name}] Running benchmark...")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            with amp_context:
                for _ in range(num_runs):
                    _ = model(f)
        end_event.record()

        # Ensure CUDA kernels finish before stopping timer
        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_runs
        print(f"[{model_name}] Benchmark complete.")
        return avg_time_ms

    except Exception as e:
        print(f"[{model_name}] Benchmark failed: {e}")
        traceback.print_exc()
        return None
    finally:
        # Clean up GPU memory
        if 'f' in locals(): del f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- Main Execution ---
if __name__ == "__main__":
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # MgConv specific
    num_iterations = [(1, 1), (1, 1), (1, 1), (1, 1),] # Example: 4 levels
    num_levels = len(num_iterations)
    # FNO specific
    modes = 240 # Number of Fourier modes for FNO
    # Shared config
    in_channels = 24
    out_channels = 24
    resolution = 512
    batch_size = 16
    num_runs = 50

    results = {} # Dictionary to store results

    # --- MgConv Benchmarks ---
    print("\n" + "="*20 + " MgConv Benchmarks " + "="*20)
    # --- 1. MgConv Baseline Benchmark ---
    print("\n--- Benchmarking MgConv Baseline Model (FP32) ---")
    baseline_model = None
    try:
        baseline_model = MgConv(num_iterations, out_channels, in_channels, resolution=resolution, normalization=True)
        baseline_time = benchmark_inference("MgConv Baseline FP32", baseline_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32)
        if baseline_time is not None:
            results["MgConv Baseline FP32"] = baseline_time
            print(f"MgConv Baseline FP32 Average Inference Time: {baseline_time:.4f} ms")
    except Exception as e:
        print(f"[MgConv Baseline FP32] Failed: {e}")
        traceback.print_exc()
    finally:
        if baseline_model is not None: del baseline_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 2. MgConv torch.compile Benchmark ---
    print("\n--- Benchmarking MgConv torch.compile Model (FP32) ---")
    compile_model = None
    compiled_model = None
    if sys.version_info.major == 3 and sys.version_info.minor >= 12:
        print("[MgConv torch.compile FP32] Skipped: torch.compile not supported on Python 3.12+")
    elif not hasattr(torch, 'compile'):
         print("[MgConv torch.compile FP32] Skipped: torch.compile not available (requires PyTorch 2.0+).")
    else:
        try:
            compile_model = MgConv(num_iterations, out_channels, in_channels, resolution=resolution, normalization=True)
            print("[MgConv torch.compile FP32] Compiling model (mode='max-autotune')...")
            compiled_model = torch.compile(compile_model, mode="max-autotune")
            print("[MgConv torch.compile FP32] Compilation complete.")
            compile_time = benchmark_inference("MgConv torch.compile FP32", compiled_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32)
            if compile_time is not None:
                results["MgConv torch.compile FP32"] = compile_time
                print(f"MgConv torch.compile FP32 Average Inference Time: {compile_time:.4f} ms")
        except Exception as e:
            print(f"[MgConv torch.compile FP32] Failed: {e}")
            traceback.print_exc()
        finally:
            if compile_model is not None: del compile_model
            if compiled_model is not None: del compiled_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 3. MgConv TorchScript Benchmark (using Trace) ---
    print("\n--- Benchmarking MgConv TorchScript Model (FP32) ---")
    model_to_trace = None
    traced_model = None
    try:
        model_to_trace = MgConv(num_iterations, out_channels, in_channels, resolution=resolution, normalization=True)
        model_to_trace.to(device).eval()
        f_ex = torch.randn((batch_size, in_channels, resolution, resolution), device=device, dtype=torch.float32)
        print("[MgConv TorchScript FP32] Tracing model...")
        traced_model = torch.jit.trace(model_to_trace, f_ex)
        print("[MgConv TorchScript FP32] Tracing complete.")
        script_time = benchmark_inference("MgConv TorchScript FP32", traced_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32)
        if script_time is not None:
            results["MgConv TorchScript FP32"] = script_time
            print(f"MgConv TorchScript FP32 Average Inference Time: {script_time:.4f} ms")
    except Exception as e:
        print(f"[MgConv TorchScript FP32] Failed: {e}")
        traceback.print_exc()
    finally:
        if model_to_trace is not None: del model_to_trace
        if traced_model is not None: del traced_model
        if 'f_ex' in locals(): del f_ex
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 4. MgConv Half Precision (FP16) Benchmark ---
    print("\n--- Benchmarking MgConv Baseline Model (FP16) ---")
    fp16_model = None
    if device.type == 'cuda':
        try:
            fp16_model = MgConv(num_iterations, out_channels, in_channels, resolution=resolution, normalization=True)
            fp16_model.half()
            fp16_time = benchmark_inference("MgConv Baseline FP16", fp16_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float16)
            if fp16_time is not None:
                results["MgConv Baseline FP16"] = fp16_time
                print(f"MgConv Baseline FP16 Average Inference Time: {fp16_time:.4f} ms")
        except Exception as e:
            print(f"[MgConv Baseline FP16] Failed: {e}")
            traceback.print_exc()
        finally:
            if fp16_model is not None: del fp16_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[MgConv Baseline FP16] Skipped (requires CUDA).")

    # --- 5. MgConv Half Precision with AMP ---
    print("\n--- Benchmarking MgConv Baseline Model (FP32 with AMP) ---")
    amp_model = None
    if device.type == 'cuda':
        try:
            amp_model = MgConv(num_iterations, out_channels, in_channels, resolution=resolution, normalization=True)
            amp_time = benchmark_inference("MgConv Baseline AMP", amp_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32, use_amp=True)
            if amp_time is not None:
                results["MgConv Baseline AMP"] = amp_time
                print(f"MgConv Baseline AMP Average Inference Time: {amp_time:.4f} ms")
        except Exception as e:
            print(f"[MgConv Baseline AMP] Failed: {e}")
            traceback.print_exc()
        finally:
            if amp_model is not None: del amp_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[MgConv Baseline AMP] Skipped (requires CUDA).")

    # --- 6. MgConv torch.compile + AMP Benchmark ---
    print("\n--- Benchmarking MgConv torch.compile Model (FP32 with AMP) ---")
    compile_amp_model = None
    compiled_amp_model = None
    if device.type == 'cuda':
        if sys.version_info.major == 3 and sys.version_info.minor >= 12:
            print("[MgConv torch.compile AMP] Skipped: torch.compile not supported on Python 3.12+")
        elif not hasattr(torch, 'compile'):
             print("[MgConv torch.compile AMP] Skipped: torch.compile not available (requires PyTorch 2.0+).")
        else:
            try:
                compile_amp_model = MgConv(num_iterations, out_channels, in_channels, resolution=resolution, normalization=True)
                print("[MgConv torch.compile AMP] Compiling model (mode='max-autotune')...") # Changed mode
                compiled_amp_model = torch.compile(compile_amp_model, mode="max-autotune") # Changed mode
                print("[MgConv torch.compile AMP] Compilation complete.")
                compile_amp_time = benchmark_inference("MgConv torch.compile AMP", compiled_amp_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32, use_amp=True)
                if compile_amp_time is not None:
                    results["MgConv torch.compile AMP"] = compile_amp_time
                    print(f"MgConv torch.compile AMP Average Inference Time: {compile_amp_time:.4f} ms")
            except Exception as e:
                print(f"[MgConv torch.compile AMP] Failed: {e}")
                traceback.print_exc()
            finally:
                if compile_amp_model is not None: del compile_amp_model
                if compiled_amp_model is not None: del compiled_amp_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[MgConv torch.compile AMP] Skipped (requires CUDA).")


    # --- FNO Benchmarks ---
    print("\n" + "="*20 + " FNO (SpectralConv2d) Benchmarks " + "="*20)
    # --- 7. FNO Baseline Benchmark ---
    print("\n--- Benchmarking FNO Baseline Model (FP32) ---")
    fno_baseline_model = None
    try:
        fno_baseline_model = SpectralConv2d(in_channels, out_channels, modes, modes)
        fno_baseline_time = benchmark_inference("FNO Baseline FP32", fno_baseline_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32)
        if fno_baseline_time is not None:
            results["FNO Baseline FP32"] = fno_baseline_time
            print(f"FNO Baseline FP32 Average Inference Time: {fno_baseline_time:.4f} ms")
    except Exception as e:
        print(f"[FNO Baseline FP32] Failed: {e}")
        traceback.print_exc()
    finally:
        if fno_baseline_model is not None: del fno_baseline_model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 8. FNO torch.compile Benchmark ---
    print("\n--- Benchmarking FNO torch.compile Model (FP32) ---")
    fno_compile_model = None
    fno_compiled_model = None
    if sys.version_info.major == 3 and sys.version_info.minor >= 12:
        print("[FNO torch.compile FP32] Skipped: torch.compile not supported on Python 3.12+")
    elif not hasattr(torch, 'compile'):
         print("[FNO torch.compile FP32] Skipped: torch.compile not available (requires PyTorch 2.0+).")
    else:
        try:
            fno_compile_model = SpectralConv2d(in_channels, out_channels, modes, modes)
            print("[FNO torch.compile FP32] Compiling model (mode='max-autotune')...")
            fno_compiled_model = torch.compile(fno_compile_model, mode="max-autotune")
            print("[FNO torch.compile FP32] Compilation complete.")
            fno_compile_time = benchmark_inference("FNO torch.compile FP32", fno_compiled_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32)
            if fno_compile_time is not None:
                results["FNO torch.compile FP32"] = fno_compile_time
                print(f"FNO torch.compile FP32 Average Inference Time: {fno_compile_time:.4f} ms")
        except Exception as e:
            print(f"[FNO torch.compile FP32] Failed: {e}")
            traceback.print_exc()
        finally:
            if fno_compile_model is not None: del fno_compile_model
            if fno_compiled_model is not None: del fno_compiled_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 9. FNO TorchScript Benchmark (using Trace) ---
    print("\n--- Benchmarking FNO TorchScript Model (FP32) ---")
    fno_model_to_trace = None
    fno_traced_model = None
    try:
        fno_model_to_trace = SpectralConv2d(in_channels, out_channels, modes, modes)
        fno_model_to_trace.to(device).eval()
        f_ex = torch.randn((batch_size, in_channels, resolution, resolution), device=device, dtype=torch.float32)
        print("[FNO TorchScript FP32] Tracing model...")
        fno_traced_model = torch.jit.trace(fno_model_to_trace, f_ex)
        print("[FNO TorchScript FP32] Tracing complete.")
        fno_script_time = benchmark_inference("FNO TorchScript FP32", fno_traced_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32)
        if fno_script_time is not None:
            results["FNO TorchScript FP32"] = fno_script_time
            print(f"FNO TorchScript FP32 Average Inference Time: {fno_script_time:.4f} ms")
    except Exception as e:
        print(f"[FNO TorchScript FP32] Failed: {e}")
        traceback.print_exc()
    finally:
        if fno_model_to_trace is not None: del fno_model_to_trace
        if fno_traced_model is not None: del fno_traced_model
        if 'f_ex' in locals(): del f_ex
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 10. FNO Half Precision (FP16) Benchmark ---
    print("\n--- Benchmarking FNO Baseline Model (FP16) ---")
    fno_fp16_model = None
    if device.type == 'cuda':
        try:
            fno_fp16_model = SpectralConv2d(in_channels, out_channels, modes, modes)
            fno_fp16_model.half()
            fno_fp16_time = benchmark_inference("FNO Baseline FP16", fno_fp16_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float16)
            if fno_fp16_time is not None:
                results["FNO Baseline FP16"] = fno_fp16_time
                print(f"FNO Baseline FP16 Average Inference Time: {fno_fp16_time:.4f} ms")
        except Exception as e:
            print(f"[FNO Baseline FP16] Failed: {e}")
            traceback.print_exc()
        finally:
            if fno_fp16_model is not None: del fno_fp16_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[FNO Baseline FP16] Skipped (requires CUDA).")

    # --- 11. FNO Half Precision with AMP ---
    print("\n--- Benchmarking FNO Baseline Model (FP32 with AMP) ---")
    fno_amp_model = None
    if device.type == 'cuda':
        try:
            fno_amp_model = SpectralConv2d(in_channels, out_channels, modes, modes)
            fno_amp_time = benchmark_inference("FNO Baseline AMP", fno_amp_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32, use_amp=True)
            if fno_amp_time is not None:
                results["FNO Baseline AMP"] = fno_amp_time
                print(f"FNO Baseline AMP Average Inference Time: {fno_amp_time:.4f} ms")
        except Exception as e:
            print(f"[FNO Baseline AMP] Failed: {e}")
            traceback.print_exc()
        finally:
            if fno_amp_model is not None: del fno_amp_model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[FNO Baseline AMP] Skipped (requires CUDA).")

    # --- 12. FNO torch.compile + AMP Benchmark ---
    print("\n--- Benchmarking FNO torch.compile Model (FP32 with AMP) ---")
    fno_compile_amp_model = None
    fno_compiled_amp_model = None
    if device.type == 'cuda':
        if sys.version_info.major == 3 and sys.version_info.minor >= 12:
            print("[FNO torch.compile AMP] Skipped: torch.compile not supported on Python 3.12+")
        elif not hasattr(torch, 'compile'):
             print("[FNO torch.compile AMP] Skipped: torch.compile not available (requires PyTorch 2.0+).")
        else:
            try:
                fno_compile_amp_model = SpectralConv2d(in_channels, out_channels, modes, modes)
                print("[FNO torch.compile AMP] Compiling model (mode='max-autotune')...") # Changed mode
                fno_compiled_amp_model = torch.compile(fno_compile_amp_model, mode="max-autotune") # Changed mode
                print("[FNO torch.compile AMP] Compilation complete.")
                fno_compile_amp_time = benchmark_inference("FNO torch.compile AMP", fno_compiled_amp_model, device, resolution, in_channels, batch_size, num_runs, dtype=torch.float32, use_amp=True)
                if fno_compile_amp_time is not None:
                    results["FNO torch.compile AMP"] = fno_compile_amp_time
                    print(f"FNO torch.compile AMP Average Inference Time: {fno_compile_amp_time:.4f} ms")
            except Exception as e:
                print(f"[FNO torch.compile AMP] Failed: {e}")
                traceback.print_exc()
            finally:
                if fno_compile_amp_model is not None: del fno_compile_amp_model
                if fno_compiled_amp_model is not None: del fno_compiled_amp_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("[FNO torch.compile AMP] Skipped (requires CUDA).")


    # --- Results Summary ---
    print("\n" + "="*20 + " Benchmark Summary " + "="*20)
    if results:
        # Sort results by time (fastest first)
        sorted_results = sorted(results.items(), key=lambda item: item[1])
        print("Average Inference Time per Batch:")
        # Separate MgConv and FNO for clarity
        print("\n--- MgConv ---")
        for name, timing in sorted_results:
            if name.startswith("MgConv"):
                print(f"- {name}: {timing:.4f} ms")
        print("\n--- FNO (SpectralConv2d) ---")
        for name, timing in sorted_results:
            if name.startswith("FNO"):
                print(f"- {name}: {timing:.4f} ms")
        print("\n--- Overall Fastest ---")
        print(f"- {sorted_results[0][0]}: {sorted_results[0][1]:.4f} ms")
    else:
        print("No successful benchmarks to summarize.")
