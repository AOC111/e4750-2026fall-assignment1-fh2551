"""
EECS E4750 Fall 2026 - Assignment 1 - PyCUDA template

##############################################################################
#                        READ THIS BEFORE YOU START                          #
##############################################################################
# 1. ONLY write code inside the blocks marked:                               #
#        # ============ STUDENT CODE STARTS HERE ============                #
#        # ============= STUDENT CODE ENDS HERE =============                #
#    and inside the kernel bodies in CUDA_CODE.                              #
#    DO NOT touch anything else: the kernel signatures, the function         #
#    signatures/return values, relu_cpu(), and the entire TEST HARNESS       #
#    section are used for grading and must remain exactly as provided.       #
# 2. Make all times returned in SECONDS and NOT milliseconds.                #
#    (Careful: cuda events' time_till() returns MILLISECONDS.)               #
# 3. Run with:  python pycuda_assignment1.py                                 #
#    Every correctness test must print PASS. The benchmark then saves        #
#    'relu_cuda_timing.png' which you must include in your report.           #
##############################################################################

You must complete:
    - The 'relu' kernel body in CUDA_CODE
    - reluModule.relu_explicit()
    - reluModule.relu_gpuarray()
"""

import time
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


# ===================== CUDA KERNELS ====================
# Do NOT change the kernel name or its parameters - only fill in the body.
# (Note that you can't just paste OpenCL code here - you need CUDA C syntax.)
CUDA_CODE = """
// ============ ReLU ============
// Computes out[i] = max(in[i], 0.0f) for every 0 <= i < n.
// One thread handles one element.
__global__ void relu(
    float* out, const float* in, const unsigned int n
) {
    // Map the current thread to one element of the one-dimensional input.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // The final block can contain more threads than elements, so guard the
    // memory access.  ReLU keeps non-negative values and replaces negatives
    // with the float literal 0.0f.
    if (i < n) {
        out[i] = in[i] >= 0.0f ? in[i] : 0.0f;
    }
}
"""


class reluModule:
    def __init__(self):
        """
        **Do not modify this code**
        The kernel code is compiled once here so it does not need to be
        recompiled for every function call. A fixed 1D block size is
        provided; you compute the grid size from N inside each function.
        """
        self.mod = SourceModule(CUDA_CODE)
        self.block_size = 256

    def relu_explicit(self, x):
        """
        Compute ReLU of vector x on the GPU with EXPLICIT memory management:
        cuda.mem_alloc() for device memory, cuda.memcpy_htod() /
        cuda.memcpy_dtoh() for the transfers.

        Time the operation twice, both with CUDA events:
          - kernel_time: kernel execution ONLY (events recorded immediately
            around the kernel launch).
          - total_time : end-to-end time INCLUDING mem_alloc, host-to-device
            copy, kernel execution and device-to-host copy (events recorded
            around the whole sequence).

        Arguments:
            x           :   1D numpy array of np.float32
        Returns:
            y           :   1D numpy array of np.float32, y = ReLU(x)
            kernel_time :   kernel-only execution time, in seconds
            total_time  :   total end-to-end execution time, in seconds
        """
        n = np.uint32(len(x))

        # ============ STUDENT CODE STARTS HERE ============
        """
        Suggested workflow:
          1. Initialize an empty numpy array y on the host for the result.
          2. Create four event objects: total_start, total_end,
             kernel_start, kernel_end.
          3. Get the kernel function from self.mod with get_function().
          4. Compute the grid size from n and self.block_size
             (ceiling division).
          5. Record total_start.
          6. Allocate device memory for the input and output (mem_alloc).
          7. Copy the input from host to device (memcpy_htod).
          8. Record kernel_start, launch the kernel, record kernel_end.
          9. Copy the result from device to host (memcpy_dtoh).
         10. Record total_end and synchronize it.
         11. Compute kernel_time and total_time in SECONDS
             (time_till() gives milliseconds!).
        """
        # Initialize an empty numpy array y on the host for the result
        y = np.empty_like(x)

        # Create four event objects to count time
        [total_start, total_end, kernel_start, kernel_end] = [cuda.Event() for _ in range(4)]

        # Get the kernel function
        relu = self.mod.get_function("relu")

        # Ceiling division launches enough 256-thread blocks for every element.
        # The kernel's i < n guard safely disables surplus threads in the last block.
        grid_size = (int(n) + self.block_size - 1) // self.block_size

        # Start total timing before device allocation so allocation and both host/device transfers are included
        # in the end-to-end measurement.
        total_start.record()

        # Allocate raw device buffers and initialize the input buffer from x.
        x_gpu = cuda.mem_alloc(x.nbytes)
        y_gpu = cuda.mem_alloc(y.nbytes)
        cuda.memcpy_htod(x_gpu, x)

        # Bracket only the asynchronous kernel launch with the kernel event pair.
        kernel_start.record()
        relu(y_gpu, x_gpu, n, block=(self.block_size, 1, 1), grid=(grid_size, 1))
        kernel_end.record()

        # Transfer the result back to host device(CPU)
        cuda.memcpy_dtoh(y, y_gpu)

        # Record the end of the full sequence and synchronize before querying
        # either event interval; otherwise the GPU work may still be pending.
        total_end.record()
        total_end.synchronize()

        # CUDA event durations are milliseconds; multiply by 1e-3 for seconds.
        total_time = total_end.time_since(total_start) * 1e-3
        kernel_time = kernel_end.time_since(kernel_start) * 1e-3

        # ============= STUDENT CODE ENDS HERE =============

        return y, kernel_time, total_time

    def relu_gpuarray(self, x):
        """
        Compute ReLU of vector x on the GPU using the gpuarray class
        (gpuarray.to_gpu() / .get()) instead of mem_alloc, launching the
        SAME 'relu' kernel from CUDA_CODE.

        Time the operation twice, both with CUDA events:
          - kernel_time: kernel execution ONLY.
          - total_time : end-to-end time INCLUDING gpuarray allocation,
            transfer to the device, kernel execution and .get().

        Arguments:
            x           :   1D numpy array of np.float32
        Returns:
            y           :   1D numpy array of np.float32, y = ReLU(x)
            kernel_time :   kernel-only execution time, in seconds
            total_time  :   total end-to-end execution time, in seconds
        """
        n = np.uint32(len(x))

        # ============ STUDENT CODE STARTS HERE ============
        """
        Suggested workflow:
          1. Create four event objects: total_start, total_end,
             kernel_start, kernel_end.
          2. Get the kernel function from self.mod with get_function().
          3. Compute the grid size from n and self.block_size.
          4. Record total_start.
          5. Move the input to the device with gpuarray.to_gpu() and
             allocate the output with gpuarray.empty().
          6. Record kernel_start, launch the kernel, record kernel_end.
             (gpuarray objects can be passed to the kernel directly.)
          7. Fetch the result from device to host with .get().
          8. Record total_end and synchronize it.
          9. Compute kernel_time and total_time in SECONDS.
        """
        # Create four event objects to count time
        [total_start, total_end, kernel_start, kernel_end] = [cuda.Event() for _ in range(4)]

        # Reuse the same compiled kernel as the explicit-memory implementation.
        relu = self.mod.get_function("relu")

        # Round up so the grid covers n even when n cannot be divided by block_size.
        grid_size = (int(n) + self.block_size - 1) // self.block_size

        # Begin before gpuarray allocation and transfer so total_time includes the complete operation.
        total_start.record()

        # Move the input to the device with gpuarray.to_gpu() and allocate the output with gpuarray.empty().
        x_gpu = gpuarray.to_gpu(x)
        y_gpu = gpuarray.empty(x_gpu.shape, x_gpu.dtype)

        # gpuarray objects expose their device pointers to the CUDA kernel, so
        # they can be passed directly as the output and input arguments.
        kernel_start.record()
        relu(y_gpu, x_gpu, n, block=(self.block_size, 1, 1), grid=(grid_size, 1))
        kernel_end.record()

        # .get() waits
        # and returns the completed result as a NumPy array.
        y = y_gpu.get()

        # Synchronize the last total event before reading either elapsed time.
        total_end.record()
        total_end.synchronize()

        # Convert the CUDA event measurements from milliseconds to seconds.
        total_time = total_end.time_since(total_start) * 1e-3
        kernel_time = kernel_end.time_since(kernel_start) * 1e-3
        # ============= STUDENT CODE ENDS HERE =============

        return y, kernel_time, total_time

    def relu_cpu(self, x):
        """
        **Do not modify this code**
        Serial ReLU on the host (CPU) - the reference implementation.

        Arguments:
            x       :   1D numpy array of np.float32
        Returns:
            y       :   ReLU(x)
            time    :   execution time in seconds
        """
        start = time.time()
        y = np.maximum(x, np.float32(0.0))
        end = time.time()
        return y, end - start


###############################################################################
#                       TEST HARNESS - DO NOT MODIFY                          #
###############################################################################

def make_test_cases():
    """Build the correctness test suite: (name, input_vector) pairs."""
    rng = np.random.default_rng(4750)
    tests = [
        ("example_1",             np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)),
        ("example_2",             np.array([-3.5, 0.0, 4.2], dtype=np.float32)),
        ("single_element",        np.array([-7.5], dtype=np.float32)),
        ("all_negative",          -rng.random(4096, dtype=np.float32) - np.float32(0.5)),
        ("all_positive",          rng.random(4096, dtype=np.float32) + np.float32(0.5)),
        ("non_multiple_of_block", rng.standard_normal(1_000_003).astype(np.float32)),
        ("large_25M",             rng.standard_normal(25_000_000).astype(np.float32)),
    ]
    return tests


if __name__ == "__main__":
    module = reluModule()

    # ------------------------------------------------------------------
    # Part 1: Correctness tests
    # ------------------------------------------------------------------
    print("=" * 78)
    print("Part 1: Correctness tests (PyCUDA)")
    print("=" * 78)

    all_passed = True
    for name, x in make_test_cases():
        expected = np.maximum(x, np.float32(0.0))

        y_exp, _, _ = module.relu_explicit(x)
        y_gpu, _, _ = module.relu_gpuarray(x)

        ok_exp = (y_exp.shape == expected.shape) and np.array_equal(y_exp, expected)
        ok_gpu = (y_gpu.shape == expected.shape) and np.array_equal(y_gpu, expected)
        ok = ok_exp and ok_gpu
        all_passed = all_passed and ok

        print(f"[{'PASS' if ok else 'FAIL'}] {name:<24} N={len(x):>12,}   "
              f"relu_explicit: {'PASS' if ok_exp else 'FAIL'}   "
              f"relu_gpuarray: {'PASS' if ok_gpu else 'FAIL'}")

    print("-" * 78)
    if all_passed:
        print("All correctness tests PASSED.")
    else:
        print("Some correctness tests FAILED - fix your implementation before "
              "looking at the benchmark numbers.")
    print()

    # ------------------------------------------------------------------
    # Part 2: Benchmark sweep (only meaningful once Part 1 passes)
    # ------------------------------------------------------------------
    print("=" * 78)
    print("Part 2: Benchmark sweep (PyCUDA)")
    print("=" * 78)

    REPS = 10
    sizes = [10 ** k for k in range(1, 9)]  # 10, 100, ..., 100,000,000

    avg_explicit_kernel, avg_explicit_total = [], []
    avg_gpuarray_kernel, avg_gpuarray_total = [], []
    avg_cpu = []

    rng = np.random.default_rng(2026)
    for N in sizes:
        x = rng.standard_normal(N).astype(np.float32)

        t_ek, t_et, t_gk, t_gt, t_c = [], [], [], [], []
        for _ in range(REPS):
            _, kt, tt = module.relu_explicit(x)
            t_ek.append(kt); t_et.append(tt)
            _, kt, tt = module.relu_gpuarray(x)
            t_gk.append(kt); t_gt.append(tt)
            _, ct = module.relu_cpu(x)
            t_c.append(ct)

        avg_explicit_kernel.append(np.average(t_ek))
        avg_explicit_total.append(np.average(t_et))
        avg_gpuarray_kernel.append(np.average(t_gk))
        avg_gpuarray_total.append(np.average(t_gt))
        avg_cpu.append(np.average(t_c))

        print(f"N={N:>12,} | explicit kernel {avg_explicit_kernel[-1]:.3e}s "
              f"| explicit total {avg_explicit_total[-1]:.3e}s "
              f"| gpuarray kernel {avg_gpuarray_kernel[-1]:.3e}s "
              f"| gpuarray total {avg_gpuarray_total[-1]:.3e}s "
              f"| cpu {avg_cpu[-1]:.3e}s")

    plt.figure()
    plt.title('ReLU average execution time (PyCUDA)')
    plt.loglog(sizes, avg_explicit_kernel, 'o-', label='Explicit (kernel only)')
    plt.loglog(sizes, avg_explicit_total, 'o--', label='Explicit (total)')
    plt.loglog(sizes, avg_gpuarray_kernel, 's-', label='gpuarray (kernel only)')
    plt.loglog(sizes, avg_gpuarray_total, 's--', label='gpuarray (total)')
    plt.loglog(sizes, avg_cpu, '^-', label='CPU (numpy)')
    plt.xlabel('Vector length N')
    plt.ylabel('Average runtime over %d runs (seconds)' % REPS)
    plt.legend(loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('relu_cuda_timing.png', dpi=300)
    print("\nTiming plot saved as 'relu_cuda_timing.png' - include it in your report.")
