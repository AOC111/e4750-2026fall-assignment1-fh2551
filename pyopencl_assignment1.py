"""
EECS E4750 Fall 2026 - Assignment 1 - PyOpenCL template

##############################################################################
#                        READ THIS BEFORE YOU START                          #
##############################################################################
# 1. ONLY write code inside the blocks marked:                               #
#        # ============ STUDENT CODE STARTS HERE ============                #
#        # ============= STUDENT CODE ENDS HERE =============                #
#    DO NOT touch anything else: the kernel code (provided only because      #
#    this is the first assignment!), the function signatures/return          #
#    values, relu_cpu(), and the entire TEST HARNESS section are used        #
#    for grading and must remain exactly as provided.                        #
# 2. Make all times returned in SECONDS and NOT milliseconds.                #
#    (Careful: OpenCL event profiling info is in NANOSECONDS.)               #
# 3. Run with:  python pyopencl_assignment1.py                               #
#    Every correctness test must print PASS. The benchmark then saves        #
#    'relu_opencl_timing.png' which you must include in your report.         #
##############################################################################

You must complete:
    - clModule.relu_array()
    - clModule.relu_buffer()
"""

import time
import numpy as np
import pyopencl as cl
import pyopencl.array

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


# ===================== OPENCL KERNEL ====================
# Do NOT modify - the kernel will not be provided for future assignments!
KERNEL_CODE = """
// ============ ReLU ============
// Computes out[i] = max(in[i], 0.0f) for every 0 <= i < n.
// One work-item handles one element.
__kernel void relu(
    __global float* out, __global const float* in, const unsigned int n
) {
    unsigned int i = get_global_id(0);
    if (i < n) {
        out[i] = fmax(in[i], 0.0f);
    }
}
"""


class clModule:
    def __init__(self):
        """
        **Do not modify this code**
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code.
        """
        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Create Context:
        self.ctx = cl.Context(devs)

        # Setup Command Queue (profiling enabled so you can use event timing):
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Build kernel code
        self.prg = cl.Program(self.ctx, KERNEL_CODE).build()

    def relu_array(self, x):
        """
        Compute ReLU of vector x on the GPU using the pyopencl.array.Array class.

        Time the operation twice:
          - kernel_time: kernel execution ONLY, using OpenCL event profiling
            (evt.profile.end - evt.profile.start, which is in NANOSECONDS).
          - total_time : end-to-end time INCLUDING device memory allocation,
            host-to-device transfer, kernel execution and device-to-host
            transfer, using time.time().

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
          1. Start the total timer (time.time()).
          2. Move the input to the device with cl.array.to_device() and
             allocate the output with cl.array.empty_like().
          3. Launch the kernel (self.prg.relu(self.queue, ...)) and keep
             the event it returns.
          4. Wait for the event, then compute kernel_time from
             evt.profile.end - evt.profile.start (nanoseconds -> seconds!).
          5. Get the result back to the host with the .get() function.
          6. Stop the total timer.
        """

        # ============= STUDENT CODE ENDS HERE =============

        return y, kernel_time, total_time

    def relu_buffer(self, x):
        """
        Compute ReLU of vector x on the GPU using pyopencl.Buffer objects
        and cl.enqueue_copy for the transfers.

        Time the operation twice:
          - kernel_time: kernel execution ONLY, using OpenCL event profiling.
          - total_time : end-to-end time INCLUDING buffer creation, both
            transfers and kernel execution, using time.time().

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
          1. Prepare an empty numpy array y on the host for the result.
          2. Start the total timer (time.time()).
          3. Create two buffers: input (READ_ONLY | COPY_HOST_PTR is
             convenient) and output (WRITE_ONLY).
          4. Launch the kernel and keep the event it returns.
          5. Wait for the event, then compute kernel_time from the event
             profiling info (nanoseconds -> seconds!).
          6. Use cl.enqueue_copy to get the result back to the host, and
             make sure the copy is finished before you stop the timer.
          7. Stop the total timer.
        """

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
    module = clModule()

    # ------------------------------------------------------------------
    # Part 1: Correctness tests
    # ------------------------------------------------------------------
    print("=" * 78)
    print("Part 1: Correctness tests (PyOpenCL)")
    print("=" * 78)

    all_passed = True
    for name, x in make_test_cases():
        expected = np.maximum(x, np.float32(0.0))

        y_arr, _, _ = module.relu_array(x)
        y_buf, _, _ = module.relu_buffer(x)

        ok_arr = (y_arr.shape == expected.shape) and np.array_equal(y_arr, expected)
        ok_buf = (y_buf.shape == expected.shape) and np.array_equal(y_buf, expected)
        ok = ok_arr and ok_buf
        all_passed = all_passed and ok

        print(f"[{'PASS' if ok else 'FAIL'}] {name:<24} N={len(x):>12,}   "
              f"relu_array: {'PASS' if ok_arr else 'FAIL'}   "
              f"relu_buffer: {'PASS' if ok_buf else 'FAIL'}")

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
    print("Part 2: Benchmark sweep (PyOpenCL)")
    print("=" * 78)

    REPS = 10
    sizes = [10 ** k for k in range(1, 9)]  # 10, 100, ..., 100,000,000

    avg_array_kernel, avg_array_total = [], []
    avg_buffer_kernel, avg_buffer_total = [], []
    avg_cpu = []

    rng = np.random.default_rng(2026)
    for N in sizes:
        x = rng.standard_normal(N).astype(np.float32)

        t_ak, t_at, t_bk, t_bt, t_c = [], [], [], [], []
        for _ in range(REPS):
            _, kt, tt = module.relu_array(x)
            t_ak.append(kt); t_at.append(tt)
            _, kt, tt = module.relu_buffer(x)
            t_bk.append(kt); t_bt.append(tt)
            _, ct = module.relu_cpu(x)
            t_c.append(ct)

        avg_array_kernel.append(np.average(t_ak))
        avg_array_total.append(np.average(t_at))
        avg_buffer_kernel.append(np.average(t_bk))
        avg_buffer_total.append(np.average(t_bt))
        avg_cpu.append(np.average(t_c))

        print(f"N={N:>12,} | array kernel {avg_array_kernel[-1]:.3e}s "
              f"| array total {avg_array_total[-1]:.3e}s "
              f"| buffer kernel {avg_buffer_kernel[-1]:.3e}s "
              f"| buffer total {avg_buffer_total[-1]:.3e}s "
              f"| cpu {avg_cpu[-1]:.3e}s")

    plt.figure()
    plt.title('ReLU average execution time (PyOpenCL)')
    plt.loglog(sizes, avg_array_kernel, 'o-', label='Array (kernel only)')
    plt.loglog(sizes, avg_array_total, 'o--', label='Array (total)')
    plt.loglog(sizes, avg_buffer_kernel, 's-', label='Buffer (kernel only)')
    plt.loglog(sizes, avg_buffer_total, 's--', label='Buffer (total)')
    plt.loglog(sizes, avg_cpu, '^-', label='CPU (numpy)')
    plt.xlabel('Vector length N')
    plt.ylabel('Average runtime over %d runs (seconds)' % REPS)
    plt.legend(loc='upper left')
    plt.grid(True, which='both', alpha=0.3)
    plt.savefig('relu_opencl_timing.png', dpi=300)
    print("\nTiming plot saved as 'relu_opencl_timing.png' - include it in your report.")
