# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2026)

## Assignment-1: Introduction to Memory Access in PyCUDA & PyOpenCL — the ReLU Activation Function

Due date: See in the courseworks.

Total points: 100

### <span style="color:red"><strong>TODO:</strong></span> (Re)naming of the student repository (for submitting the assignments)

***INSTRUCTIONS***
* You have to write the Code with thorough and clear comments, and write a report in PDF format which includes your plots and answers to the theory questions.
* Students need to change and use the following name for the repository with their solutions: `e4750-2026fall-assignment1-UNI`
  * Good Example: `e4750-2026fall-assignment1-zz9999`
  * Bad example: `e4750-2026fall-assignments-e4750-2026fall-assign0-zz9999`
* This change can be done from the "Settings" tab which is located on the repository page.


### Assignment 1 Primer

([GCP Setup, General tutorial on PyCuda and PyOpenCL](https://github.com/eecse4750/e4750-2026-2025fall-Student-Repo)) are shared in the Wiki pages.

The goal of this assignment is your first practical encounter with GPU kernels and host-to-device memory management, and to discover the most efficient method(s) of moving data between the host and the device.

The operation you will implement is the **Rectified Linear Unit (ReLU)**, the most widely used activation function in deep neural networks. ReLU sets all negative values of a vector to zero and leaves the non-negative values unchanged:

```
ReLU(x) = max(x, 0)
```

**Example 1:**
```
Input:  input  = [-2.0, -1.0, 0.0, 1.0, 2.0]
Output: output = [ 0.0,  0.0, 0.0, 1.0, 2.0]
```

**Example 2:**
```
Input:  input  = [-3.5, 0.0, 4.2]
Output: output = [ 0.0, 0.0, 4.2]
```

You will implement ReLU on a 1D vector of 32-bit floating point numbers (`float32`) in both PyOpenCL and PyCUDA. For each API you will implement the **same computation twice**, differing only in how they interact with device memory, and you will time each version both **kernel-only** and **end-to-end (including memory allocation and transfers)**. Comparing these timings against each other and against a serial CPU baseline is the core learning objective of this assignment.

The assignment should be completed in Google Cloud Hosted VMs.

### Relevant Documentation

By the time you got the HW, you should have already gotten a GCP coupon, if not, send email to `E4750TAs@columbia.edu`, title **Fall 2026 E4750 Student Request for GCP Coupon**.

([GCP Setup, General tutorial on PyCuda and PyOpenCL](https://github.com/eecse4750/e4750-2026-2025fall-Student-Repo)) are shared in the Wiki pages.

Please take a look at the *Wiki* Pages at the Github repo webpage and setup your GCP VM, you are **not** to set up the GUI desktop environment for the linux VM for this assignment.

Please share any questions you come across on Ed Discussion for everyone to see.

For PyOpenCL:
1. [OpenCL Runtime: Platforms, Devices & Contexts](https://documen.tician.de/pyopencl/runtime_platform.html)
2. [pyopencl.array](https://documen.tician.de/pyopencl/array.html#the-array-class)
3. [pyopencl.Buffer](https://documen.tician.de/pyopencl/runtime_memory.html#buffer)

For PyCUDA:
1. [Documentation Root](https://documen.tician.de/pycuda/index.html)
2. [Memory tools](https://documen.tician.de/pycuda/util.html#memory-pools)
3. [gpuarrays](https://documen.tician.de/pycuda/array.html)

### **Deliverables and Submission Instructions**
1. Complete the code templates provided in this folder for both PyOpenCL and PyCUDA. Keep the file names `pyopencl_assignment1.py` and `pycuda_assignment1.py` respectively.
    - Ensure your code runs without errors, **passes all the built-in correctness tests** (the test harness prints `PASS`/`FAIL` for every test case — all of them must print `PASS`), and produces the required timing plots.
2. Create a report in PDF format which includes:
    - A screenshot of the correctness-test output (all `PASS`) for both files.
    - Your timing plots for both the PyOpenCL and PyCUDA tasks.
    - A thorough analysis/comparison of the results shown on the plots.
    - Your answers to the theory questions.
    - Report File Name: `Assignment1_Report_{First Name}_{Last Name}_{uni}.pdf`


## Programming Problem (80 points)

### Problem set up

Consider a 1D vector *x* of length **N** containing 32-bit floats. The task is to write code for PyOpenCL and for PyCUDA which computes `y = ReLU(x)` on the GPU in two ways each, differentiated by how they interact with device memory. The programming problem is divided into two tasks, one each for OpenCL and CUDA. Make sure you complete both by following the instructions exactly.

**Implementation requirements (both tasks):**
* You **must** adhere to the templates given in the starter code files — this is essential for all assignments to be graded fairly and equally. You are free to add helper functions, but the class structure and the function signatures/return values must remain as provided.
* The kernel must compute the result on the device — computing ReLU on the host (e.g., with `np.maximum`) inside a GPU function will receive no credit, even if the tests pass.
* Your kernel must produce correct results for **any** vector length N ≥ 1.
* All returned times must be in **seconds**, NOT milliseconds.

**Testing:** the `main` section of each template is **fully provided** and must not be modified. It runs two parts:
1. **Correctness tests** — a suite of named test cases; read `make_test_cases()` in the templates to see exactly what is tested. Each case prints `PASS` or `FAIL` for each of your two implementations, so you always know whether your code is correct.
2. **Benchmark sweep** — vector sizes from 10 to 100,000,000 in factors of 10, each repeated 10 times and averaged, timing both of your implementations (kernel-only and end-to-end) against the provided serial CPU baseline, and saving a log-log timing plot for your report.

#### Task-1: PyOpenCL (30 points)

For PyOpenCL, you have been provided with the kernel code along with the assignment template *(the kernel is only provided for OpenCL, and only because this is the first assignment of the course!)*. Your task is to build this kernel and use it as the basis for two methods of computing ReLU:

1. *(15 points)* **`relu_array()`** — implement ReLU using the `pyopencl.array.Array` class to move data to/from device memory.
   - Return the result vector, the **kernel-only** execution time measured with **OpenCL event profiling**, and the **total** end-to-end time (including device memory allocation and both transfers) measured with `time.time()`.

2. *(15 points)* **`relu_buffer()`** — implement the same operation using `pyopencl.Buffer` objects and `cl.enqueue_copy`.
   - Return the result vector, the **kernel-only** time from **OpenCL event profiling**, and the **total** end-to-end time measured with `time.time()`.

#### Task-2: PyCUDA (50 points)

For PyCUDA the kernel is **not** provided — writing it is part of the task. Note that you cannot just paste the OpenCL kernel; you need CUDA C syntax.

1. *(15 points)* Write the CUDA kernel code for ReLU in the `CUDA_CODE` string at the top of the template. The kernel name and its parameters are provided — fill in the body only.

2. *(20 points)* **`relu_explicit()`** — implement ReLU with **explicit** memory management: allocate device memory with `pycuda.driver.mem_alloc()`, copy the input with `memcpy_htod()`, launch the kernel compiled with `SourceModule`, and retrieve the result with `memcpy_dtoh()`.
   - Return the result vector, the **kernel-only** execution time, and the **total** end-to-end time (including `mem_alloc` and both copies). Use **CUDA events** (`cuda.Event()`) for both measurements, and do not forget to synchronize.

3. *(15 points)* **`relu_gpuarray()`** — implement the same operation using the `gpuarray` class (`gpuarray.to_gpu()` / `.get()`) instead of `mem_alloc`, launching the **same kernel** compiled with `SourceModule`.
   - Return the result vector, the **kernel-only** time, and the **total** end-to-end time, both measured with CUDA events.

### Hints

Few things which can help you in this homework.

1. Synchronization:
    1. There are two ways to synchronize threads across blocks in PyCUDA:
        1. Using `pycuda.driver.Context.synchronize()`
        2. Using CUDA Events. Usually using CUDA Events is a better way to synchronize, for details you can go through: [https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/], it's a short interesting read.
            1. You can get an instance of a cuda event using `event = cuda.Event()`.
            2. You can record particular time instances using `event.record()`.
            3. You can synchronize using `event.synchronize()`.
            4. You can use `start_event.time_till(end_event)` to get the elapsed time — note that it is returned in **milliseconds**.
    2. To synchronize PyOpenCL kernels you can use `event.wait()`. PyOpenCL kernel launches return an event object.
2. OpenCL event profiling: with a `PROFILING_ENABLE` command queue, a kernel launch event `evt` exposes `evt.profile.start` and `evt.profile.end` in **nanoseconds**.
3. Consider a case in which a kernel call is followed by an `enqueue_copy` call / `memcpy_dtoh` call from device to host — the calls are enqueued in the proper sequence, and a device-to-host copy is blocking on the host unless you explicitly use an async copy.
4. In case you get an error like "device not ready", most likely the error is with synchronization.
5. Choose a fixed 1D block size that is a multiple of 32 (e.g., 256), and compute the grid size from N with ceiling division. Do not forget boundary check in the kernel.

## Theory Problems (20 points)

Questions 2, 3 and 4 must be answered using the numbers from your own timing plots; answers to those that do not reference your own measurements will not receive credit. Questions 1 and 5 are reasoning questions and need no measurements.

1. *(4 points)* **Synchronization.** A GPU kernel launch returns control to the host before the kernel has finished executing.
    - (a) Explain what "asynchronous" means in this context, and what the host gains from it.
    - (b) In `relu_explicit`, the device-to-host copy is enqueued after the kernel on the same stream, and that copy blocks the host. Given that, is an explicit synchronization call required for the **result** in `y` to be correct?
    - (c) Is an explicit synchronization call required for the **timing** you report to be correct?

    Explain why your answers to (b) and (c) differ.

2. *(4 points)* **Kernel-only versus end-to-end time.** From your plots, report the ratio of end-to-end time to kernel-only time at `N = 10` and at `N = 100,000,000`, for one implementation in each API. State what the difference between the two curves physically consists of, and explain why the ratio changes with `N`.

3. *(4 points)* **Array/gpuarray versus Buffer/mem_alloc.** Compare `relu_array` against `relu_buffer`, and `relu_gpuarray` against `relu_explicit`. State which is faster at large `N` and by how much. Explain what each abstraction does underneath, and account for why the kernel-only times of the two implementations within each API are nearly identical while their end-to-end times need not be.

4. *(4 points)* **GPU versus CPU crossover.** At what `N` does the GPU first beat the numpy baseline on end-to-end time? Does the crossover occur at a different `N` if you compare kernel-only time instead? Explain why the GPU loses at small `N`, and identify which fixed costs dominate there.

5. *(4 points)* **Scaling limits.** This ReLU uses one thread per element in a single kernel launch. Describe what breaks when `N` grows beyond what one such launch can index, and separately what breaks when the input no longer fits in device memory. For each case, describe how you would restructure the code to handle it.

## Code Templates

The starter code is provided as two separate files in this folder:

* [`pyopencl_assignment1.py`](./pyopencl_assignment1.py)
* [`pycuda_assignment1.py`](./pycuda_assignment1.py)

Write your code **only** inside the blocks marked `STUDENT CODE STARTS HERE` / `STUDENT CODE ENDS HERE` (and, for PyCUDA, the kernel body in `CUDA_CODE`) — each block contains a suggested workflow as a comment. Everything else, including the kernel signatures, function signatures and the test harness in `__main__`, must not be modified. Your code has to run from the main function with no parameters passed from the command line and be error/exception-free.
