# warp-specialization

This repository contains optimized CUDA implementations, automated benchmarking harnesses, and Nsight Compute profiling reports for selected kernels from the Polybench suite. The project focuses on performance optimization techniques including Shared Memory tiling, Warp Specialization, and Tensor Core usage.

## Repository Structure

The repository is organized into specific directories for each kernel type. Each directory contains the source code for multiple distinct implementations (variants) of that kernel, along with build and run scripts.

### Source Code & Benchmarking Directories

**[Polybench_GEMM_FP32/](./Polybench_GEMM_FP32)** (Single-Precision GEMM)
    * Contains implementations for:
        * **Baseline**: Naive global memory implementation.
        * **Shared Memory**: Tiled implementation to reduce global memory traffic.
        * **Warp Specialization**: Producer/Consumer model to overlap compute and memory latency.

**[Polybench_GEMM_FP64/](./Polybench_GEMM_FP64)** (Double-Precision GEMM)
    * Contains implementations for:
        * **Baseline**
        * **Shared Memory**
        * **Warp Specialization**

**[Polybench_GEMM_TENSOR/](./Polybench_GEMM_TENSOR)** (Tensor Core GEMM)
    * Contains implementations for:
        * **Baseline**
        * **Shared Memory**: Utilizes nvcuda::wmma API.
        * **Warp Specialization**: Optimized pipeline using Tensor Cores.

**[Polybench_JACOBI2D_FP32/](./Polybench_JACOBI2D_FP32)** (2D Jacobi Stencil)
    * Contains implementations for:
        * **Baseline**: Naive "Ping-Pong" grids using global memory.
        * **Pointer Swapping**: Optimization to eliminate excessive cudaMemcpy overhead.
        * **Shared Memory**: Loads tiles and halo regions into shared memory.
        * **Warp Specialization**: Separates halo loading from interior computation.

**[Polybench_MVT_FP32/](./Polybench_MVT_FP32)** (Matrix-Vector Product Transpose)
    * Contains multiple kernel variants for MVT operations.

### Profiling & Analysis Directories

**[GEMM_FP32_Nsight_Compute_Reports/](./GEMM_FP32_Nsight_Compute_Reports)**
    * Contains .ncu-rep files and PDF exports generated via NVIDIA Nsight Compute for detailed GEMM analysis.
**[JACOBI2D_FP32_Nsight_Compute_Reports/](./JACOBI2D_FP32_Nsight_Compute_Reports)**
    * Contains Nsight Compute project files (.ncu-proj) and reports for the Jacobi2D variants.

---

## Automated Benchmarking (run.sh)

Each implementation directory includes a robust run.sh script designed to automate the experimental workflow for that specific kernel.

### How it Works
The script handles the entire compilation and execution pipeline automatically:
1.  **Configuration Injection**: It uses sed to dynamically modify problem size macros (e.g., #define NI, #define N) inside the .cu or .cuh files.
2.  **Compilation**: It triggers make to compile all implementations (Baseline, Shared Mem, etc.) for that specific size.
3.  **Execution**: It runs each executable multiple times.
4.  **Data Extraction**: It parses stdout to capture GPU time, CPU time, and verification mismatch counts.
5.  **Reporting**: It aggregates the data into CSV and TXT files.

### Configuration
You can customize the experiments by editing the variables at the top of any run.sh file:
bash

# Example from Polybench_GEMM_FP32/run.sh
NUM_RUNS=5               # Number of times to run each executable per size (averages results)
SIZES=(32 64 ... 8192)   # List of matrix dimensions to test
OUTPUT_CSV="results.csv" # Raw data output
OUTPUT_TXT="summary.txt" # Formatted table output

# Example from Polybench_JACOBI2D_FP32/run.sh
TIMESTEPS=100            # Number of time steps for stencil simulation
