/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * Modified for Tensor Core (WMMA) implementation using __half for A and B
 * with a Shared Memory tiling strategy for high performance.
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_fp16.h> // Include for __half type
#include <cuda.h>
#include <mma.h>      // Include for WMMA APIs

// Use the nvcuda namespace for WMMA functions
using namespace nvcuda;
using namespace wmma;

#define POLYBENCH_TIME 1

#include "gemm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

// Define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 1.5f
#define BETA 0.5f

#define RUN_ON_CPU

// --- WMMA Tiling Configuration (Shared Memory) ---
// Thread block size (64 threads = 2 warps, requested by user)
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 2 // Two warps in the block (32 * 2 = 64 threads)

// M, N, K dimensions of the C matrix block processed by ONE thread block
#define BLOCK_M 16 // The requested 16x16 tile size
#define BLOCK_N 16 // The requested 16x16 tile size
#define BLOCK_K 16 // K dimension block fetched from global memory in each iteration (matches WMMA_K)

// Dimensions for the WMMA operations (fixed by hardware)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Shared memory tile sizes (using +1 for banking padding on the inner dimension)
#define SHARED_A_WIDTH (BLOCK_K + 1) // 17
#define SHARED_B_WIDTH (BLOCK_N + 1) // 17
// ---------------------------------------------------

// WMMA fragment definitions
// Since BLOCK_M=WMMA_M, BLOCK_N=WMMA_N, BLOCK_K=WMMA_K, only one fragment is needed per type.
typedef fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> fragment_a_t;
typedef fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> fragment_b_t;
typedef fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragment_c_t;


// Function to convert float array to half array
void float_to_half_array(const DATA_TYPE *input, __half *output, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        output[i] = __float2half(input[i]);
    }
}


/* CPU implementation for verification (Unchanged) */
void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
      DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i,j,k;
    
    for (i = 0; i < _PB_NI; i++)
    {
            for (j = 0; j < _PB_NJ; j++)
            {
            C[i][j] *= beta;
    
            for (k = 0; k < _PB_NK; ++k)
            {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
            }
    }
}


void init(int ni, int nj, int nk, DATA_TYPE* alpha, DATA_TYPE* beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i, j;

    *alpha = ALPHA;
    *beta = BETA;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE) i*j) / NI;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE) i*j) / NI;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((DATA_TYPE) i*j) / NI;
        }
    }
}


void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    int i, j, fail;
    fail = 0;
    
    // Compare CPU and GPU outputs
    for (i=0; i < ni; i++) 
    {
        for (j=0; j < nj; j++) 
        {
            // Note: Since we used FP16 for multiplication, we must increase the tolerance.
            // Using 10x the standard threshold for FP16 precision loss.
            if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD * 10) 
            {
                fail++;
            }
        }
    }
    
    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD * 10, fail);
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    
    // Check for Tensor Core support (Compute Capability 7.0 or higher)
    if (deviceProp.major < 7) {
        printf("Error: GPU (Device %d) does not support Tensor Cores (Requires Compute Capability 7.0+).\n", GPU_DEVICE);
        printf("Exiting. Cannot run WMMA kernel.\n");
        exit(1);
    }
    
    printf("Setting device %d with name %s, Compute Capability %d.%d\n", 
        GPU_DEVICE, deviceProp.name, deviceProp.major, deviceProp.minor);
    cudaSetDevice( GPU_DEVICE );
}


/**
 * __global__ gemm_wmma_kernel: Performs GEMM using WMMA API and Shared Memory for high performance.
 * Each thread block (64 threads = 2 warps) computes a 16x16 tile of C.
 */
__global__ void gemm_wmma_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, const __half *a, const __half *b, DATA_TYPE *c)
{
    // Shared Memory allocation for A and B tiles (16x16 tiles with 1-unit padding)
    __shared__ __half sh_A[BLOCK_M * SHARED_A_WIDTH]; // 16 * 17
    __shared__ __half sh_B[BLOCK_K * SHARED_B_WIDTH]; // 16 * 17

    // Block C tile start coordinates
    const int start_m = blockIdx.y * BLOCK_M;
    const int start_n = blockIdx.x * BLOCK_N;

    // Thread local indices
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x; // thread_id will be 0-63 (2 warps)

    // Since BLOCK_M=WMMA_M and BLOCK_N=WMMA_N, M_WMMA_OPS and N_WMMA_OPS are 1.
    // Since BLOCK_K=WMMA_K, K_WMMA_OPS is 1 for the inner loop over the shared tile.
    
    // Initial accumulator fragment (C/D)
    fragment_c_t accum;
    
    // Initialize accumulator to 0.0f
    fill_fragment(accum, 0.0f);

    // Load C tile to local fragments for initial scaling (C = beta * C)
    fragment_c_t frag_c_in;
    
    // C tile starts at row/col
    const int row_start = start_m;
    const int col_start = start_n;

    // Load C fragment (or fill with zero if out of bounds)
    // Note: Since only 32 threads (one warp) are needed for WMMA, 
    // we only need one warp to load/store the C fragments. The second warp will be idle here.
    if (threadIdx.y == 0) // Only the first warp participates in C fragment handling
    {
        if (row_start < NI && col_start < NJ)
        {
            load_matrix_sync(frag_c_in, c + row_start * NJ + col_start, NJ, mem_row_major);
        }
        else
        {
            fill_fragment(frag_c_in, 0.0f);
        }
    }


    // Outer loop over the K dimension (tiles of size BLOCK_K=16)
    const int K_BLOCKS = (NK + BLOCK_K - 1) / BLOCK_K;
    for (int tile_k = 0; tile_k < K_BLOCKS; tile_k++)
    {
        const int start_k = tile_k * BLOCK_K;
        
        // 1. Cooperative load from Global Memory (A/B) to Shared Memory (sh_A/sh_B)
        
        // A is BLOCK_M x BLOCK_K (16 x 16). Total elements: 256
        const int A_elements = BLOCK_M * BLOCK_K;
        // B is BLOCK_K x BLOCK_N (16 x 16). Total elements: 256
        const int B_elements = BLOCK_K * BLOCK_N;
        
        const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y; // 64 threads
        
        // Load A tile (16x16) - 64 threads cooperate
        for (int idx = thread_id; idx < A_elements; idx += num_threads)
        {
            int block_row = idx / BLOCK_K;
            int block_col = idx % BLOCK_K;
            
            int global_row = start_m + block_row;
            int global_col = start_k + block_col;
            
            // Check bounds for A (NI x NK)
            if (global_row < NI && global_col < NK)
            {
                sh_A[block_row * SHARED_A_WIDTH + block_col] = a[global_row * NK + global_col];
            }
            else
            {
                sh_A[block_row * SHARED_A_WIDTH + block_col] = __float2half(0.0f);
            }
        }

        // Load B tile (16x16) - 64 threads cooperate
        for (int idx = thread_id; idx < B_elements; idx += num_threads)
        {
            int block_row = idx / BLOCK_N;
            int block_col = idx % BLOCK_N;
            
            int global_row = start_k + block_row;
            int global_col = start_n + block_col;
            
            // Check bounds for B (NK x NJ)
            if (global_row < NK && global_col < NJ)
            {
                sh_B[block_row * SHARED_B_WIDTH + block_col] = b[global_row * NJ + global_col];
            }
            else
            {
                sh_B[block_row * SHARED_B_WIDTH + block_col] = __float2half(0.0f);
            }
        }
        
        // Wait for all 64 threads to finish loading to shared memory
        __syncthreads();

        // 2. WMMA operation (Inner loop k_step is removed as BLOCK_K=WMMA_K=16)
        
        // Load A and B fragments from Shared Memory
        fragment_a_t frag_a; // Single fragment
        fragment_b_t frag_b; // Single fragment

        // Load 1 fragment for A (from sh_A starting at k_step * WMMA_K = 0)
        const __half* ptr_a = sh_A; 
        
        // Only the first warp needs to load the fragment for WMMA
        if (threadIdx.y == 0)
        {
            load_matrix_sync(frag_a, ptr_a, SHARED_A_WIDTH);
            
            // Load 1 fragment for B (from sh_B starting at k_step * WMMA_K = 0)
            const __half* ptr_b = sh_B;
            load_matrix_sync(frag_b, ptr_b, SHARED_B_WIDTH);
    
            // Perform the WMMA operation: D = D + A * B
            mma_sync(accum, frag_a, frag_b, accum);
        }
        
        // Wait for all 64 threads to finish WMMA operations before next global load
        __syncthreads();
    } // End of K-loop

    // 3. Final Store (Apply ALPHA/BETA and Write back to C)
    // Only the first warp processes and stores the C tile.
    if (threadIdx.y == 0)
    {
        if (row_start < NI && col_start < NJ)
        {
            // Apply C_new = beta * C_old + alpha * (A*B accumulation is in accum)
            for (int k = 0; k < accum.num_elements; k++) {
                // C_new = beta * C_old + alpha * (A*B)
                frag_c_in.x[k] = beta * frag_c_in.x[k] + alpha * accum.x[k];
            }
    
            // Store the result back to global memory C
            float* ptr_c = c + row_start * NJ + col_start;
            store_matrix_sync(ptr_c, frag_c_in, NJ, mem_row_major);
        }
    }
}


void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    // A and B will be stored as __half on the device
    __half *A_gpu_half;
    __half *B_gpu_half;
    // C will remain float (DATA_TYPE) on the device for higher precision accumulation
    DATA_TYPE *C_gpu;

    // 1. Allocate host memory for FP16 conversion
    __half *A_host_half = (__half*)malloc(sizeof(__half) * NI * NK);
    __half *B_host_half = (__half*)malloc(sizeof(__half) * NK * NJ);

    // 2. Convert FP32 host data to FP16 host data
    float_to_half_array(A[0], A_host_half, NI * NK);
    float_to_half_array(B[0], B_host_half, NK * NJ);

    // 3. Allocate device memory (A and B as __half, C as DATA_TYPE/float)
    cudaMalloc((void **)&A_gpu_half, sizeof(__half) * NI * NK);
    cudaMalloc((void **)&B_gpu_half, sizeof(__half) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ); // C remains float/DATA_TYPE

    // 4. Copy data to device
    cudaMemcpy(A_gpu_half, A_host_half, sizeof(__half) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu_half, B_host_half, sizeof(__half) * NK * NJ, cudaMemcpyHostToDevice);
    // C is initially FP32
    cudaMemcpy(C_gpu, C[0], sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    
    // 5. Define launch configuration for WMMA
    // Thread block size (32x2 = 64 threads)
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); 
    
    // Grid size, based on tiling BLOCK_M x BLOCK_N (16x16)
    dim3 grid((size_t)ceil(((float)NJ) / ((float)BLOCK_N)), (size_t)ceil(((float)NI) / ((float)BLOCK_M)));

    /* Start timer. */
    polybench_start_instruments;

    // Launch the WMMA kernel. Shared memory size is for 16x16 blocks.
    gemm_wmma_kernel<<< grid, block, sizeof(__half) * (BLOCK_M * SHARED_A_WIDTH + BLOCK_K * SHARED_B_WIDTH) >>>(
        ni, nj, nk, alpha, beta, A_gpu_half, B_gpu_half, C_gpu
    );
    cudaThreadSynchronize();

    /* Stop and print timer. */
    printf("GPU Time (WMMA Shared Memory) in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    // 6. Copy result back from device (C_outputFromGpu is FP32/DATA_TYPE)
    cudaMemcpy(C_outputFromGpu[0], C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    
    
    // 7. Cleanup
    cudaFree(A_gpu_half);
    cudaFree(B_gpu_half);
    cudaFree(C_gpu);
    free(A_host_half);
    free(B_host_half);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
          DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
    if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    // Note: A and B are initialized as FP32 on the host but converted to FP16 for the GPU.
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);

    // Host array C_cpu_copy for CPU verification
    POLYBENCH_2D_ARRAY_DECL(C_cpu_copy,DATA_TYPE,NI,NJ,ni,nj);

    init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
    
    // Copy C for CPU verification before GPU overwrites it
    memcpy(C_cpu_copy[0], C[0], sizeof(DATA_TYPE) * NI * NJ);

    GPU_argv_init();
    
    // Run WMMA kernel
    // We use C_cpu_copy as the initial C for the GPU since the original C is needed for CPU verification
    gemmCuda(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C_cpu_copy), POLYBENCH_ARRAY(C_outputFromGpu));


    #ifdef RUN_ON_CPU
        // Now run the CPU version on the original C array (which was initialized in init)
        printf("Running CPU verification...\n");
        /* Start timer. */
        polybench_start_instruments;

        gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
        
        /* Stop and print timer. */
        printf("CPU Time in seconds:\n");
        polybench_stop_instruments;
        polybench_print_instruments;
    
        // Compare CPU result (in C) with GPU result (in C_outputFromGpu)
        compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

    #else //print output to stderr so no dead code elimination

        print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));

    #endif //RUN_ON_CPU


    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);  
    POLYBENCH_FREE_ARRAY(C);  
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);  
    POLYBENCH_FREE_ARRAY(C_cpu_copy);

    return 0;
}

#include "../../common/polybench.c"
