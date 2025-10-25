/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * Modified for Tensor Core (WMMA) implementation using __half for A and B.
 * UPDATED: Shared Memory padding has been RE-INTRODUCED for correctness
 * and bank conflict avoidance with larger thread blocks.
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
#include <mma.h> // Include for WMMA APIs

// Use the nvcuda namespace for WMMA functions
using namespace nvcuda;
using namespace wmma;

#define POLYBENCH_TIME 1

#include "gemm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

// Define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 1.5f
#define BETA 0.5f

#define RUN_ON_CPU

// --- WMMA Tiling Configuration ---
// Note: We will use a standard 16x16 thread block size (256 threads)
// to compute a 128x128 tile of the C matrix.
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

// M, N, K tile size for the thread block
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16 // The K dimension is fixed by the WMMA API (16 for 16x16x16)

// Dimensions for the Shared Memory block.
#define K_TILE_SIZE 128 // Load 128 elements of K dimension at once into shared memory

// PADDING: Add padding to the shared memory B matrix to avoid bank conflicts.
#define SHMEM_PADDING 32 // Standard padding size to misalign bank access
#define TILE_N_PADDED (TILE_N + SHMEM_PADDING)

// The WMMA configuration uses 16x16x16 for __half inputs and float accumulator.
typedef fragment<matrix_a, 16, 16, 16, __half, row_major> fragment_a_t;
typedef fragment<matrix_b, 16, 16, 16, __half, row_major> fragment_b_t;
typedef fragment<accumulator, 16, 16, 16, float> fragment_c_t;
// ---------------------------------

// Function to convert float array to half array
void float_to_half_array(const DATA_TYPE *input, __half *output, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        output[i] = __float2half(input[i]);
    }
}


/* CPU implementation for verification */
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
        printf("Proceeding with standard GEMM kernel as fallback might be necessary.\n");
        // In a real application, you would switch to the standard kernel here.
    }
    
    printf("Setting device %d with name %s, Compute Capability %d.%d\n", 
        GPU_DEVICE, deviceProp.name, deviceProp.major, deviceProp.minor);
    cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_wmma_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, const __half *a, const __half *b, DATA_TYPE *c)
{
    // The starting point in global memory for the C tile computed by this thread block
    int start_m = blockIdx.y * TILE_M;
    int start_n = blockIdx.x * TILE_N;

    // Dimensions of the WMMA fragments (fixed)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Number of WMMA operations per thread block to cover the output tile
    const int M_TILES = TILE_M / WMMA_M; // 128/16 = 8
    const int N_TILES = TILE_N / WMMA_N; // 128/16 = 8
    
    // Shared Memory Declaration
    // sh_a: TILE_M x K_TILE_SIZE
    __shared__ __half sh_a[TILE_M * K_TILE_SIZE];
    // sh_b: K_TILE_SIZE x TILE_N_PADDED (with padding)
    __shared__ __half sh_b[K_TILE_SIZE * TILE_N_PADDED];

    // Initial accumulator fragment (C/D)
    fragment_c_t accum[M_TILES][N_TILES];
    
    // Initialize accumulator to 0.0f
    for (int i = 0; i < M_TILES; i++)
    {
        for (int j = 0; j < N_TILES; j++)
        {
            fill_fragment(accum[i][j], 0.0f);
        }
    }

    // Loop over the K dimension, loading K_TILE_SIZE elements into shared memory each iteration
    for (int tile_k = 0; tile_k < NK; tile_k += K_TILE_SIZE)
    {
        // --- 1. Load A Tile from Global to Shared Memory (TILE_M x K_TILE_SIZE) ---
        // Each thread cooperatively loads the block.
        for (int i = threadIdx.y; i < TILE_M; i += blockDim.y) 
        {
            for (int j = threadIdx.x; j < K_TILE_SIZE; j += blockDim.x) 
            {
                int global_row = start_m + i;
                int global_col = tile_k + j;

                if (global_row < NI && global_col < NK)
                {
                    sh_a[i * K_TILE_SIZE + j] = a[global_row * NK + global_col];
                }
                else
                {
                    sh_a[i * K_TILE_SIZE + j] = __float2half(0.0f);
                }
            }
        }

        // --- 2. Load B Tile from Global to Shared Memory (K_TILE_SIZE x TILE_N) ---
        // Load with PADDING: B is stored with stride TILE_N_PADDED in shared memory
        for (int i = threadIdx.y; i < K_TILE_SIZE; i += blockDim.y) 
        {
            for (int j = threadIdx.x; j < TILE_N; j += blockDim.x) 
            {
                int global_row = tile_k + i;
                int global_col = start_n + j;

                if (global_row < NK && global_col < NJ)
                {
                    // Store into shared memory using padded stride (TILE_N_PADDED)
                    sh_b[i * TILE_N_PADDED + j] = b[global_row * NJ + global_col];
                }
                else
                {
                    sh_b[i * TILE_N_PADDED + j] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads(); // Wait for shared memory load to complete

        // --- 3. Compute using WMMA on Shared Memory Tiles ---
        for (int inner_k = 0; inner_k < K_TILE_SIZE; inner_k += WMMA_K)
        {
            fragment_a_t frag_a[M_TILES];
            fragment_b_t frag_b[N_TILES];
            
            // Load A fragments from shared memory (unpadded, K_TILE_SIZE stride)
            for (int i = 0; i < M_TILES; i++)
            {
                const __half* ptr_a = sh_a + (i * WMMA_M) * K_TILE_SIZE + inner_k;
                load_matrix_sync(frag_a[i], ptr_a, K_TILE_SIZE);
            }
            
            // Load B fragments from shared memory (padded, TILE_N_PADDED stride)
            for (int j = 0; j < N_TILES; j++)
            {
                const __half* ptr_b = sh_b + inner_k * TILE_N_PADDED + (j * WMMA_N);
                load_matrix_sync(frag_b[j], ptr_b, TILE_N_PADDED); // <-- Using PADDED stride
            }

            // Perform the WMMA operation: D = D + A * B
            for (int i = 0; i < M_TILES; i++)
            {
                for (int j = 0; j < N_TILES; j++)
                {
                    mma_sync(accum[i][j], frag_a[i], frag_b[j], accum[i][j]);
                }
            }
        }
        
        __syncthreads(); // Wait for all WMMA ops to complete before next global load
    }
    
    // Final Store (Apply ALPHA/BETA and Write back to C)
    for (int i = 0; i < M_TILES; i++)
    {
        for (int j = 0; j < N_TILES; j++)
        {
            int row = start_m + i * WMMA_M;
            int col = start_n + j * WMMA_N;
            
            if (row < NI && col < NJ)
            {
                // Load existing C block (FP32) to apply BETA and accumulate ALPHA * A*B
                fragment_c_t frag_c;
                float* ptr_c = c + row * NJ + col;
                
                // Load/Initialization
                load_matrix_sync(frag_c, ptr_c, NJ, mem_row_major);
                
                // C_new = beta * C_old + alpha * (A*B accumulation is already in accum)
                for (int k = 0; k < frag_c.num_elements; k++) {
                    frag_c.x[k] = beta * frag_c.x[k] + alpha * accum[i][j].x[k];
                }

                // Store the final FP32 result back to global memory C
                store_matrix_sync(ptr_c, frag_c, NJ, mem_row_major);
            }
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
    // Thread block size (16x16 = 256 threads)
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); 
    
    // Grid size, based on tiling TILE_M x TILE_N (128x128)
    dim3 grid((size_t)(ceil( ((float)NJ) / ((float)TILE_N) )), (size_t)(ceil( ((float)NI) / ((float)TILE_M) )));

    /* Start timer. */
    polybench_start_instruments;

    // Launch the WMMA kernel
    gemm_wmma_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu_half, B_gpu_half, C_gpu);
    cudaThreadSynchronize();

    /* Stop and print timer. */
    printf("GPU Time (WMMA) in seconds:\n");
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
         DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), const char* name) // Added name parameter
{
  int i, j;
  fprintf (stdout, "== Printing Array %s (%d x %d) ==\n", name, ni, nj);

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
    // Print to stdout for user visibility
    fprintf (stdout, DATA_PRINTF_MODIFIER, C[i][j]);
    // Print 10 elements per row for better readability
    if ((i * ni + j) % 10 == 9) fprintf (stdout, "\n"); 
    }
  fprintf (stdout, "\n");
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

        // Explicitly print the results for viewing
        // print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu), "GPU Output");
        // print_array(ni, nj, POLYBENCH_ARRAY(C), "CPU Verification Output");

    #else //print output to stderr so no dead code elimination

        // For non-CPU verification runs, just print the GPU result
        print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu), "GPU Output");

    #endif //RUN_ON_CPU


    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);  
    POLYBENCH_FREE_ARRAY(C);  
    POLYBENCH_FREE_ARRAY(C_outputFromGpu); 
    POLYBENCH_FREE_ARRAY(C_cpu_copy);

    return 0;
}

#include "../../common/polybench.c"

