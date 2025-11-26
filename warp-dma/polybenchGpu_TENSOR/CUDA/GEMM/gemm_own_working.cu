/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * Modified for Tensor Core (WMMA) implementation using __half for A and B.
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
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 1.0f
#define BETA 1.0f

#define RUN_ON_CPU

// --- WMMA Tiling Configuration ---
// Note: We will use a standard 16x16 thread block size (256 threads)
// to compute a 128x128 tile of the C matrix.
#define BLOCK_DIM_X 32 * 4
#define BLOCK_DIM_Y 4

/*** 
* M, N, K tile size for the thread block 
* (tile size we defined for input matrices. For matrix_a it is "TILE_M * TILE_K". For matrix_b it is "TILE_K * TILE_N")
****/
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16 // The K dimension is fixed by the WMMA API (16 for 16x16x16)

// This tile of over the WMMA 16*16 computation
// each cell in this tile 16*16 (coz tiling over 16*16 WMMA tile computation)
#define CUSTOM_TILE_ROW 4
#define CUSTOM_TILE_COL 4

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


/* Old standard kernel (retained for comparison, not used in gemmCuda now)
__global__ void gemm_standard_kernel(...) { ... }
*/

__global__ void gemm_wmma_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, const __half *a, const __half *b, DATA_TYPE *c)
{
    // The starting point for the C tile computed by this thread block
    int start_m = blockIdx.y * TILE_M;
    int start_n = blockIdx.x * TILE_N;

    // Dimensions of the fragments
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Number of WMMA operations per thread block to cover the tile
    const int K_TILES = NK / (WMMA_K * CUSTOM_TILE_COL);
    const int COL_TILE = 2;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y; 
    const int warp_row = tidy;
    const int warp_col = tidx / 32;

    // if(tidx % 32 == 0)
    // printf("\nblock: (%d, %d) -> warp: (%d, %d)", blockIdx.x, blockIdx.y, warp_row, warp_col);

    // Initial accumulator fragment (C/D)
    fragment_c_t accum;
    
    // Initialize accumulator to 0.0f
    fill_fragment(accum, 0.0f);

    // Outer loop over the K dimension (tiles of size WMMA_K=16)
    for (int tile_k = 0; tile_k < K_TILES; tile_k++)
    {
        // Load A and B fragments from global memory
        fragment_a_t frag_a[CUSTOM_TILE_ROW];
        fragment_b_t frag_b[CUSTOM_TILE_COL];

        // Load M * K fragments for A
        for(int i=0; i<CUSTOM_TILE_ROW; i++)
        {     
            int a_index = (blockIdx.x * (CUSTOM_TILE_ROW * WMMA_M) * NK)
                            + (tile_k * (CUSTOM_TILE_COL * WMMA_N))
                            + ((warp_row * WMMA_M * NK) + (i * WMMA_N));
            const __half* ptr_a = a + a_index;

            // if(tidx % 32 == 0)
            // printf("\nblock: (%d, %d, %d) -> warp: (%d, %d) -> a_index: (%d, %d)", blockIdx.x, blockIdx.y, tile_k, warp_row, warp_col, i, a_index);
                                    
            // Ensure we only load if the tile is within bounds (NI and NK)
            // if (start_m + i * WMMA_M < NI && tile_k * WMMA_K < NK) {
                    load_matrix_sync(frag_a[i], ptr_a, NK);
            // } else {
            //         fill_fragment(frag_a[i], __float2half(0.0f));
            // }
        }
        
        // Load K * N fragments for B
        for(int i=0; i<CUSTOM_TILE_COL; i++)
        {    
            int b_index = (tile_k * (CUSTOM_TILE_ROW * WMMA_M) * NJ)
                            + (blockIdx.y * (CUSTOM_TILE_COL * WMMA_N))
                            + ((i * WMMA_M * NJ) + (warp_col * WMMA_N));
            const __half* ptr_b = b + b_index;

            // if(tidx % 32 == 0)
            // printf("\nblock: (%d, %d, %d) -> warp: (%d, %d) -> b_index: (%d, %d)", blockIdx.x, blockIdx.y, tile_k, warp_row, warp_col, i, b_index);

            // Ensure we only load if the tile is within bounds (NK and NJ)
            // if (tile_k * WMMA_K < NK && start_n + j * WMMA_N < NJ) {
                load_matrix_sync(frag_b[i], ptr_b, NJ);
            // } else {
            //     fill_fragment(frag_b[i], __float2half(0.0f));
            // }
        }

        // Perform the WMMA operation: D = D + A * B
        for(int k=0; k<CUSTOM_TILE_ROW; k++){
            //if (start_m + i * WMMA_M < NI && start_n + j * WMMA_N < NJ) {
                // mma_sync(D, A, B, C); -> D is same as C in this case
                mma_sync(accum, frag_a[k], frag_b[k], accum);
            //}
        }
     
    }
    
    // Final Store (Accumulate and Write back to C)
    int row = (start_m * CUSTOM_TILE_ROW + warp_row) * WMMA_M;
    int col = (start_n * CUSTOM_TILE_COL + warp_col )* WMMA_N;
    
    // if (row < NI && col < NJ)
    {
        // Load existing C block (FP32) to apply BETA and accumulate ALPHA * A*B
        fragment_c_t frag_c;
        float* ptr_c = c + (blockIdx.x * (CUSTOM_TILE_ROW * WMMA_M) * NK)
                            + (blockIdx.y * (CUSTOM_TILE_COL * WMMA_N))
                            + ((warp_row * WMMA_M * NJ) + (warp_col * WMMA_N));

        load_matrix_sync(frag_c, ptr_c, NJ, mem_row_major);
        
        // C_new = beta * C_old + alpha * (A*B accumulation is already in accum)
        for (int k = 0; k < frag_c.num_elements; k++) {
            frag_c.x[k] = beta * frag_c.x[k] + alpha * accum.x[k];
        }

        // Store the result back to global memory C
        store_matrix_sync(ptr_c, frag_c, NJ, mem_row_major);
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
    dim3 grid((size_t)(ceil( ((float)NJ) / (((float)TILE_N) * CUSTOM_TILE_ROW) )), (size_t)(ceil( ((float)NI) / (((float)TILE_M) * CUSTOM_TILE_COL) )));

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
        // print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));
        // print_array(ni, nj, POLYBENCH_ARRAY(C));

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

