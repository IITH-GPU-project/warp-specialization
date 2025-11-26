/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * Modified for Tensor Core (WMMA) implementation using __half for A and B.
 * Implemented with shared memory, warp specialization, and double buffering.
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
#include <omp.h> // Added for OpenMP

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
#define BETA 2.0f

#define RUN_ON_CPU

// --- WMMA Tiling Configuration ---
// WMMA intrinsic tile size
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16 // The K dimension is fixed by the WMMA API (16x16x16)

// --- Block & Warp Configuration ---

// Warps per block for computation (must be a square, e.g., 16 = 4x4)
#define CUSTOM_TILE_ROW 2
#define CUSTOM_TILE_COL 2  
#define NUM_COMPUTE_WARPS 1 // 16 warps

// Warps per block for DMA (data loading)
#define NUM_DMA_WARPS 2 // Adjustable

// Total warps in the block
#define BLOCK_WARPS (NUM_COMPUTE_WARPS + NUM_DMA_WARPS) // 16 + 4 = 20 warps

// Thread block dimensions
#define BLOCK_DIM_X 32        // Threads per warp
#define BLOCK_DIM_Y BLOCK_WARPS // Total warps

// Define the full block tile dimensions based on the compute warp layout
// A block computes a (4*16) x (4*16) = 64x64 tile of C.
#define BLOCK_TILE_M (CUSTOM_TILE_ROW * TILE_M) // 64
#define BLOCK_TILE_N (CUSTOM_TILE_COL * TILE_N) // 64
#define BLOCK_TILE_K TILE_K                      // 16

// --- Padding for Shared Memory Bank Conflicts ---
// Pad the N-dim of sh_B to avoid bank conflicts.
// Stride = 64 * __half = 128 bytes = 32 words (conflict!)
// Add 8 __half elements (16 bytes) to make the stride 72 * __half = 144 bytes = 36 words (no conflict)
#define SHMEM_B_N_PADDING 8
#define SHMEM_B_N_DIM_PADDED (BLOCK_TILE_N + SHMEM_B_N_PADDING) // 64 + 8 = 72

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
    
    // Parallelize the outer loop with OpenMP
    #pragma omp parallel for private(j, k)
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


/*********************************************************************************
 * DMA FUNCTION for WARP SPECIALIZATION
 *********************************************************************************/

// This function is executed ONLY by DMA warps.
// It loads one tile of A and one tile of B from global to shared memory.
__device__ void load_global_to_shared(int k_base, int ni, int nj, int nk,
                                      int block_start_m, int block_start_n,
                                      const __half *a, const __half *b,
                                      __half* sh_A_buf, __half* sh_B_buf)
{
    // DMA warp/thread identification
    short dma_warp_id = (threadIdx.y - NUM_COMPUTE_WARPS);      // 0 to NUM_DMA_WARPS-1
    short dma_tid = threadIdx.x + dma_warp_id * 32;           // 0 to (NUM_DMA_WARPS * 32)-1
    int dma_stride = NUM_DMA_WARPS * 32;                    // Total DMA threads

    // Load sh_A (BLOCK_TILE_M * BLOCK_TILE_K = 64*16 = 1024 elements)
    for (int i = dma_tid; i < (BLOCK_TILE_M * BLOCK_TILE_K); i += dma_stride)
    {
        int r = i / BLOCK_TILE_K; // 0-63
        int c = i % BLOCK_TILE_K; // 0-15
        
        int gbl_r = block_start_m + r;
        int gbl_c = k_base + c;

        if (gbl_r < ni && gbl_c < nk)
            sh_A_buf[i] = a[gbl_r * nk + gbl_c]; // Use 1D indexing for sh_A_buf
        else
            sh_A_buf[i] = __float2half(0.0f);
    }

    // Load sh_B (BLOCK_TILE_K * BLOCK_TILE_N = 16*64 = 1024 elements)
    // Store into padded shared memory (16 * 72)
    for (int i = dma_tid; i < (BLOCK_TILE_K * BLOCK_TILE_N); i += dma_stride)
    {
        int r = i / BLOCK_TILE_N; // logical row 0-15
        int c = i % BLOCK_TILE_N; // logical col 0-63

        int gbl_r = k_base + r;
        int gbl_c = block_start_n + c;
        
        // Calculate 1D index in padded shared memory
        int padded_sh_idx = r * SHMEM_B_N_DIM_PADDED + c;
        
        if (gbl_r < nk && gbl_c < nj)
            sh_B_buf[padded_sh_idx] = b[gbl_r * nj + gbl_c];
        else
            sh_B_buf[padded_sh_idx] = __float2half(0.0f);
    }
}


/*********************************************************************************
 * WARP-SPECIALIZED WMMA KERNEL (Double Buffered)
 *********************************************************************************/
__global__ void gemm_wmma_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, const __half *a, const __half *b, DATA_TYPE *c)
{
    // --- Shared Memory Double Buffers ---
    // Use 1D raw buffers for DMA, cast to 2D for compute
    __shared__ __half sh_A_raw[2][BLOCK_TILE_M * BLOCK_TILE_K]; // 2 x (64*16)
    // Pad B's N-dimension (leading dimension)
    __shared__ __half sh_B_raw[2][BLOCK_TILE_K * SHMEM_B_N_DIM_PADDED]; // 2 x (16*72)

    // --- Warp Specialization ---
    short warp_id = threadIdx.y;
    bool is_compute_warp = (warp_id < NUM_COMPUTE_WARPS);
    bool is_dma_warp = (warp_id >= NUM_COMPUTE_WARPS);

    // --- Thread Block Identification ---
    int block_start_m = blockIdx.y * BLOCK_TILE_M;
    int block_start_n = blockIdx.x * BLOCK_TILE_N;

    // --- Compute Warp Identification ---
    int warp_row, warp_col;
    fragment_c_t accum[CUSTOM_TILE_ROW][CUSTOM_TILE_ROW];

    if (is_compute_warp)
    {
        for(int localRowTileIndex=0; localRowTileIndex<CUSTOM_TILE_ROW; localRowTileIndex++) {        
            for(int localColTileIndex=0; localColTileIndex<CUSTOM_TILE_ROW; localColTileIndex++) {
                
                // Initialize accumulator to 0.0f
                fill_fragment(accum[localRowTileIndex][localColTileIndex], 0.0f);
            }
        }
    }

    // --- Pipelined K-Loop ---
    int k_base = 0;
    const int num_k_tiles = nk / BLOCK_TILE_K;

    // Prologue: DMA loads first tile (k=0) into buffer 0
    if (is_dma_warp)
    {
        load_global_to_shared(k_base, ni, nj, nk, block_start_m, block_start_n,
                              a, b, sh_A_raw[0], sh_B_raw[0]);
    }
    __syncthreads();

    // Main Loop: DMA loads k+1 while Compute processes k
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++)
    {
        short buf_idx = k_tile % 2;      // Buffer to COMPUTE from
        short next_buf_idx = 1 - buf_idx; // Buffer to LOAD into
        
        // Update k_base for the *next* tile to be loaded
        k_base = (k_tile + 1) * BLOCK_TILE_K;

        // --- STAGE 1: Load (k+1) + Compute (k) ---
        
        // DMA Warps: Load data for the *next* iteration (k+1)
        if (is_dma_warp && (k_tile < num_k_tiles - 1))
        {
            load_global_to_shared(k_base, ni, nj, nk, block_start_m, block_start_n, a, b, sh_A_raw[next_buf_idx], sh_B_raw[next_buf_idx]);
            int endtime = clock();
            if(threadIdx.x % 32 == 0) printf("DMA_clock(%d, %d), warp-id-%d, k_tile_%d: %d\n", blockIdx.x, blockIdx.y, warp_id, k_tile, endtime);
        }

        // Compute Warps: Process data from the *current* iteration (k)
        if (is_compute_warp)
        {
            for(int localRowTileIndex=0; localRowTileIndex<CUSTOM_TILE_ROW; localRowTileIndex++) {        
                for(int localColTileIndex=0; localColTileIndex<CUSTOM_TILE_ROW; localColTileIndex++) {
                    // Cast 1D raw shared buffers to 2D pointers for easy indexing
                    __half (*sh_A_buf)[BLOCK_TILE_K] = (__half (*)[BLOCK_TILE_K]) &sh_A_raw[buf_idx][0];
                    // Cast B to use the PADDED dimension
                    __half (*sh_B_buf)[SHMEM_B_N_DIM_PADDED] = (__half (*)[SHMEM_B_N_DIM_PADDED]) &sh_B_raw[buf_idx][0];

                    fragment_a_t frag_a;
                    fragment_b_t frag_b;

                    // Load this warp's 16x16 tile from sh_A
                    load_matrix_sync(frag_a, &sh_A_buf[localRowTileIndex * 16][0], BLOCK_TILE_K); // ld = 16

                    // Load this warp's 16x16 tile from sh_B
                    // Pass the PADDED leading dimension
                    load_matrix_sync(frag_b, &sh_B_buf[0][localColTileIndex * 16], SHMEM_B_N_DIM_PADDED); // ld = 72
                    
                    // Perform the WMMA operation: D = D + A * B
                    mma_sync(accum[localRowTileIndex][localColTileIndex], frag_a, frag_b, accum[localRowTileIndex][localColTileIndex]);
                }
            }
            int endtime = clock();
            if(threadIdx.x % 32 == 0) printf("compute_clock(%d, %d), warp-id-%d, k_tile_%d: %d\n", blockIdx.x, blockIdx.y, warp_id, k_tile, endtime);
        }
        
        // --- STAGE 2: Synchronize ---
        // Wait for both DMA and Compute to finish before swapping buffers
        __syncthreads();
    }
    
    // --- 3. Final Store (Accumulate and Write back to C) ---
    if (is_compute_warp)
    {
        for(int localRowTileIndex=0; localRowTileIndex<CUSTOM_TILE_ROW; localRowTileIndex++) {        
            for(int localColTileIndex=0; localColTileIndex<CUSTOM_TILE_ROW; localColTileIndex++) {
                // Find this warp's C tile's top-left corner
                int c_row = block_start_m + localRowTileIndex * 16;
                int c_col = block_start_n + localColTileIndex * 16;
                
                if (c_row < ni && c_col < nj)
                {
                    // Load existing C block (FP32) to apply BETA and accumulate ALPHA * A*B
                    fragment_c_t frag_c;
                    float* ptr_c = c + c_row * nj + c_col;

                    load_matrix_sync(frag_c, ptr_c, nj, mem_row_major);
                    
                    // C_new = beta * C_old + alpha * (A*B accumulation)
                    for (int k = 0; k < frag_c.num_elements; k++) {
                        frag_c.x[k] = beta * frag_c.x[k] + alpha * accum[localRowTileIndex][localColTileIndex].x[k];
                    }

                    // Store the result back to global memory C
                    store_matrix_sync(ptr_c, frag_c, nj, mem_row_major);
                }
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
    // Thread block size (32 * 20 = 640 threads)
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); 
    
    // Grid size, based on tiling 64x64 C tiles (BLOCK_TILE_M x BLOCK_TILE_N)
    dim3 grid((size_t)(ceil( ((float)NJ) / ((float)BLOCK_TILE_N) )), 
              (size_t)(ceil( ((float)NI) / ((float)BLOCK_TILE_M) )));

    /* Start timer. */
    polybench_start_instruments;

    // Launch the WMMA kernel
    gemm_wmma_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu_half, B_gpu_half, C_gpu);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    printf("GPU Time in seconds:\n"); // Modified label
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
    POLYBENCH_FREE_ARRAY(C_outputFromGpu); // Corrected typo here
    POLYBENCH_FREE_ARRAY(C_cpu_copy);

    return 0;
}

#include "../../common/polybench.c"




