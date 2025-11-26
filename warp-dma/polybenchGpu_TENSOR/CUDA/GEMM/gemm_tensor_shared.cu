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
#include <omp.h> //added for OpenMP
#include "gemm.cuh"

//
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"
#include "../../common/polybench.c"

//
using namespace nvcuda;
using namespace wmma;

#define POLYBENCH_TIME 1

#define GPU_DEVICE 0

// Define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 1.5f
#define BETA 2.0f

#define RUN_ON_CPU

// Thread block dimensions
#define BLOCK_DIM_X 128
#define BLOCK_DIM_Y 4

// WMMA tile size (fixed computation size by tensor cores)
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16 // The K dimension is fixed by the WMMA API (16x16x16)


// This is tiling over (16*16) output matrix, since tensor cores executes (16*16) elements at a time.
// each cell in this tile 16*16 (coz tiling over 16*16 WMMA tile computation)
#define CUSTOM_TILE_ROW 4
#define CUSTOM_TILE_COL 4

// Define the full block tile dimensions based on the compute warp layout
// A block computes a (BLOCK_TILE_M*16) * (16*BLOCK_TILE_N) = (BLOCK_TILE_M * BLOCK_TILE_N) tile of C.
#define BLOCK_TILE_M (CUSTOM_TILE_ROW * TILE_M)
#define BLOCK_TILE_N (CUSTOM_TILE_COL * TILE_N)
#define BLOCK_TILE_K TILE_K      

// The WMMA configuration uses 16x16x16 for __half inputs and float accumulator.
typedef fragment<matrix_a, 16, 16, 16, __half, row_major> fragment_a_t;
typedef fragment<matrix_b, 16, 16, 16, __half, row_major> fragment_b_t;
typedef fragment<accumulator, 16, 16, 16, float> fragment_c_t;
// ---------------------------------

//function to convert float array to half array
void float_to_half_array(const DATA_TYPE *input, __half *output, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        output[i] = __float2half(input[i]);
    }
}


/* CPU implementation for verification */
void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
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


void init(int ni, int nj, int nk, DATA_TYPE* alpha, DATA_TYPE* beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
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
            // since we used FP16 for multiplication, we must increase the tolerance.
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
    
    //
    if (deviceProp.major < 7) {
        printf("Error: GPU (Device %d) does not support Tensor Cores (Requires Compute Capability 7.0+).\n", GPU_DEVICE);
        printf("Proceeding with standard GEMM kernel as fallback might be necessary.\n");
        //
    }
    
    printf("Setting device %d with name %s, Compute Capability %d.%d\n", GPU_DEVICE, deviceProp.name, deviceProp.major, deviceProp.minor);
    cudaSetDevice( GPU_DEVICE );
}



__global__ void gemm_wmma_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, const __half *a, const __half *b, DATA_TYPE *c)
{
    //
    __shared__ __half sh_A[BLOCK_TILE_M][BLOCK_TILE_K]; // 64x16
    __shared__ __half sh_B[BLOCK_TILE_K][BLOCK_TILE_N]; // 16x64

    // Identify this thread block's C-tile (top-left corner)
    int block_start_m = blockIdx.y * BLOCK_TILE_M;
    int block_start_n = blockIdx.x * BLOCK_TILE_N;

    // Identify this warp's C-tile within the block
    int warp_row = threadIdx.y;                // 0-CUSTOM_TILE_ROW
    int warp_col = threadIdx.x / 32;           // 0-CUSTOM_TILE_COL
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    //
    fragment_c_t accum;
    
    //
    fill_fragment(accum, 0.0f);

    //
    for (int k_tile = 0; k_tile < (nk / BLOCK_TILE_K); k_tile++)
    {
        int k_base = k_tile * BLOCK_TILE_K;
        
        // Load sh_A (64x16)
        int sh_a_idx0 = tid;                     // 0-511
        int sh_a_idx1 = tid + 512;               // 512-1023
        
        int r0 = sh_a_idx0 / BLOCK_TILE_K; // row 0-31
        int c0 = sh_a_idx0 % BLOCK_TILE_K; // col 0-15
        int r1 = sh_a_idx1 / BLOCK_TILE_K; // row 32-63
        int c1 = sh_a_idx1 % BLOCK_TILE_K; // col 0-15

        int gbl_a_row0 = block_start_m + r0;
        int gbl_a_col0 = k_base + c0;
        int gbl_a_row1 = block_start_m + r1;
        int gbl_a_col1 = k_base + c1;

        if (gbl_a_row0 < ni && gbl_a_col0 < nk) 
            sh_A[r0][c0] = a[gbl_a_row0 * nk + gbl_a_col0];
        else 
            sh_A[r0][c0] = __float2half(0.0f);

        if (gbl_a_row1 < ni && gbl_a_col1 < nk)
            sh_A[r1][c1] = a[gbl_a_row1 * nk + gbl_a_col1];
        else
            sh_A[r1][c1] = __float2half(0.0f);

        // Load sh_B (16x64)
        int sh_b_idx0 = tid;                     // 0-511
        int sh_b_idx1 = tid + 512;               // 512-1023

        int r_b0 = sh_b_idx0 / BLOCK_TILE_N; // row 0-7
        int c_b0 = sh_b_idx0 % BLOCK_TILE_N; // col 0-63
        int r_b1 = sh_b_idx1 / BLOCK_TILE_N; // row 8-15
        int c_b1 = sh_b_idx1 % BLOCK_TILE_N; // col 0-63

        int gbl_b_row0 = k_base + r_b0;
        int gbl_b_col0 = block_start_n + c_b0;
        int gbl_b_row1 = k_base + r_b1;
        int gbl_b_col1 = block_start_n + c_b1;

        if (gbl_b_row0 < nk && gbl_b_col0 < nj)
            sh_B[r_b0][c_b0] = b[gbl_b_row0 * nj + gbl_b_col0];
        else
            sh_B[r_b0][c_b0] = __float2half(0.0f);
        
        if (gbl_b_row1 < nk && gbl_b_col1 < nj)
            sh_B[r_b1][c_b1] = b[gbl_b_row1 * nj + gbl_b_col1];
        else
            sh_B[r_b1][c_b1] = __float2half(0.0f);
        
        __syncthreads();

        //
        fragment_a_t frag_a;
        fragment_b_t frag_b;

        //
        load_matrix_sync(frag_a, &sh_A[warp_row * 16][0], BLOCK_TILE_K); // ld = 16

        //
        load_matrix_sync(frag_b, &sh_B[0][warp_col * 16], BLOCK_TILE_N); // ld = 64
        
        // Perform the WMMA operation: D = D + A * B
        mma_sync(accum, frag_a, frag_b, accum);
        
        // Wait for all shared memory reads to complete before next k-tile
        __syncthreads();
    }
    
    
    // Find this warp's C tile's top-left corner
    int c_row = block_start_m + warp_row * 16;
    int c_col = block_start_n + warp_col * 16;
    
    if (c_row < ni && c_col < nj)
    {
        // Load existing C block (FP32) to apply BETA and accumulate ALPHA * A*B
        fragment_c_t frag_c;
        float* ptr_c = c + c_row * nj + c_col;

        load_matrix_sync(frag_c, ptr_c, nj, mem_row_major);
        
        // C_new = beta * C_old + alpha * (A*B accumulation)
        for (int k = 0; k < frag_c.num_elements; k++) {
            frag_c.x[k] = beta * frag_c.x[k] + alpha * accum.x[k];
        }

        // Store the result back to global memory C
        store_matrix_sync(ptr_c, frag_c, nj, mem_row_major);
    }
}


void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{

    //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // A and B will be stored as __half on the device
    __half *A_gpu_half;
    __half *B_gpu_half;
    DATA_TYPE *C_gpu;

    //Allocate host memory for FP16 conversion
    __half *A_host_half = (__half*)malloc(sizeof(__half) * NI * NK);
    __half *B_host_half = (__half*)malloc(sizeof(__half) * NK * NJ);

    //Convert FP32 host data to FP16 host data
    float_to_half_array(A[0], A_host_half, NI * NK);
    float_to_half_array(B[0], B_host_half, NK * NJ);

    //Allocate device memory (A and B as __half, C as DATA_TYPE/float)
    cudaMalloc((void **)&A_gpu_half, sizeof(__half) * NI * NK);
    cudaMalloc((void **)&B_gpu_half, sizeof(__half) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ); // C remains float/DATA_TYPE

    //Copy data to device
    cudaMemcpy(A_gpu_half, A_host_half, sizeof(__half) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu_half, B_host_half, sizeof(__half) * NK * NJ, cudaMemcpyHostToDevice);
    // C is initially FP32
    cudaMemcpy(C_gpu, C[0], sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    
    //Define launch configuration for WMMA
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); 
    
    //
    dim3 grid((size_t)(ceil( ((float)NJ) / ((float)BLOCK_TILE_N) )), 
              (size_t)(ceil( ((float)NI) / ((float)BLOCK_TILE_M) )));

    /* Start timer. */
    // polybench_start_instruments;
    cudaEventRecord(start);

    // Launch the WMMA kernel
    gemm_wmma_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu_half, B_gpu_half, C_gpu);
    
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time in seconds: %f secs\n", milliseconds * 0.001);
    // polybench_stop_instruments;
    // polybench_print_instruments;

    //Copy result back from device (C_outputFromGpu is FP32/DATA_TYPE)
    cudaMemcpy(C_outputFromGpu[0], C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);     
    
    //Cleanup
    cudaFree(A_gpu_half);
    cudaFree(B_gpu_half);
    cudaFree(C_gpu);
    free(A_host_half);
    free(B_host_half);
}


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
    //
    int ni = NI; // related to A's row
    int nj = NJ; // related to B's col
    int nk = NK; // related to both A's col and B's row.

    //
    DATA_TYPE alpha;
    DATA_TYPE beta;

    //A and B are initialized as FP32 on the host but converted to FP16 for the GPU.
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


