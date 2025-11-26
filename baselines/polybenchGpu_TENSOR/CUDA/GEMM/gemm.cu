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
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 1

// WMMA tile size (fixed computation size by tensor cores)
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16 // The K dimension is fixed by the WMMA API (16 for 16x16x16)

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
    // The starting point for the C tile computed by this thread block
    int start_m = blockIdx.y * TILE_M;
    int start_n = blockIdx.x * TILE_N;

    // Dimensions of the fragments
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Number of WMMA operations per thread block to cover the tile
    const int K_TILES = NK / WMMA_K;
    const int M_TILES = TILE_M / WMMA_M;
    const int N_TILES = TILE_N / WMMA_N;

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

    // Outer loop over the K dimension (tiles of size WMMA_K=16)
    for (int tile_k = 0; tile_k < K_TILES; tile_k++)
    {
        // Load A and B fragments from global memory
        fragment_a_t frag_a[M_TILES];
        fragment_b_t frag_b[N_TILES];

        // Load M * K fragments for A
        for (int i = 0; i < M_TILES; i++)
        {
            const __half* ptr_a = a + (start_m + i * WMMA_M) * NK + tile_k * WMMA_K;
            // Ensure we only load if the tile is within bounds (NI and NK)
            if (start_m + i * WMMA_M < NI && tile_k * WMMA_K < NK) {
                 load_matrix_sync(frag_a[i], ptr_a, NK);
            } else {
                 fill_fragment(frag_a[i], __float2half(0.0f));
            }
        }
        
        // Load K * N fragments for B
        for (int j = 0; j < N_TILES; j++)
        {
            const __half* ptr_b = b + (tile_k * WMMA_K) * NJ + (start_n + j * WMMA_N);
            // Ensure we only load if the tile is within bounds (NK and NJ)
            if (tile_k * WMMA_K < NK && start_n + j * WMMA_N < NJ) {
                load_matrix_sync(frag_b[j], ptr_b, NJ);
            } else {
                fill_fragment(frag_b[j], __float2half(0.0f));
            }
        }

        //
        for (int i = 0; i < M_TILES; i++)
        {
            for (int j = 0; j < N_TILES; j++)
            {
                if (start_m + i * WMMA_M < NI && start_n + j * WMMA_N < NJ) {
                    // mma_sync(D, A, B, C); -> D is same as C in this case
                    mma_sync(accum[i][j], frag_a[i], frag_b[j], accum[i][j]);
                }
            }
        }
    }
    
    // Final Store (Accumulate and Write back to C)
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
                load_matrix_sync(frag_c, ptr_c, NJ, mem_row_major);
                
                // C_new = beta * C_old + alpha * (A*B accumulation is already in accum)
                for (int k = 0; k < frag_c.num_elements; k++) {
                    frag_c.x[k] = beta * frag_c.x[k] + alpha * accum[i][j].x[k];
                }

                // Store the result back to global memory C
                store_matrix_sync(ptr_c, frag_c, NJ, mem_row_major);
            }
        }
    }
}


void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    
    //
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //
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
    cudaMemcpy(C_gpu, C[0], sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    
    //Define launch configuration for WMMA
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((size_t)(ceil( ((float)NJ) / ((float)TILE_N) )), (size_t)(ceil( ((float)NI) / ((float)TILE_M) )));

    /* Start timer. */
    // polybench_start_instruments;
    cudaEventRecord(start);

    // Launch the WMMA kernel
    gemm_wmma_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu_half, B_gpu_half, C_gpu);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /* Stop and print timer. */
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

