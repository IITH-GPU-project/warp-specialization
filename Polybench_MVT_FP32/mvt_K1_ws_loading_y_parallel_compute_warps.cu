/**
 * mvt_K1_ws_loading_y_testing.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 * This code implements only kernel one (matrix * vector) which gives 1.5x-1.6x speedup compare to shared mem impl. 
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1
#define PARALLEL_COMPUTE 2
#define ROW_COMPUTE_WARPS 11

#define DMA_WARPS 1
#define COMPUTE_WARPS (ROW_COMPUTE_WARPS * PARALLEL_COMPUTE)
#define THREADS_PER_WARP 32

#define TILE 64
#define TILE2 16

#define N 256

#define NUM_THREADS_IN_BLOCK ((COMPUTE_WARPS + DMA_WARPS) * THREADS_PER_WARP )

#include "mvt.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU


void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        x1[i] = ((DATA_TYPE) i) / N;
        x2[i] = ((DATA_TYPE) i + 1) / N;
        y1[i] = ((DATA_TYPE) i + 3) / N;
        y2[i] = ((DATA_TYPE) i + 4) / N;
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE) i*j) / N;
        }
    }
}


void runMvt(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
    int i, j;
    
    for (i=0; i<_PB_N; i++) 
    {
        for (j=0; j<N; j++) 
        {
            x1[i] = x1[i] + a[i][j] * y1[j];
            }
        }
    
    // for (i=0; i<_PB_N; i++) 
    // {
    //  for (j=0; j<_PB_N; j++) 
    //  {
    //           x2[i] = x2[i] + a[j][i] * y2[j];
    //           }
    //  }
}


void compareResults(int n, DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n))
{
    int i, fail;
    fail = 0;
    
    for (i=0; i<n; i++) 
    {
        if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }

        // if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        // {
        //  fail++;
        // }
    }
    
    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice( GPU_DEVICE );
}

// row-wise
__global__ void mvt_kernel1_ws(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
    const int Loader_Warps = DMA_WARPS;
    const int Parallel_Compute = PARALLEL_COMPUTE;
    const int Threads_Per_Warp = THREADS_PER_WARP;
    const int Tile = TILE;
    
    int warp_id = threadIdx.x / Threads_Per_Warp;
    int lane_id = threadIdx.x % Threads_Per_Warp;
    const int NUM_WARPS = blockDim.x / Threads_Per_Warp;
    const int NUM_COMPUTE_WARPS = NUM_WARPS - Loader_Warps;
    const bool is_loader = (warp_id < Loader_Warps);
    int compute_warp_id = warp_id - Loader_Warps; // compute_warp_id ranges from 0 to NUM_COMPUTE_WARPS - 1

    // shared memory for ping-pong buffering
    __shared__ DATA_TYPE ytile[2][TILE];
    
    int row_start = blockIdx.x * Tile; 
    int rows_in_block = min(Tile, max(0, N - row_start)); 
    
    int num_tiles = (N - 1) / (Tile) + 1;

    //
    int row_group_start = (compute_warp_id / Parallel_Compute) * Parallel_Compute;
    int warp_in_row_group = compute_warp_id % Parallel_Compute;
    const int ROW_STEP = NUM_COMPUTE_WARPS / Parallel_Compute;

    // Correct column distribution each of the Parallel_Compute warps
    // handles a non-overlapping chunk of the Tile.
    const int chunk_size = Tile / Parallel_Compute;
    const int col_start = warp_in_row_group * chunk_size;
    const int col_end = col_start + chunk_size;

    // ---------------------------------------------
    
    // Initial load of first tile
    if (is_loader) {
        for (int col = threadIdx.x; col < Tile; col += Threads_Per_Warp * Loader_Warps) {
            ytile[0][col] = (col < N) ? y_1[col] : (DATA_TYPE)0.0;
        }
    }
    __syncthreads();
    
    //
    for (int t = 0; t < num_tiles; ++t) {
        int current = t % 2;
        int next = (t + 1) % 2;
        
        // Loader warps Load next tile
        if (is_loader && t + 1 < num_tiles) {
            int stime = clock();
            for (int col = threadIdx.x; col < Tile; col += Threads_Per_Warp * Loader_Warps) {
                ytile[next][col] = ( (t + 1) * Tile + col < N ) ? y_1[(t + 1) * Tile + col] : (DATA_TYPE)0.0;
            }
            // int endtime = clock();
            // if(threadIdx.x % Threads_Per_Warp ==0) printf("DMA_clock(%d, %d), warp-id-%d, k_tile_%d: %f\n", blockIdx.x, blockIdx.y, warp_id, t, 0.1*(endtime - stime)/CLOCKS_PER_SEC);
        }
        
        // Compute warps Perform partial dot product
        if (!is_loader) {
            // int stime = clock();

            // Loop over rows assigned to this compute warp's group
            for (int r_in_block = row_group_start / Parallel_Compute; r_in_block < rows_in_block; r_in_block += ROW_STEP) {
                int row = row_start + r_in_block;
                if (row >= N) break;

                DATA_TYPE sum = (DATA_TYPE)0.0;
                

                for (int k = col_start + lane_id; k < col_end; k += Threads_Per_Warp) {
                    int c = t * Tile + k; // Global column index
                    if (c < N)
                        sum += a[row * N + c] * ytile[current][k];
                }
                
                // Warp-level reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2)
                {
                    sum += __shfl_down_sync(0xffffffff, sum, offset);
                }
                
                // Only the first lane in each warp updates the result atomically.
                if (lane_id == 0) {
                    atomicAdd(&x1[row], sum);
                }
            }
            
            // int endtime = clock();
            // if(threadIdx.x % Threads_Per_Warp ==0) printf("COMPUTE_clock(%d, %d), warp-id-%d, k_tile_%d: %f\n", blockIdx.x, blockIdx.y, warp_id, t, 0.1*(endtime - stime)/CLOCKS_PER_SEC);
        }
        __syncthreads(); 
    }
}

__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
    // shared memory tile
    const int Tile2 = TILE2;
    __shared__ DATA_TYPE Atile[TILE2][TILE2+1];
    DATA_TYPE sum = (DATA_TYPE)0.0;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int t = 0; t < (N - 1) / Tile2 + 1; ++t){ //Loading tile A
         int tile_row = t * Tile2 + threadIdx.y;
         int tile_col = blockIdx.x * Tile2 + threadIdx.x;

         if (tile_row < N && tile_col < N)
              Atile[threadIdx.y][threadIdx.x] = a[tile_row * N + tile_col];
         else
              Atile[threadIdx.y][threadIdx.x] = (DATA_TYPE)0.0;
         __syncthreads();

        //  Now partial dot product
        if (col < N){
         for (int k = 0; k < Tile2; ++k){
                  int r = t * Tile2 + k;
                  if (r < N)
                      sum += Atile[k][threadIdx.x] * y_2[r];
            }
        }
        __syncthreads();

    }
         if (col < N)
             x2[col] += sum;
}

void mvtCuda(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y_1, N, n), DATA_TYPE POLYBENCH_1D(y_2, N, n), 
             DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n), DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n))
{
    DATA_TYPE* a_gpu;
    DATA_TYPE* x1_gpu;
    DATA_TYPE* x2_gpu;
    DATA_TYPE* y_1_gpu;
    DATA_TYPE* y_2_gpu;

    cudaMalloc((void **)&a_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&x1_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&x2_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_1_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_2_gpu, sizeof(DATA_TYPE) * N);
    cudaMemcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    
    // TILE as block size
    dim3 block(NUM_THREADS_IN_BLOCK);
    dim3 grid((size_t)ceil((float)N/ ((float)TILE)));

    dim3 block2(TILE2, TILE2);
    dim3 grid2((size_t)ceil((float)N/ ((float)TILE2)));

    /* Start timer. */
    polybench_start_instruments;
    
    mvt_kernel1_ws<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu);
    // mvt_kernel2<<<grid2,block2>>>(a_gpu,x2_gpu,y_2_gpu);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;
    
    cudaMemcpy(x1_outputFromGpu, x1_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(x2_outputFromGpu, x2_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);    
    
    cudaFree(a_gpu);
    cudaFree(x1_gpu);
    cudaFree(x2_gpu);
    cudaFree(y_1_gpu);
    cudaFree(y_2_gpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
             DATA_TYPE POLYBENCH_1D(x1,N,n),
             DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, x1[i]);
    fprintf (stderr, DATA_PRINTF_MODIFIER, x2[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main()
{
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
    POLYBENCH_1D_ARRAY_DECL(x1,DATA_TYPE,N,n);
    POLYBENCH_1D_ARRAY_DECL(x2,DATA_TYPE,N,n);
    POLYBENCH_1D_ARRAY_DECL(x1_outputFromGpu,DATA_TYPE,N,n);
    POLYBENCH_1D_ARRAY_DECL(x2_outputFromGpu,DATA_TYPE,N,n);
    POLYBENCH_1D_ARRAY_DECL(y_1,DATA_TYPE,N,n);
    POLYBENCH_1D_ARRAY_DECL(y_2,DATA_TYPE,N,n);

    init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));
    
    GPU_argv_init();

    mvtCuda(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu));

    #ifdef RUN_ON_CPU
    
        /* Start timer. */
        polybench_start_instruments;

        //run the algorithm on the CPU
        runMvt(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));

        /* Stop and print timer. */
        printf("CPU Time in seconds:\n");
        polybench_stop_instruments;
        polybench_print_instruments;
    
        compareResults(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(x2_outputFromGpu));


    #else //print output to stderr so no dead code elimination

        print_array(n, POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu));

    #endif //RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(x1);
    POLYBENCH_FREE_ARRAY(x2);
    POLYBENCH_FREE_ARRAY(x1_outputFromGpu);
    POLYBENCH_FREE_ARRAY(x2_outputFromGpu);
    POLYBENCH_FREE_ARRAY(y_1);
    POLYBENCH_FREE_ARRAY(y_2);

    return 0;
}

#include "../../common/polybench.c"