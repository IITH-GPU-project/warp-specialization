/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define TILE 32
#define TILE2 16 

#define N 16384

#define THREADS_PER_WARP 32
#define PARALLEL_DMA 10
#define WARPS_PER_ROW (TILE / THREADS_PER_WARP) // 64 / 32 = 2
#define DMA_WARPS (WARPS_PER_ROW) * PARALLEL_DMA // 2 * 10 = 20
#define COMPUTE_WARPS 2

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
	// 	for (j=0; j<_PB_N; j++) 
	// 	{
 	// 	      	x2[i] = x2[i] + a[j][i] * y2[j];
    //   		}
    // 	}
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
		// 	fail++;
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
    
    int warp_id = threadIdx.x / THREADS_PER_WARP;
	int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int NUM_WARPS = blockDim.x / THREADS_PER_WARP;
    const bool is_loader = (warp_id < DMA_WARPS);
    const int loader_warp_id = warp_id; // For loaders: 0 to DMA_WARPS - 1

    // shared memory for ping-pong buffering
    __shared__ DATA_TYPE ytile[2][TILE];  
    __shared__ DATA_TYPE Atile[2][TILE][TILE];
	
	int row_start = blockIdx.x * TILE;
	int rows_in_block = min(TILE, max(0, N - row_start));
    
    // For compute warps:
	int compute_row_id = (is_loader) ? -1 : (lane_id + (warp_id - DMA_WARPS) * THREADS_PER_WARP);
    const int COMPUTE_THREAD_STRIDE = THREADS_PER_WARP * COMPUTE_WARPS;
    
    int num_tiles = (N - 1) / (TILE) + 1;
    int global_row_A = (TILE * blockIdx.x);
    
    
    // 
    if (is_loader) {
        
        // 
		for (int col = threadIdx.x; col < TILE; col += THREADS_PER_WARP * DMA_WARPS) {
			ytile[0][col] = (col < N) ? y_1[col] : 0.0;
		}
        int row_offset = loader_warp_id / WARPS_PER_ROW; // 0 to PARALLEL_DMA - 1
        
        // warp_in_row_group: Which column chunk within the row this warp is assigned to.
        int warp_in_row_group = loader_warp_id % WARPS_PER_ROW; // 0 to WARPS_PER_ROW - 1
        
        // Parallelized loop over the rows of the TILE
        for(int r = row_offset; r < TILE; r += PARALLEL_DMA) {
            int row = global_row_A + r;
            if(row >= N) {continue;}
            
            // Stride over the TILE column dimension using Warps_Per_Row * Threads_Per_Warp stride.
            for(int col_A = warp_in_row_group * THREADS_PER_WARP + lane_id; 
                col_A < TILE; 
                col_A += WARPS_PER_ROW * THREADS_PER_WARP) 
            {
                int global_col_A = col_A; // Initial tile start column is 0
                
                // 
                if(global_col_A < N) {
                    Atile[0][r][col_A] = a[row * N + global_col_A];
                } else {
                    //
                    Atile[0][r][col_A] = 0.0;
                }
            }
        }
	}
	__syncthreads();
    
    
    //
    for (int t = 0; t < num_tiles; ++t) {

        int current = t % 2;
        int next = (t + 1) % 2;
        int next_tile_start_col = (t + 1) * TILE;
        
        //
        if (is_loader && t + 1 < num_tiles) {

			// int stime = clock();
            
			for (int col = threadIdx.x; col < TILE; col += THREADS_PER_WARP * DMA_WARPS) {
				ytile[next][col] = (next_tile_start_col + col < N) ? y_1[next_tile_start_col + col] : 0.0;
			}
            
            // Load Atile[next]: Parallelized by row and column
            int row_offset = loader_warp_id / WARPS_PER_ROW; 
            int warp_in_row_group = loader_warp_id % WARPS_PER_ROW;

            for(int r = row_offset; r < TILE; r += PARALLEL_DMA) {
                int row = global_row_A + r;
                if(row >= N) continue;
                
                for(int col_A = warp_in_row_group * THREADS_PER_WARP + lane_id; 
                    col_A < TILE; 
                    col_A += WARPS_PER_ROW * THREADS_PER_WARP) 
                {
                    int global_col_A = next_tile_start_col + col_A;
                    if(global_col_A < N) {
                        Atile[next][r][col_A] = a[row * N + global_col_A];
                    } else {
                        Atile[next][r][col_A] = 0.0;
                    }
                }
            }

			// int endtime = clock();
            // if(threadIdx.x % 32 ==0) printf("DMA_clock(%d, %d), warp-id-%d, k_tile_%d: %f\n", blockIdx.x, blockIdx.y, warp_id, t, 0.1*(endtime - stime)/CLOCKS_PER_SEC);

        }
        
        //
		if (!is_loader) {

			//
			// int stime = clock();
            
            // Stride through the rows assigned to this block, using the entire compute section stride.
			for (int r_local = compute_row_id; r_local < rows_in_block; r_local += COMPUTE_THREAD_STRIDE) {
				int row = row_start + r_local;
				if (row >= N) break;
                
				DATA_TYPE sum = 0.0;
				
				// Standard dot product using shared tiles
				for (int k = 0; k < TILE; ++k) {
                    sum += Atile[current][r_local][k] * ytile[current][k];
				}
                
                // Write back result
				atomicAdd(&x1[row], sum);
			}

			// int endtime = clock();
            // if(threadIdx.x % 32 ==0) printf("COMPUTE_clock(%d, %d), warp-id-%d, k_tile_%d: %f\n", blockIdx.x, blockIdx.y, warp_id, t, 0.1*(endtime - stime)/CLOCKS_PER_SEC);

		}
		__syncthreads(); 
	}
}
	

__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
	// shared memory tile
	__shared__ DATA_TYPE Atile[TILE2][TILE2+1];
	DATA_TYPE sum = 0.0;

	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int t = 0; t < (N - 1) / TILE2 + 1; ++t){ //Loading tile A
		 int tile_row = t * TILE2 + threadIdx.y;
		 int tile_col = blockIdx.x * TILE2 + threadIdx.x;

		 if (tile_row < N && tile_col < N)
			  Atile[threadIdx.y][threadIdx.x] = a[tile_row * N + tile_col];
		 else
			  Atile[threadIdx.y][threadIdx.x] = (DATA_TYPE)0.0;
		 __syncthreads();

		//  Now partial dot product
		if (col < N){
		 for (int k = 0; k < TILE2; ++k){
				int r = t * TILE2 + k;
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

    // Explicitly set the L1/Shared memory preference.
    // cudaError_t setCacheError = cudaFuncSetCacheConfig(mvt_kernel1_ws, cudaFuncCachePreferShared);
    // if (setCacheError != cudaSuccess) {
    //     fprintf(stderr, "Warning: Failed to set CUDA cache config to prefer shared memory: %s\n", cudaGetErrorString(setCacheError));
    // }

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