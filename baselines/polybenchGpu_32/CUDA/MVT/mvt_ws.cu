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

#define N 1024

// KERNEL 1
#define TILE 64 // or 512 or 1024 depending on GPU  FOR  Row-wise
#define num_threads_in_block 256//2*TILE 
#define DMA_WARPS1 4//1

// KERNEL 2
#define TILE2 16 // or 32 (for 2D block) FOR Column-wise
#define num_threads_in_block_x 16//16
#define num_threads_in_block_y 16//32
#define DMA_WARPS2 7//1

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
	
	for (i=0; i<_PB_N; i++) 
	{
		for (j=0; j<_PB_N; j++) 
		{
 		      	x2[i] = x2[i] + a[j][i] * y2[j];
      		}
    	}
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

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
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
__global__ void mvt_kernel1_ws(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1, int Loader_Warps)
{
    int warp_id = threadIdx.x / 32;
	int lane_id = threadIdx.x % 32;
    const int NUM_WARPS = blockDim.x / 32;
    const bool is_loader = (warp_id < Loader_Warps);

    // shared memory for ping-pong buffering
    __shared__ DATA_TYPE ytile[2][TILE];  
	
	int row_start = blockIdx.x * TILE;
	int rows_in_block = min(TILE, max(0, N - row_start));
	int compute_row = (is_loader) ? -1 : (lane_id + (warp_id - Loader_Warps) * 32);
    int num_tiles = (N - 1) / (TILE) + 1;
    
    // Loading the y tile for the first time (prefetch)
    if (is_loader) {
		for (int col = threadIdx.x; col < TILE; col += 32*Loader_Warps) {
			ytile[0][col] = (col < N) ? y_1[col] : 0.0;
		}
	}
	__syncthreads();
    
    // Starting ping-pong buffering loop
    for (int t = 0; t < num_tiles; ++t) {
        int current = t % 2;
        int next = (t + 1) % 2;
        
        // Loader warps: Load next tile
        if (is_loader && t + 1 < num_tiles) {
			for (int col = threadIdx.x; col < TILE; col += 32*Loader_Warps) {
				ytile[next][col] = ( (t + 1) * TILE + col < N ) ? y_1[(t + 1) * TILE + col] : 0.0;
			}
        }
        
        // Compute warps: Process current tile
		if (!is_loader) {
			for (int r = compute_row; r < rows_in_block; r += 32 * (NUM_WARPS - Loader_Warps)) {
				int row = row_start + r;
				if (row >= N) continue;
				DATA_TYPE sum = 0.0;
				
				for (int k = 0; k < TILE; ++k) {
					int c = t * TILE + k;
					if (c < N)
						sum += a[row * N + c] * ytile[current][k];
				}
				x1[row] += sum;
			}
		}
		__syncthreads(); 
	}
}
	


// column-wise loading y into shared memory
__global__ void mvt_kernel2_ws(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2, int Loader_Warps)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    const int NUM_WARPS = blockDim.x / 32;
    const bool is_loader = (warp_id < Loader_Warps);

    __shared__ DATA_TYPE y_tile[2][TILE2];
    
    int col_start = blockIdx.x * TILE2;
    int cols_in_block = min(TILE2, max(0, N - col_start));
    int compute_col = (is_loader) ? -1 : (lane_id + (warp_id - Loader_Warps) * 32);
    int num_tiles = (N + TILE2 - 1) / TILE2;
    
    
    // Prefetch: Load first y_2 tile
    if (is_loader) {
        for (int row = threadIdx.x; row < TILE2; row += 32 * Loader_Warps) {
            y_tile[0][row] = (row < N) ? y_2[row] : 0.0;
        }
    }
    __syncthreads();
    
    // starting ping-pong buffering
    for (int t = 0; t < num_tiles; ++t) {
        int current = t % 2;
        int next = (t + 1) % 2;
        
        // Loader warps: Load next y_2 tile
        if (is_loader && t + 1 < num_tiles) {
            for (int row = threadIdx.x; row < TILE2; row += 32 * Loader_Warps) {
                int global_row = (t + 1) * TILE2 + row;
                y_tile[next][row] = (global_row < N) ? y_2[global_row] : 0.0;
            }
        }
        
        // Compute warps: Process current tile
        if (!is_loader) {
			for (int c = compute_col; c < cols_in_block; c += 32 * (NUM_WARPS - Loader_Warps)) {
				int col = col_start + c;
				if (col >= N) continue;
				DATA_TYPE sum = 0.0;

				for (int k = 0; k < TILE2; ++k) {
					int row = t * TILE2 + k;
					if (row < N) {
						sum += a[row * N + col] * y_tile[current][k];
					}
				}
				x2[col] += sum;
        }
        __syncthreads();
    	}
	}
}


// column-wise loading A into shared memory
// __global__ void mvt_kernel2_ws_A(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2, int Loader_Warps)
// {
// 	int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
// 	int warp_id = local_tid / 32;
// 	int lane_id = local_tid % 32;
// 	const int NUM_WARPS = blockDim.x * blockDim.y / 32;
// 	const bool is_loader = (warp_id < Loader_Warps);

// 	// shared memory for ping-pong buffering
// 	__shared__ DATA_TYPE Atile[2][TILE2][TILE2+1];  

	
// 	int col_start = blockIdx.x * TILE2;
// 	int cols_in_block = min(TILE2, max(0, N - col_start));
// 	int compute_col = (is_loader) ? -1 : (lane_id + (warp_id - Loader_Warps) * 32);
// 	int num_tiles = (N - 1) / (TILE2) + 1;
	
// 	// Loading the A tile for the first time (prefetch)
// 	if (is_loader) {
// 		for (int idx = local_tid; idx < TILE2 * TILE2; idx += 32 * Loader_Warps) {
// 			int r = idx / TILE2;
// 			int c = idx % TILE2;
// 			Atile[0][r][c] = (r < N && (col_start + c) < N) ? a[r*N + (col_start + c)] : 0.0;
// 		}
// 	}
// 	__syncthreads();
	
// 	// Starting ping-pong buffering loop
// 	for (int t = 0; t < num_tiles; ++t) {
// 		int current = t % 2;
// 		int next = (t + 1) % 2;
		
// 		// Loader warps: Load next tile
// 		if (is_loader && t + 1 < num_tiles) {
// 			for (int idx = local_tid; idx < TILE2 * TILE2; idx += 32 * Loader_Warps) {
// 				int r = idx / TILE2;
// 				int c = idx % TILE2;
// 				int global_r = (t + 1) * TILE2 + r;
// 				Atile[next][r][c] = (global_r < N && (col_start + c) < N) ? a[global_r*N + (col_start + c)] : 0.0;
// 			}
// 		}

// 		// Compute warps: Process current tile
// 		if (!is_loader) {
// 			for (int c = compute_col; c < cols_in_block; c += 32 * (NUM_WARPS - Loader_Warps)) {
// 				int col = col_start + c;
// 				if (col >= N) continue;
// 				DATA_TYPE sum = 0.0;
// 				for (int k = 0; k < TILE2; ++k) {
// 					int r = t * TILE2 + k;
// 					if (r < N)
// 						sum += Atile[current][k][c] * y_2[r];
// 				}
// 				x2[col] += sum;
// 			}
// 		}
// 		__syncthreads(); 
// 	}
// }



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
	dim3 block(num_threads_in_block);
	dim3 grid((size_t)ceil((float)N/ ((float)TILE)));

	dim3 block2(num_threads_in_block_x, num_threads_in_block_y);
	dim3 grid2((size_t)ceil((float)N/ ((float)TILE2)));

	/* Start timer. */
  	polybench_start_instruments;
	
	mvt_kernel1_ws<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu, DMA_WARPS1);
	mvt_kernel2_ws<<<grid2,block2>>>(a_gpu,x2_gpu,y_2_gpu, DMA_WARPS2);
	cudaThreadSynchronize();

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
