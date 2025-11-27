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

#define PARALLEL_COMPUTE 2
#define ROW_COMPUTE_WARPS 11 // TILE/THREADS_PER_WARP = 512/32 = 16
#define THREADS_PER_WARP 32
#define COMPUTE_WARPS (ROW_COMPUTE_WARPS * PARALLEL_COMPUTE) // 32

#define TILE 32
#define TILE2 16 // or 32 (for 2D block) FOR Column-wise


#define N 16384


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
__global__ void mvt_kernel1(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
    const int Parallel_Compute = PARALLEL_COMPUTE;
    const int Threads_Per_Warp = THREADS_PER_WARP;
    const int Tile = TILE;
    const int ROW_WARPS = ROW_COMPUTE_WARPS;
    const int NUM_WARPS = COMPUTE_WARPS; //

	// shared memory tile
	__shared__ DATA_TYPE ytile[TILE];
    
	int warp_id = threadIdx.x / Threads_Per_Warp;
    int lane_id = threadIdx.x % Threads_Per_Warp;
    int row_group_id = warp_id / Parallel_Compute; 
    int warp_in_group = warp_id % Parallel_Compute;
    int row_start = blockIdx.x * ROW_WARPS; // Block is responsible for ROW_WARPS number of rows
    int row = row_start + row_group_id;

    // Define the column chunk for this specific warp within its row group.
    const int chunk_size = Tile / Parallel_Compute;
    const int col_start = warp_in_group * chunk_size;
    const int col_end = col_start + chunk_size;

	// Tiling loop over the columns
	for (int t = 0; t < (N - 1) / Tile + 1; ++t){
		
        
        if (threadIdx.x < Tile) // All threads in the block cooperate
        {
            int col_global = t * Tile + threadIdx.x;
            ytile[threadIdx.x] = (col_global < N) ? y_1[col_global] : (DATA_TYPE)0.0;
        }
		 __syncthreads();

		 if (row < N && row_group_id < ROW_WARPS)
         {
            DATA_TYPE sum = (DATA_TYPE)0.0;
            

            // Inner loop iterates over the assigned column chunk using lane_id stride.
            for (int k = col_start + lane_id; k < col_end; k += Threads_Per_Warp)
            {
                int c = t * Tile + k;
                if (c < N)
                    sum += a[row * N + c] * ytile[k];
            }
            
            // Warp-level reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            // 
            if (lane_id == 0) 
            {
                atomicAdd(&x1[row], sum);
            }
		 }
		 __syncthreads();
	}
}


// column-wise (mvt_kernel2)
__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
    const int Parallel_Compute = PARALLEL_COMPUTE; // Used for row chunking (transpose context)
    const int Threads_Per_Warp = THREADS_PER_WARP;
    const int Tile2 = TILE2;
    const int ROW_WARPS_2D = TILE2 / Threads_Per_Warp; // Assuming 2D block (TILE2 x TILE2)
    const int BLOCK_THREADS_X = TILE2;
    const int BLOCK_THREADS_Y = TILE2;
    const int BLOCK_WARPS_X = TILE2 / Threads_Per_Warp;

    // shared memory tile
	__shared__ DATA_TYPE Atile[TILE2][TILE2+1];
	DATA_TYPE sum = (DATA_TYPE)0.0;
    
    int row_in_block = threadIdx.y;
    int col_in_block = threadIdx.x;
    
    // Global column index (output index)
	int col = blockIdx.x * BLOCK_THREADS_X + col_in_block;
    
    // Warp IDs
    int warp_id_x = threadIdx.x / Threads_Per_Warp;
    int warp_id_y = threadIdx.y; // Each row in the block is a thread

    // The entire (TILE2 x TILE2) block cooperates on a single output column chunk.
    // Since this is MVT transpose (column-wise), each output element x2[col] is a dot product across a column of A.
    // The inner loop iterates over rows (k) of the tile.
    
    // Row parallelization (which corresponds to column of A in A^T*y)
    int row_group_id;
    int row_in_group;
    
    if (TILE2 >= Threads_Per_Warp) { // Vertical partitioning
        // Warps are naturally aligned vertically. Each warp handles a column of the tile.
        // Parallel_Compute warps collaborate on the inner loop (dot product) iteration space.
        // We will assign the dot product iteration (over rows k) to the threads.
        // A (TILE2 x TILE2) block contains TILE2 rows of threads.
        
        row_group_id = row_in_block / Parallel_Compute; // Which of the simplified row tasks this thread is on
        row_in_group = row_in_block % Parallel_Compute; // Used to partition the inner loop (k) space
    } else {
        // TILE2 < Threads_Per_Warp, less common for MVT.
        row_group_id = threadIdx.y; // Each row is unique
        row_in_group = 0;
    }

	for (int t = 0; t < (N - 1) / Tile2 + 1; ++t){ // Tiling loop over the shared dimension (k in dot product)
		 int tile_row = t * Tile2 + row_in_block;
		 int tile_col = blockIdx.x * Tile2 + col_in_block;

         // 1. Load A and y_2 (assuming A is loaded here, y_2 is needed globally)
		 if (tile_row < N && tile_col < N)
			  Atile[row_in_block][col_in_block] = a[tile_row * N + tile_col];
		 else
			  Atile[row_in_block][col_in_block] = (DATA_TYPE)0.0;
		 __syncthreads();

		// 2. Now partial dot product: x2[col] = x2[col] + A[row][col] * y2[row]
        // The dot product runs over the row index (r/k). This is where PARALLEL_COMPUTE is applied.
		if (col < N){
            // Each of the TILE2 threads in the column (threadIdx.x is constant) calculates a portion of the inner sum.
            
            const int chunk_size = Tile2 / Parallel_Compute; // Parallelize the k-loop (dot product)
            const int k_start = row_in_group * chunk_size;
            const int k_end = k_start + chunk_size;

            // threadIdx.y is used to distribute the k-loop
		    for (int k = k_start; k < k_end; ++k){
				int r = t * Tile2 + k;
                
                // Thread at (row_in_block, col_in_block) reads Atile[k][col_in_block]
				if (r < N)
					 sum += Atile[k][col_in_block] * y_2[r]; // y_2[r] is global
		 	}
		}
		__syncthreads();

	}
    
    // The sum is currently only a partial sum based on the 'k' chunk assigned to this thread (row_in_group).
    // A reduction is needed across all Parallel_Compute threads contributing to the same final column.
    
    // Since TILE2 is small (16), we can use the shared memory block for reduction.
    // The threads for a given col (constant threadIdx.x) are threads 0...15 in y-dimension.
    
    // 3. Reduction across the y-dimension (threads working on the same output column 'col')
    if (col < N){
        
        __shared__ DATA_TYPE s_reduction_buffer[TILE2];
        s_reduction_buffer[threadIdx.y] = sum;
        __syncthreads();
        
        // Only threads in the first row_group_id need to perform the final summation
        if (row_in_group == 0) { // Thread is one of the Parallel_Compute groups
            
            // Sum across all contributions in the column (y-dimension)
            if (row_in_block == 0) {
                // Thread (0, threadIdx.x) performs final column sum
                DATA_TYPE final_sum = (DATA_TYPE)0.0;
                for (int i = 0; i < TILE2; ++i) {
                    final_sum += s_reduction_buffer[i];
                }
                x2[col] += final_sum;
            }
        }
    }
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
    // For mvt_kernel1: BLOCK = COMPUTE_WARPS * THREADS_PER_WARP = 32 * 32 = 1024 (must be multiple of 32 and <= 1024)
    // TILE is 512, which is ROW_COMPUTE_WARPS * THREADS_PER_WARP. This does not match the block size definition.
    // To implement the requested row-stepping, we must use a block size that is a multiple of ROW_COMPUTE_WARPS * PARALLEL_COMPUTE * THREADS_PER_WARP.
    
    // Correcting block/grid for mvt_kernel1 based on the new constants:
    const int BLOCK_SIZE_K1 = ROW_COMPUTE_WARPS * PARALLEL_COMPUTE * THREADS_PER_WARP; // 16 * 2 * 32 = 1024
	dim3 block(BLOCK_SIZE_K1); 
	dim3 grid((size_t)ceil((float)N/ ((float)ROW_COMPUTE_WARPS))); // Grid size: N / ROW_COMPUTE_WARPS

	dim3 block2(TILE2, TILE2);
	dim3 grid2((size_t)ceil((float)N/ ((float)TILE2)));

	/* Start timer. */
  	polybench_start_instruments;
	
	mvt_kernel1<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu);
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