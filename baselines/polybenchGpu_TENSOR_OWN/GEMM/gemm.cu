#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>

// Use nvcuda namespace for WMMA types and functions
using namespace nvcuda;

// ================= WMMA and Tiling Parameters =================
// WMMA standard tile sizes for FP16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Thread Block Tiling Parameters (defines the work done by one thread block)
// 4 warps efficiently compute a 2x2 grid of WMMA tiles (2*16 x 2*16 = 32x32 output block)
#define BLOCK_M 32  // M dimension of the block tile
#define BLOCK_N 32  // N dimension of the block tile
#define BLOCK_K 32  // K dimension of the block tile (must be multiple of WMMA_K=16)

// Thread block size (fixed at 128 for 4 warps, standard for WMMA tiling)
#define THREADS_PER_BLOCK 128
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / 32)
#define WMMA_M_TILES_PER_BLOCK (BLOCK_M / WMMA_M) // 32/16 = 2
#define WMMA_N_TILES_PER_BLOCK (BLOCK_N / WMMA_N) // 32/16 = 2

// The number of WMMA operations each thread block performs: (2x2)
// TILE_SIZE_C must be (BLOCK_M / WMMA_M) * (BLOCK_N / WMMA_N)

// Utility macro for 2D to 1D index conversion
#define SHARED_MEM_INDEX(r, c, ld) ((r) * (ld) + (c))


// ================= CSV Utilities (Unchanged) =================
std::vector<half> readCSV(const std::string &filename, int &rows, int &cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;
    std::vector<half> data;
    rows = 0;
    cols = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        int tempCols = 0;
        while (std::getline(ss, val, ',')) {
            try {
                float f_val = std::stod(val);
                data.push_back(__float2half(f_val));
            } catch (const std::exception& e) {
                data.push_back(__float2half(0.0f)); 
            }
            tempCols++;
        }
        if (cols == 0) cols = tempCols;
        else if (cols != tempCols && tempCols > 0) {
        }
        rows++;
    }
    return data;
}

void writeCSV(const std::string &filename, const std::vector<float> &data, int rows, int cols) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << data[i * cols + j];
            if (j != cols - 1) file << ",";
        }
        file << "\n";
    }
}

// ================= Tiled GEMM Kernel with Shared Memory and WMMA =================
/**
 * @brief GEMM using Shared Memory Tiling and WMMA (Tensor Cores).
 * Computes C = A * B where A, B are FP16 and C is FP32.
 */
__global__ void wmmaTiledGemm(const half *A, const half *B, float *C, int M, int N, int K) {
    
    // --- 1. Shared Memory Allocation ---
    // A tile: BLOCK_M x BLOCK_K (Padded for bank conflicts: +1 on K dimension)
    __shared__ half sA[BLOCK_M][BLOCK_K + 1];
    
    // B tile: BLOCK_K x BLOCK_N (Padded for bank conflicts: +1 on N dimension)
    // Canonical layout for C=A*B: B is stored row-major in sB, but accessed column-major by WMMA.
    __shared__ half sB[BLOCK_K][BLOCK_N + 1];

    // --- 2. Thread and Block Indexing ---
    
    // Global row and column index of the block's work area in C
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    // Local warp index within the block (0 to 3 for 128 threads)
    int warp_idx = threadIdx.x / 32; 
    // Local thread index within the warp (0 to 31)
    int lane_idx = threadIdx.x % 32; 

    // Determine which WMMA tile this warp is responsible for in the C block
    // Warps are assigned in a 2x2 grid (4 warps for 32x32 block)
    int warp_tile_m = warp_idx / WMMA_N_TILES_PER_BLOCK; // 0 or 1
    int warp_tile_n = warp_idx % WMMA_N_TILES_PER_BLOCK; // 0 or 1

    // Global starting row/col for the warp's C accumulation
    int warp_start_row = block_row + warp_tile_m * WMMA_M;
    int warp_start_col = block_col + warp_tile_n * WMMA_N;

    // Print block setup information
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("--- Kernel Setup ---\n");
        printf("Grid: %dx%d. Block: %dx%d. Threads: %d. Tile: %dx%d\n", 
               gridDim.x, gridDim.y, blockDim.x, blockDim.y, THREADS_PER_BLOCK, BLOCK_M, BLOCK_N);
        printf("Block (0,0) starts at global C index (%d, %d)\n", block_row, block_col);
    }
    
    // Print warp assignment information
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Warp 0 (Thread 0) targets C tile at (%d, %d) (Size %dx%d)\n", warp_start_row, warp_start_col, WMMA_M, WMMA_N);
    }

    // --- 3. Fragment Initialization ---
    
    // Accumulator fragment (FP32)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // A fragment (FP16, row_major)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    
    // B fragment (FP16, COL_MAJOR) - CRITICAL FIX for C=A*B (Required by WMMA hardware)
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

    // --- 4. Tiled K-Loop ---
    int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int t = 0; t < num_k_tiles; ++t) {
        int k_offset = t * BLOCK_K;

        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("\n--- K-Tile %d/%d (k_offset: %d) ---\n", t + 1, num_k_tiles, k_offset);
        }

        // --- Shared Memory Load (Cooperative) ---
        // Load A tile (Total elements: BLOCK_M * BLOCK_K = 1024)
        #pragma unroll
        for (int i = 0; i < (BLOCK_M * BLOCK_K) / THREADS_PER_BLOCK; ++i) {
            int linear_idx = threadIdx.x + i * THREADS_PER_BLOCK;
            int load_row = linear_idx / BLOCK_K; // M dimension index
            int load_col = linear_idx % BLOCK_K; // K dimension index

            // Global A index
            int global_A_idx = (block_row + load_row) * K + (k_offset + load_col);
            
            if ( (block_row + load_row) < M && (k_offset + load_col) < K ) {
                sA[load_row][load_col] = A[global_A_idx];
            } else {
                sA[load_row][load_col] = __float2half(0.0f);
            }
        }

        // Load B tile (Total elements: BLOCK_K * BLOCK_N = 1024)
        #pragma unroll
        for (int i = 0; i < (BLOCK_K * BLOCK_N) / THREADS_PER_BLOCK; ++i) {
            int linear_idx = threadIdx.x + i * THREADS_PER_BLOCK;
            int load_row = linear_idx / BLOCK_N; // K dimension index
            int load_col = linear_idx % BLOCK_N; // N dimension index

            // Global B index (B is K x N, stored row-major)
            int global_B_idx = (k_offset + load_row) * N + (block_col + load_col);

            if ( (k_offset + load_row) < K && (block_col + load_col) < N ) {
                // FIX: Load B untransposed into K x N+1 shared memory space: sB[K_index][N_index]
                sB[load_row][load_col] = B[global_B_idx];
            } else {
                // Padding unused elements with zero
                sB[load_row][load_col] = __float2half(0.0f);
            }
        }

        // Wait for all threads to finish loading the tiles
        __syncthreads();
        
        // --- Shared Memory Diagnostics ---
        if (t == 0 && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            // Print the first element of the loaded shared tiles
            printf("DIAG: Shared Memory Load Complete. sA[0][0]=%f, sB[0][0]=%f\n", 
                   __half2float(sA[0][0]), 
                   __half2float(sB[0][0]));
            // Print a corner element for bounds check sanity
            printf("DIAG: sA[%d][%d]=%f, sB[%d][%d]=%f\n", 
                   BLOCK_M - 1, BLOCK_K - 1, __half2float(sA[BLOCK_M-1][BLOCK_K-1]),
                   BLOCK_K - 1, BLOCK_N - 1, __half2float(sB[BLOCK_K-1][BLOCK_N-1])); // Back to K x N sB layout
        }

        // --- WMMA Multiplication ---
        
        // Loop over the K-dimension of the shared memory tile (from 0 to BLOCK_K/WMMA_K)
        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += WMMA_K) {
            
            // Pointers to the 16x16x16 WMMA sub-tile in shared memory
            const half *a_ptr_sm = &sA[warp_tile_m * WMMA_M][k_step];
            
            // B is K x N+1. Accessing sB[K_start][N_start]
            const half *b_ptr_sm = &sB[k_step][warp_tile_n * WMMA_N]; // K, N index (K is now outer)

            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_step == 0) {
                printf("WMMA STEP: K_Inner=%d. Load A from sA[0][%d]. Load B from sB[%d][%d].\n", 
                       k_step, k_step, k_step, warp_tile_n * WMMA_N);
                // Stride for B is BLOCK_N + 1 (pitch of K x N+1 array)
                printf("WMMA STEP: Strides A: %d, B: %d (Canonical Stride)\n", BLOCK_K + 1, BLOCK_N + 1);
            }
            
            // Load A fragment (stride is BLOCK_K + 1 due to padding)
            wmma::load_matrix_sync(a_frag, a_ptr_sm, BLOCK_K + 1);

            // Load B fragment (stride is BLOCK_N + 1)
            wmma::load_matrix_sync(b_frag, b_ptr_sm, BLOCK_N + 1);

            // --- Fragment Data Diagnostic ---
            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && k_step == 0) {
                printf("FRAG DIAG: a_frag[0]=%f, a_frag[1]=%f\n", 
                       __half2float(a_frag.x[0]), 
                       __half2float(a_frag.x[1]));
                printf("FRAG DIAG: b_frag[0]=%f, b_frag[1]=%f\n", 
                       __half2float(b_frag.x[0]), 
                       __half2float(b_frag.x[1]));
            }
            // --- End Fragment Data Diagnostic ---

            // Matrix Multiply and Accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Wait for all warps to finish computing with the current tile
        __syncthreads();
    } // End of K-loop

    // --- 5. Store Result to Global Memory (Robust Bounds Check) ---

    // 5A. Store the fragment to a local temporary buffer (FP32)
    float temp_C[WMMA_M * WMMA_N];
    // Leading dimension is WMMA_N for contiguous storage in the temporary array
    wmma::store_matrix_sync(temp_C, c_frag, WMMA_N, wmma::mem_row_major);
    
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("\n--- Final Accumulation Check ---\n");
        // Print the first accumulated value (temp_C[0,0])
        printf("FINAL: C[warp 0, 0] accumulated value (temp_C[0]): %f\n", temp_C[0]);
        printf("FINAL: Storing to global C...\n");
    }
    
    // 5B. Perform per-element bounds check before writing to global memory C
    #pragma unroll
    for (int i = 0; i < WMMA_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WMMA_N; ++j) {
            int global_row = warp_start_row + i;
            int global_col = warp_start_col + j;

            // Write to global memory only if the element is within the matrix bounds (M x N)
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = temp_C[i * WMMA_N + j];
            }
        }
    }
}

// ================= Main (Host Code Updated) =================
int main() {
    
    int A_rows, A_cols, B_rows, B_cols;
    
    // --- Data Loading (Host) ---
    std::vector<half> h_a;
    std::vector<half> h_b;
    try {
        h_a = readCSV("input/matrix_a.csv", A_rows, A_cols);
        h_b = readCSV("input/matrix_b.csv", B_rows, B_cols);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << ". Ensure 'input/matrix_a.csv' and 'input/matrix_b.csv' exist.\n";
        return -1;
    }

    if (A_cols != B_rows) {
        std::cerr << "Matrix dimensions do not match for multiplication (A_cols != B_rows)!\n";
        return -1;
    }

    int M = A_rows, N = B_cols, K = A_cols;
    std::vector<float> h_c(M * N, 0.0f);

    // --- Device Memory Allocation and Transfer ---
    half *d_a, *d_b;
    float *d_c;
    // NOTE: In production code, memory allocation failure checks should be added here
    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    

    // --- Kernel Launch Configuration ---
    // Thread block size is fixed for efficiency with WMMA
    dim3 threadsPerBlock(THREADS_PER_BLOCK); 
    
    // Grid size calculates the number of BLOCK_M x BLOCK_N tiles needed
    dim3 numBlocks(
        (N + BLOCK_N - 1) / BLOCK_N, 
        (M + BLOCK_M - 1) / BLOCK_M 
    );

    // Launch the WMMA Tiled kernel
    wmmaTiledGemm<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);
    
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(kernel_error) << "\n";
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return -1;
    }

    cudaDeviceSynchronize();

    // --- Data Retrieval and Output ---
    cudaMemcpy(h_c.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    writeCSV("c.csv", h_c, M, N);

    // --- Cleanup ---
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    std::cout << "WMMA Tiled GEMM (Tensor Core) completed. Result saved to c.csv\n";
    std::cout << "Dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Using BLOCK_M=" << BLOCK_M << ", BLOCK_N=" << BLOCK_N << ", BLOCK_K=" << BLOCK_K << " tiles.\n";
    return 0;
}
