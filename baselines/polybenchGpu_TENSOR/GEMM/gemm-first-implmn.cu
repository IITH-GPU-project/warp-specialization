#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace nvcuda; 


/*

-> This implementation runs on input matrix size "multiples of 16" of half precision.
-> Input matrix are FP16 and output matrix is FP32.


*/

// Define WMMA tile size
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ================= CSV Utilities =================
// (CSV Utilities remain unchanged)
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
                // Handle parsing error
                data.push_back(__float2half(0.0f)); 
            }
            tempCols++;
        }
        if (cols == 0) cols = tempCols;
        else if (cols != tempCols && tempCols > 0) {
            // Optional warning for ragged CSV files
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

// ================= WMMA GEMM Kernel =================
/**
 * @brief Corrected WMMA GEMM Kernel using explicit wmma:: prefixes for all WMMA types.
 * @note M, N, K should be multiples of 16 for full matrix computation.
 */
__global__ void wmmaGemmFlexible(half *a, half *b, float *c, int M, int N, int K) {
    
    // Determine the tile (16x16) this block/warp is responsible for
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    int start_row = tile_row * WMMA_M;
    int start_col = tile_col * WMMA_N;

    if (start_row >= M || start_col >= N) return;


    // 1. Declare and Initialize Fragments with explicit wmma:: prefix
    
    // Fragment for Matrix A
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    
    // Fragment for Matrix B
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    
    // Fragment for Accumulator C
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);


    // 2. K-Dimension Iteration
    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        
        // Base pointers for the current 16x16 tile in global memory
        const half* a_ptr = a + (start_row * K + k0); 
        const half* b_ptr = b + (k0 * N + start_col); 

        // Load fragments cooperatively from global memory
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        
        // Load B fragment, note that wmma::matrix_b expects a different layout (col_major is default, but here we use row_major)
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        
        // Matrix multiply and accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 3. Store Result
    float* c_ptr = c + (start_row * N + start_col);
    
    // Store the result cooperatively back to global memory (row-major)
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

// ================= Main =================
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

    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        std::cerr << "Warning: M, N, K should ideally be multiples of 16 for this basic kernel.\n";
        std::cerr << "M=" << M << ", N=" << N << ", K=" << K << ". Results may be truncated or incorrect.\n";
    }


    std::vector<float> h_c(M * N, 0.0f);

    // --- Device Memory Allocation and Transfer ---
    half *d_a, *d_b;
    float *d_c;
    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    

    // --- Kernel Launch Configuration ---
    dim3 threadsPerBlock(32, 1); 
    
    dim3 numBlocks(
        (N + WMMA_N - 1) / WMMA_N, 
        (M + WMMA_M - 1) / WMMA_M 
    );

    wmmaGemmFlexible<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);
    
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

    std::cout << "Flexible WMMA GEMM completed. Result saved to c.csv\n";
    std::cout << "Dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";
    return 0;
}
