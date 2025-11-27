# ============================
# Makefile for GEMM Variants
# ============================

# Compiler
NVCC = nvcc
CFLAGS = -O3 -std=c++17 --extended-lambda -arch=sm_120


# Header
INC = gemm_fp32.cuh

# Sources
SRC_BASELINE = gemm_fp32_baseline.cu
SRC_SHARED   = gemm_fp32_shared_mem.cu
SRC_WARP     = gemm_fp32_warp_spl.cu

# Executables
EXE_BASELINE = gemm_fp32_baseline.exe
EXE_SHARED   = gemm_fp32_shared_mem.exe
EXE_WARP     = gemm_fp32_warp_spl.exe

all: $(EXE_BASELINE) $(EXE_SHARED) $(EXE_WARP)

$(EXE_BASELINE): $(SRC_BASELINE) $(INC)
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_SHARED): $(SRC_SHARED) $(INC)
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_WARP): $(SRC_WARP) $(INC)
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f *.exe *.o
