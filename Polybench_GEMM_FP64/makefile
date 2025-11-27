# ============================
# Makefile for GEMM Variants
# ============================

# Compiler
NVCC = nvcc
CFLAGS = -O3 -std=c++17 --extended-lambda -arch=sm_120

# Header
INC = gemm_fp64.cuh

# Sources
SRC_BASELINE = gemm_fp64_baseline.cu
SRC_SHARED   = gemm_fp64_shared_mem.cu
SRC_WARP     = gemm_fp64_warp_spl.cu

# Executables
EXE_BASELINE = gemm_fp64_baseline.exe
EXE_SHARED   = gemm_fp64_shared_mem.exe
EXE_WARP     = gemm_fp64_warp_spl.exe

all: $(EXE_BASELINE) $(EXE_SHARED) $(EXE_WARP)

$(EXE_BASELINE): $(SRC_BASELINE) $(INC)
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_SHARED): $(SRC_SHARED) $(INC)
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_WARP): $(SRC_WARP) $(INC)
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f *.exe *.o
