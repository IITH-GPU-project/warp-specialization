# ============================
# Makefile for GEMM Variants
# ============================

NVCC = nvcc
ARCH_FLAGS = -arch=sm_120 -O3 -Xcompiler "-fopenmp" -maxrregcount=40 

# Header
INC = gemm_tensor.cuh

# Sources
SRC_BASELINE            = gemm_tensor_baseline.cu
SRC_SHARED              = gemm_tensor_shared_mem.cu
SRC_WARP                = gemm_tensor_warp_spl.cu
SRC_WARP_SINGLE_WARP    = gemm_tensor_warp_spl_single_warp.cu

# Executables
EXE_BASELINE            = gemm_tensor_baseline.exe
EXE_SHARED              = gemm_tensor_shared_mem.exe
EXE_WARP                = gemm_tensor_warp_spl.exe
EXE_WARP_SINGLE_WARP    = gemm_tensor_warp_spl_single_warp.exe

all: $(EXE_BASELINE) $(EXE_SHARED) $(EXE_WARP) $(EXE_WARP_SINGLE_WARP)

$(EXE_BASELINE): $(SRC_BASELINE) $(INC)
	$(NVCC) $(ARCH_FLAGS) $< -o $@

$(EXE_SHARED): $(SRC_SHARED) $(INC)
	$(NVCC) $(ARCH_FLAGS) $< -o $@

$(EXE_WARP): $(SRC_WARP) $(INC)
	$(NVCC) $(ARCH_FLAGS) $< -o $@

$(EXE_WARP_SINGLE_WARP): $(SRC_WARP_SINGLE_WARP) $(INC)
	$(NVCC) $(ARCH_FLAGS) $< -o $@

clean:
	rm -f *.exe *.o
