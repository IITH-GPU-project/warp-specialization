# Makefile for Jacobi2D FP32 Implementations

NVCC = nvcc
CFLAGS = -O3 -std=c++17 -arch=sm_120

# Executables
EXE_BASELINE = jacobi2D_fp32_baseline.exe
EXE_POINTER  = jacobi2D_fp32_pointer_swap.exe
EXE_SHARED   = jacobi2D_fp32_shared_mem.exe
EXE_WARP     = jacobi2D_fp32_warp_spl.exe

all: $(EXE_BASELINE) $(EXE_POINTER) $(EXE_SHARED) $(EXE_WARP)

$(EXE_BASELINE): jacobi2D_fp32_baseline.cu
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_POINTER): jacobi2D_fp32_pointer_swap.cu
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_SHARED): jacobi2D_fp32_shared_mem.cu
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_WARP): jacobi2D_fp32_warp_spl.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f *.exe
