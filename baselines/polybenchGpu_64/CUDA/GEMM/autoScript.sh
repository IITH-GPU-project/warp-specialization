#!/bin/bash

OUTPUT_CSV="results_fp64_shared.csv"
echo "MatrixSize,GPU_Time(s),CPU_Time(s),Mismatches" > $OUTPUT_CSV

SIZES=(256 512 1024 2048)
CUH_FILE="gemm.cuh"

for SIZE in "${SIZES[@]}"
do
    echo "========================================"
    echo "Running GEMM for Matrix Size = $SIZE"
    echo "========================================"

    # Replace #define NI/NJ/NK in STANDARD_DATASET block
    sed -i "/#  ifdef STANDARD_DATASET/,/#  endif/ s/#define NI .*/#define NI $SIZE/" $CUH_FILE
    sed -i "/#  ifdef STANDARD_DATASET/,/#  endif/ s/#define NJ .*/#define NJ $SIZE/" $CUH_FILE
    sed -i "/#  ifdef STANDARD_DATASET/,/#  endif/ s/#define NK .*/#define NK $SIZE/" $CUH_FILE

    # Rebuild quietly
    make clean > /dev/null
    make > /dev/null

    # Run and capture output
    OUTPUT=$(./gemm.exe 2>&1)

    # Extract GPU time
    GPU_TIME=$(echo "$OUTPUT" | grep -A1 "GPU Time in seconds:" | tail -n1)

    # Extract CPU time
    CPU_TIME=$(echo "$OUTPUT" | grep -A1 "CPU Time in seconds:" | tail -n1)

    # Extract mismatches
    MISMATCH=$(echo "$OUTPUT" | grep -oP "Non-Matching CPU-GPU Outputs Beyond Error Threshold of 0.05 Percent:\s\K[0-9]+")
    MISMATCH=${MISMATCH:-0}

    # Append to CSV
    echo "${SIZE},${GPU_TIME},${CPU_TIME},${MISMATCH}" >> $OUTPUT_CSV

    echo "✅ Logged: Size=${SIZE}, GPU=${GPU_TIME}s, CPU=${CPU_TIME}s, Mismatch=${MISMATCH}"
done

echo "========================================"
echo "All runs complete. Results saved to $OUTPUT_CSV"
echo "========================================"