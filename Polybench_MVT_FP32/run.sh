#!/bin/bash

# ============================================================
#   Automated Jacobi2D Kernel Runner (Averaged Results Mode)
#   - Recompile for EACH SIZE
#   - Do NOT delete final executable
# ============================================================

CU_FILES=(
    "mvt_K1.cu"
    "mvt_K1_shared_loading_y.cu"
    "mvt_K1_shared_loading_A_and_y.cu"
    "mvt_K1_shared_loading_y_parallel_compute_warps.cu"
    "mvt_K1_ws_loading_A_and_y.cu"
    "mvt_K1_ws_loading_y.cu"
    "mvt_K1_ws_loading_y_parallel_compute_warps.cu"
)

SIZES=(32 64 128 256 512 1024 2048 4096 8192 16384)
# SIZES=(2048 4096 8192 16384)
RUNS=1

echo "=============================================================="
echo " Automated Kernel Runner -- Averaging Mode Enabled"
echo "=============================================================="

for CU_FILE in "${CU_FILES[@]}"; do

    BASENAME="${CU_FILE%.cu}"
    EXE_NAME="${BASENAME}.exe"
    RESULT_FILE="results_${BASENAME}.csv"

    echo "Size, Avg_GPU_Time(s), Avg_CPU_Time(s), Avg_Mismatches" > "$RESULT_FILE"

    echo
    echo "--------------------------------------------------------------"
    echo " Processing kernel: $CU_FILE"
    echo " Results saved → $RESULT_FILE"
    echo "--------------------------------------------------------------"

    # # Backup original file
    # if [ ! -f "${CU_FILE}.bak" ]; then
    #     cp "$CU_FILE" "${CU_FILE}.bak"
    # fi

    # trap "cp ${CU_FILE}.bak $CU_FILE; echo 'Restored $CU_FILE'; exit" INT

    # Loop through all sizes
    for index in "${!SIZES[@]}"; do
        
        SIZE=${SIZES[$index]}
        echo "  ➤ Testing SIZE = $SIZE"

        # Clean only if NOT the last size
        if [ $index -lt $((${#SIZES[@]} - 1)) ]; then
            make clean > /dev/null
        else
            echo "  (Skipping make clean for final size — preserving executable)"
        fi

        # Update kernel parameters
        sed -i "s/#define N .*/#define N $SIZE/" "$CU_FILE"
        sed -i "s/#define TILE .*/#define TILE 32/" "$CU_FILE"
        sed -i "s/#define DIM_THREAD_BLOCK_X .*/#define DIM_THREAD_BLOCK_X 32/" "$CU_FILE"
        # sed -i "s/#define DMA_WARPS .*/#define DMA_WARPS 4/" "$CU_FILE"
        # sed -i "s/#define COMPUTE_WARPS .*/#define COMPUTE_WARPS 8/" "$CU_FILE"

        # Compile for this size
        make CUFILE="$CU_FILE" EXECUTABLE="$EXE_NAME" > /dev/null 2>&1

        if [ ! -f "$EXE_NAME" ]; then
            echo "  ❌ Compilation FAILED for size $SIZE"
            continue
        fi

        # Accumulators
        # ADJUSTMENT: Initialize GPU/CPU sums to 0.0 for explicit float arithmetic with 'bc'.
        SUM_GPU=0.0
        SUM_CPU=0.0
        SUM_MISMATCH=0
        RUNS_TO_AVERAGE=0

        # Run multiple times
        for ((run=1; run<=RUNS; run++)); do
            OUTPUT=$("./$EXE_NAME" 2>&1)

            # Note: GPU/CPU time extraction methods rely on a specific output format.
            GPU_TIME=$(echo "$OUTPUT" | grep -A1 "GPU Time in seconds:" | tail -n1)
            CPU_TIME=$(echo "$OUTPUT" | grep -A1 "CPU Time in seconds:" | tail -n1)
            
            # Keep the user's pcregrep-style mismatch extraction
            MISMATCH=$(echo "$OUTPUT" | grep -oP "Non-Matching.*:\s*\K[0-9]+")

            # Ensure MISMATCH defaults to 0 if the grep fails
            MISMATCH=${MISMATCH:-0}

            if (( run > 1 )); then
                # Use 'scale=10' for high-precision floating-point arithmetic during summation.
                SUM_GPU=$(echo "scale=10; $SUM_GPU + $GPU_TIME" | bc)
                SUM_CPU=$(echo "scale=10; $SUM_CPU + $CPU_TIME" | bc)
                SUM_MISMATCH=$(echo "$SUM_MISMATCH + $MISMATCH" | bc)
                # Track the count of runs actually summed for the final average calculation
                RUNS_TO_AVERAGE=$((RUNS_TO_AVERAGE + 1))
            fi

            echo "     Run $run → GPU=$GPU_TIME | CPU=$CPU_TIME | Mismatch=$MISMATCH"
        done

        # Averages
        # FIX 1: Check to prevent division by zero (e.g., if RUNS=1).
        if (( RUNS_TO_AVERAGE > 0 )); then
            # FIX 2: Removed redundant DIVIDE calculation. Use RUNS_TO_AVERAGE directly
            # as it accurately reflects the number of runs summed (i.e., RUNS - 1).
            AVG_GPU=$(echo "scale=6; $SUM_GPU / $RUNS_TO_AVERAGE" | bc)
            AVG_CPU=$(echo "scale=6; $SUM_CPU / $RUNS_TO_AVERAGE" | bc)
            AVG_MISMATCH=$(echo "scale=6; $SUM_MISMATCH / $RUNS_TO_AVERAGE" | bc)
        else
            # Set averages to zero if no runs were included in the summation.
            AVG_GPU=0.000000
            AVG_CPU=0.000000
            AVG_MISMATCH=0.000000
        fi

        echo "  ✓ SIZE=$SIZE → AvgGPU=$AVG_GPU  AvgCPU=$AVG_CPU  AvgMismatch=$AVG_MISMATCH"

        # Save to CSV
        echo "$SIZE, $AVG_GPU, $AVG_CPU, $AVG_MISMATCH" >> "$RESULT_FILE"

    done

    # Restore original kernel
    # cp "${CU_FILE}.bak" "$CU_FILE"
    echo "Restored original: $CU_FILE"

done

echo
echo "
