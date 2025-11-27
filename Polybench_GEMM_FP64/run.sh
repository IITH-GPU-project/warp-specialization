#!/bin/bash

# ============================================
# CONFIGURATION
# ============================================
NUM_RUNS=5   # <---- Change this value as needed
OUTPUT_CSV="GEMM_fp64_results.csv"
OUTPUT_TXT="GEMM_fp64_results.txt"

# Matrix sizes to test
SIZES=(32 64 128 256 512 1024 2048 4096 8192)

CUH_FILE="gemm_fp64.cuh"

echo "Number of runs per executable = $NUM_RUNS"
echo "MatrixSize,BL_GPU,BL_CPU,BL_Mis,SH_GPU,SH_CPU,SH_Mis,WARP_GPU,WARP_CPU,WARP_Mis,Speedup_BL_vs_SH,Speedup_SH_vs_WARP" > $OUTPUT_CSV

# Reset TXT file
echo "===================== FP64 GEMM RESULTS =====================" > $OUTPUT_TXT
echo "" >> $OUTPUT_TXT

# Array to store summary for Warp Specialization at the end
declare -A WARP_SUMMARY

# ============================================
# Helper extraction functions
# ============================================

extract_gpu() {
    echo "$1" | grep -A1 "GPU Time in seconds:" | tail -n1
}

extract_cpu() {
    echo "$1" | grep -A1 "CPU Time in seconds:" | tail -n1
}

extract_mis() {
    MIS=$(echo "$1" | grep -oP "Non-Matching.*:\s*\K[0-9]+")
    echo ${MIS:-0}
}

# ============================================
# Run executable multiple times and average
# ============================================

run_and_average() {
    EXE=$1
    TOTAL_GPU=0
    TOTAL_CPU=0
    MAX_MIS=0

    for ((i=1; i<=NUM_RUNS; i++))
    do
        OUT=$($EXE 2>&1)

        GPU=$(extract_gpu "$OUT")
        CPU=$(extract_cpu "$OUT")
        MIS=$(extract_mis "$OUT")

        TOTAL_GPU=$(awk -v a="$TOTAL_GPU" -v b="$GPU" 'BEGIN { print a+b }')
        TOTAL_CPU=$(awk -v a="$TOTAL_CPU" -v b="$CPU" 'BEGIN { print a+b }')

        if (( MIS > MAX_MIS )); then MAX_MIS=$MIS; fi
    done

    AVG_GPU=$(awk -v a="$TOTAL_GPU" -v n="$NUM_RUNS" 'BEGIN { print a/n }')
    AVG_CPU=$(awk -v a="$TOTAL_CPU" -v n="$NUM_RUNS" 'BEGIN { print a/n }')

    echo "$AVG_GPU,$AVG_CPU,$MAX_MIS"
}

# ============================================
# Pretty Table Writer → TXT file
# ============================================

write_table() {
    local SIZE=$1

    {
        echo "==================== GEMM FP64 SUMMARY (SIZE = $SIZE) ===================="
        printf "%-30s | %-12s | %-12s | %-8s\n" "Variant" "GPU Time (s)" "CPU Time (s)" "Mismatch"
        echo "--------------------------------------------------------------------------"
        printf "%-30s | %-12.6f | %-12.6f | %-8d\n" "FP64 Baseline" $BL_GPU $BL_CPU $BL_MIS
        printf "%-30s | %-12.6f | %-12.6f | %-8d\n" "FP64 Shared Memory" $SH_GPU $SH_CPU $SH_MIS
        printf "%-30s | %-12.6f | %-12.6f | %-8d\n" "FP64 Warp Specialization" $W_GPU $W_CPU $W_MIS
        echo "--------------------------------------------------------------------------"
        printf "%-30s | %-12s\n" "Speedup: Baseline / SharedMem" "$BL_vs_SH"
        printf "%-30s | %-12s\n" "Speedup: SharedMem / WarpSpl" "$SH_vs_WARP"
        echo "========================================================================="
        echo ""
    } >> $OUTPUT_TXT
}

# ============================================
# MAIN LOOP
# ============================================

for SIZE in "${SIZES[@]}"
do
    echo "========================================"
    echo "Running GEMM Variants for Size = $SIZE"
    echo "========================================"

    # Update NI/NJ/NK in the gemm.cuh file
    sed -i "/#  ifdef STANDARD_DATASET/,/#  endif/ s/#define NI .*/#define NI $SIZE/" $CUH_FILE
    sed -i "/#  ifdef STANDARD_DATASET/,/#  endif/ s/#define NJ .*/#define NJ $SIZE/" $CUH_FILE
    sed -i "/#  ifdef STANDARD_DATASET/,/#  endif/ s/#define NK .*/#define NK $SIZE/" $CUH_FILE

    # Build
    make clean > /dev/null
    make > /dev/null

    # Run baseline
    RES_BL=$(run_and_average "./gemm_fp64_baseline.exe")
    BL_GPU=$(echo $RES_BL | cut -d',' -f1)
    BL_CPU=$(echo $RES_BL | cut -d',' -f2)
    BL_MIS=$(echo $RES_BL | cut -d',' -f3)

    # Run shared mem
    RES_SH=$(run_and_average "./gemm_fp64_shared_mem.exe")
    SH_GPU=$(echo $RES_SH | cut -d',' -f1)
    SH_CPU=$(echo $RES_SH | cut -d',' -f2)
    SH_MIS=$(echo $RES_SH | cut -d',' -f3)

    # Run warp specialization
    RES_W=$(run_and_average "./gemm_fp64_warp_spl.exe")
    W_GPU=$(echo $RES_W | cut -d',' -f1)
    W_CPU=$(echo $RES_W | cut -d',' -f2)
    W_MIS=$(echo $RES_W | cut -d',' -f3)

    # Speedups
    BL_vs_SH=$(awk -v a="$BL_GPU" -v b="$SH_GPU" 'BEGIN { if (b==0) print 0; else print a/b }')
    SH_vs_WARP=$(awk -v a="$SH_GPU" -v b="$W_GPU" 'BEGIN { if (b==0) print 0; else print a/b }')

    # Store Warp specialization summary
    WARP_SUMMARY[$SIZE]=$SH_vs_WARP

    # Write row to CSV
    echo "${SIZE},${BL_GPU},${BL_CPU},${BL_MIS},${SH_GPU},${SH_CPU},${SH_MIS},${W_GPU},${W_CPU},${W_MIS},${BL_vs_SH},${SH_vs_WARP}" >> $OUTPUT_CSV

    echo "✔ Logged results for size $SIZE"

    # Append formatted summary to TXT file
    write_table $SIZE
done

# ============================================
# Write final Warp Specialization summary
# ============================================

{
    echo ""
    echo "================ WARP SPECIALIZATION SUMMARY ================="
    printf "%-12s | %-20s\n" "Matrix Size" "SharedMem / WarpSpl"
    echo "----------------------------------------"
    for SIZE in "${SIZES[@]}"; do
        printf "%-12s | %-20s\n" "$SIZE" "${WARP_SUMMARY[$SIZE]}"
    done
    echo "=============================================================="
    echo ""
} >> $OUTPUT_TXT

echo "========================================"
echo "All runs complete."
echo "Results saved to:"
echo " - $OUTPUT_CSV"
echo " - $OUTPUT_TXT"
echo "========================================"
