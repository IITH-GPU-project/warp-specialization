#!/bin/bash

# ============================================
# CONFIGURATION
# ============================================
NUM_RUNS=10
OUTPUT_CSV="Jacobi2D_fp32_results.csv"
OUTPUT_TXT="Jacobi2D_fp32_results.txt"
TIMESTEPS=100

SIZES=(32 64 128 256 512 1024 2048 4096 8192)

CU_FILES=("jacobi2D_fp32_baseline.cu" \
          "jacobi2D_fp32_pointer_swap.cu" \
          "jacobi2D_fp32_shared_mem.cu" \
          "jacobi2D_fp32_warp_spl.cu")

EXE_FILES=("./jacobi2D_fp32_baseline.exe" \
           "./jacobi2D_fp32_pointer_swap.exe" \
           "./jacobi2D_fp32_shared_mem.exe" \
           "./jacobi2D_fp32_warp_spl.exe")

echo "MatrixSize,BL_GPU,BL_CPU,BL_MIS,PS_GPU,PS_CPU,PS_MIS,SH_GPU,SH_CPU,SH_MIS,W_GPU,W_CPU,W_MIS,BL_vs_PS,BL_vs_SH,BL_vs_WARP" \
> $OUTPUT_CSV

echo "===================== Jacobi2D FP32 RESULTS =====================" > $OUTPUT_TXT
echo "" >> $OUTPUT_TXT

# ============================================
# Helper functions
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

write_table() {
    local SIZE=$1
    {
        echo "==================== Jacobi2D FP32 SUMMARY (SIZE = $SIZE) ===================="
        printf "%-30s | %-12s | %-12s | %-8s\n" "Variant" "GPU Time (s)" "CPU Time (s)" "Mismatch"
        echo "--------------------------------------------------------------------------"
        printf "%-30s | %-12.6f | %-12.6f | %-8d\n" "Baseline FP32" $BL_GPU $BL_CPU $BL_MIS
        printf "%-30s | %-12.6f | %-12.6f | %-8d\n" "Pointer Swap" $PS_GPU $PS_CPU $PS_MIS
        printf "%-30s | %-12.6f | %-12.6f | %-8d\n" "Shared Mem" $SH_GPU $SH_CPU $SH_MIS
        printf "%-30s | %-12.6f | %-12.6f | %-8d\n" "Warp Split" $W_GPU $W_CPU $W_MIS
        echo "--------------------------------------------------------------------------"
        printf "%-30s | %-12s\n" "Speedup: Baseline / PointerSwap" "$BL_vs_PS"
        printf "%-30s | %-12s\n" "Speedup: Baseline / SharedMem" "$BL_vs_SH"
        printf "%-30s | %-12s\n" "Speedup: Baseline / WarpSpl" "$BL_vs_WARP"
        echo "==========================================================================="
        echo ""
    } >> $OUTPUT_TXT
}

# ============================================
# MAIN LOOP
# ============================================
for SIZE in "${SIZES[@]}"
do
    echo "Running Size = $SIZE"

    # Update macros in .cu files
    for CU in "${CU_FILES[@]}"; do
        sed -i "s/#define N .*/#define N $SIZE/" "$CU"
        sed -i "s/#define TSTEPS .*/#define TSTEPS $TIMESTEPS/" "$CU"
    done

    make clean > /dev/null
    make > /dev/null

    RES_BL=$(run_and_average "${EXE_FILES[0]}")
    BL_GPU=$(echo $RES_BL | cut -d',' -f1)
    BL_CPU=$(echo $RES_BL | cut -d',' -f2)
    BL_MIS=$(echo $RES_BL | cut -d',' -f3)

    RES_PS=$(run_and_average "${EXE_FILES[1]}")
    PS_GPU=$(echo $RES_PS | cut -d',' -f1)
    PS_CPU=$(echo $RES_PS | cut -d',' -f2)
    PS_MIS=$(echo $RES_PS | cut -d',' -f3)

    RES_SH=$(run_and_average "${EXE_FILES[2]}")
    SH_GPU=$(echo $RES_SH | cut -d',' -f1)
    SH_CPU=$(echo $RES_SH | cut -d',' -f2)
    SH_MIS=$(echo $RES_SH | cut -d',' -f3)

    RES_W=$(run_and_average "${EXE_FILES[3]}")
    W_GPU=$(echo $RES_W | cut -d',' -f1)
    W_CPU=$(echo $RES_W | cut -d',' -f2)
    W_MIS=$(echo $RES_W | cut -d',' -f3)

    # Correct Speedups (baseline / variant)
    BL_vs_PS=$(awk -v a="$BL_GPU" -v b="$PS_GPU" 'BEGIN { if(b==0) print 0; else print a/b }')
    BL_vs_SH=$(awk -v a="$BL_GPU" -v b="$SH_GPU" 'BEGIN { if(b==0) print 0; else print a/b }')
    BL_vs_WARP=$(awk -v a="$BL_GPU" -v b="$W_GPU" 'BEGIN { if(b==0) print 0; else print a/b }')

    echo "$SIZE,$BL_GPU,$BL_CPU,$BL_MIS,$PS_GPU,$PS_CPU,$PS_MIS,$SH_GPU,$SH_CPU,$SH_MIS,$W_GPU,$W_CPU,$W_MIS,$BL_vs_PS,$BL_vs_SH,$BL_vs_WARP" \
    >> $OUTPUT_CSV

    write_table $SIZE
done

# ============================================
# FINAL NORMALIZED TABLE
# ============================================
{
    echo "==================== NORMALIZED SPEEDUP TABLE ===================="
    printf "%-10s | %-10s | %-12s | %-12s | %-12s\n" "Matrix" \
           "Baseline" "PointerSwap" "SharedMem" "WarpSpl"
    echo "-------------------------------------------------------------------"

    NR=2
    for SIZE in "${SIZES[@]}"; do
        ROW=$(awk -v n=$NR 'NR==n {print}' $OUTPUT_CSV)
        BL=$(echo $ROW | cut -d',' -f2)
        PS=$(echo $ROW | cut -d',' -f5)
        SH=$(echo $ROW | cut -d',' -f8)
        W=$(echo $ROW | cut -d',' -f11)

        PSN=$(awk -v a="$BL" -v b="$PS" 'BEGIN {print a/b}')
        SHN=$(awk -v a="$BL" -v b="$SH" 'BEGIN {print a/b}')
        WN=$(awk -v a="$BL" -v b="$W" 'BEGIN {print a/b}')

        printf "%-10s | %-10s | %-12.6f | %-12.6f | %-12.6f\n" "$SIZE" "1.0" "$PSN" "$SHN" "$WN"
        NR=$((NR+1))
    done
    echo "==================================================================="
} >> $OUTPUT_TXT

echo "Done."
