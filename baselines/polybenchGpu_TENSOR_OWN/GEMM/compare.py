import numpy as np
import pandas as pd
import os

# --- Configuration ---
FILE_A = 'input/matrix_a.csv'
FILE_B = 'input/matrix_b.csv'
FILE_OUT = 'out.csv'
FILE_EXPECTED = 'c.csv'

def read_matrix_from_csv(filename):
    """Reads a matrix from a CSV file using pandas."""
    try:
        # Read the CSV, treating all data as floats/integers
        df = pd.read_csv(filename, header=None)
        # Convert the DataFrame to a NumPy array
        matrix = df.values
        print(f"✅ Successfully read matrix from '{filename}' with shape: {matrix.shape}")
        return matrix
    except FileNotFoundError:
        print(f"❌ Error: Input file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error reading matrix from '{filename}': {e}")
        return None

def write_matrix_to_csv(matrix, filename):
    """Writes a NumPy matrix to a CSV file."""
    try:
        # Convert the NumPy array back to a pandas DataFrame
        df = pd.DataFrame(matrix)
        # Write to CSV without header or index
        df.to_csv(filename, header=False, index=False)
        print(f"✅ Successfully wrote result matrix to '{filename}'")
    except Exception as e:
        print(f"❌ Error writing matrix to '{filename}': {e}")

def matrix_multiplication(matrix_a, matrix_b):
    """Performs matrix multiplication using numpy's dot product."""
    try:
        if matrix_a.shape[1] != matrix_b.shape[0]:
            print("❌ Error: Matrix dimensions are incompatible for multiplication.")
            print(f"   Shape A: {matrix_a.shape}, Shape B: {matrix_b.shape}")
            return None
        
        # Perform matrix multiplication (dot product)
        result_matrix = np.dot(matrix_a, matrix_b)
        print(f"✅ Matrix multiplication successful. Result shape: {result_matrix.shape}")
        return result_matrix
        
    except Exception as e:
        print(f"❌ Error during matrix multiplication: {e}")
        return None

def compare_matrices(actual_file, expected_file):
    """Compares the generated output CSV with an expected CSV."""
    print(f"\n--- Comparing '{actual_file}' with '{expected_file}' ---")
    
    # Read the actual and expected matrices
    actual_matrix = read_matrix_from_csv(actual_file)
    expected_matrix = read_matrix_from_csv(expected_file)
    
    if actual_matrix is None or expected_matrix is None:
        print("🛑 Comparison failed due to errors reading one or both files.")
        return

    # Check for shape equality first
    if actual_matrix.shape != expected_matrix.shape:
        print("❌ **FAILURE:** Matrices have different shapes.")
        print(f"   Actual Shape: {actual_matrix.shape}, Expected Shape: {expected_matrix.shape}")
        return

    # Compare elements using numpy's built-in comparison (handling floating-point tolerance)
    # np.allclose is better for floating-point numbers than simple ==
    is_equal = np.allclose(actual_matrix, expected_matrix)

    if is_equal:
        print("✅ **SUCCESS:** The generated matrix matches the expected matrix.")
    else:
        print("❌ **FAILURE:** The generated matrix DOES NOT match the expected matrix.")
        # Optional: Print where the difference is (for debugging)
        # diff = actual_matrix - expected_matrix
        # print(f"Differences (Actual - Expected):\n{diff}")


def main():
    """Main function to orchestrate the process."""
    print("--- Starting Matrix Operation Process ---")

    # 1. Read input matrices
    matrix_a = read_matrix_from_csv(FILE_A)
    matrix_b = read_matrix_from_csv(FILE_B)

    if matrix_a is None or matrix_b is None:
        print("\n🛑 Cannot proceed without both input matrices.")
        return

    # 2. Perform matrix multiplication
    matrix_c = matrix_multiplication(matrix_a, matrix_b)
    
    if matrix_c is None:
        print("\n🛑 Cannot proceed without a valid resulting matrix.")
        return

    # 3. Write the result to a CSV file
    write_matrix_to_csv(matrix_c, FILE_OUT)

    # 4. Compare the output file with the expected file
    if os.path.exists(FILE_EXPECTED):
        compare_matrices(FILE_OUT, FILE_EXPECTED)
    else:
        print(f"\n⚠️ Comparison skipped: Expected output file ('{FILE_EXPECTED}') not found.")
        print("To complete the final step, create this file.")
    
    print("\n--- Process Finished ---")

if __name__ == "__main__":
    main()