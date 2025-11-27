import pandas as pd
import os

def create_performance_summary_excel():
    """
    Reads the five CSV files, calculates GPU time and speedup over the 
    baseline (results_mvt.csv), and saves the result to an Excel file.
    """
    
    # 1. Define file names and the desired column order
    file_info = {
        "mvt": "results_mvt.csv",
        "mvt_shared": "results_mvt_shared.csv",
        "mvt_shared_testing": "results_mvt_shared_testing.csv",
        "mvt_shared_new": "results_mvt_shared_new.csv",
        # "mvt_K1_ws_loading_A_and_y": "results_mvt_K1_ws_loading_A_and_y.csv",
        "mvt_K1_ws_loading_A_and_y_parallelDMA": "results_mvt_K1_ws_loading_A_and_y_parallelDMA.csv",
        "mvt_K1_ws_loading_y": "results_mvt_K1_ws_loading_y.csv",
        "mvt_K1_ws_loading_y_testing": "results_mvt_K1_ws_loading_y_testing.csv",
    }
    
    # The baseline is the first file in the dictionary
    baseline_filename = file_info["mvt"]

    # Dictionary to store DataFrames
    dataframes = {}

    # 2. Load and prepare all CSV files
    print("Loading and processing CSV files...")
    for short_name, file_name in file_info.items():
        if not os.path.exists(file_name):
            print(f"Error: File not found - {file_name}")
            return

        # Read CSV, strip column names of whitespace, and set 'Size' as index
        df = pd.read_csv(file_name)
        df.columns = df.columns.str.strip()
        
        # Convert relevant columns to numeric (coerce errors will turn bad data to NaN)
        df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
        df['Avg_GPU_Time(s)'] = pd.to_numeric(df['Avg_GPU_Time(s)'], errors='coerce')
        
        # Filter for only the necessary columns and set index
        df = df[['Size', 'Avg_GPU_Time(s)']].set_index('Size')
        dataframes[short_name] = df

    # 3. Get the baseline GPU times
    baseline_times = dataframes['mvt']['Avg_GPU_Time(s)']

    # 4. Create the final result DataFrame
    result_df = pd.DataFrame(index=baseline_times.index)

    # 5. Calculate and format the data for each column
    print("Calculating speedup and formatting data...")
    for short_name in file_info.keys():
        current_times = dataframes[short_name]['Avg_GPU_Time(s)']

        # Calculate speedup (baseline / current)
        speedup = baseline_times / current_times

        # Format the cell content: "Avg_GPU_Time(s) (Speedup)"
        # Use string formatting to ensure fixed decimal places
        formatted_data = (
            current_times.round(6).astype(str) + 
            ' (' + speedup.round(2).astype(str) + ')'
        )

        # Add to the result DataFrame
        result_df[short_name] = formatted_data

    # 6. Save the final DataFrame to an Excel file
    output_filename = 'performance_summary.xlsx'
    
    try:
        # Save to Excel, ensuring 'Size' (the index) is written as a column
        result_df.to_excel(
            output_filename, 
            sheet_name='GPU Performance Summary', 
            index=True, 
            header=True,
            index_label='Size' # Name the index column
        )
        print(f"\n✅ Successfully created Excel file: {output_filename}")
        print("The table includes 'Avg_GPU_Time(s)' and 'Speedup over results_mvt.csv' in parentheses.")
    except Exception as e:
        print(f"\n❌ An error occurred while saving to Excel: {e}")

if __name__ == "__main__":
    create_performance_summary_excel()