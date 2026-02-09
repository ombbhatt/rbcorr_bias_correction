import os
import pandas as pd
import glob

def combine_csvs(folder_path, output_filename='combined_data.csv'):
    """
    Combine all CSV files in a folder into one CSV file.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        output_filename (str): Name of the output combined CSV file
    """
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return
    
    print(f"Found {len(csv_files)} CSV files to combine:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Read and combine all CSV files
    combined_df = pd.DataFrame()
    
    for i, file in enumerate(csv_files):
        try:
            # Read CSV file
            df = pd.read_csv(file)
            
            # For the first file, keep the header
            if i == 0:
                combined_df = df
            else:
                # For subsequent files, append data without header
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            print(f"Added {len(df)} rows from {os.path.basename(file)}")
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    # Save the combined data
    output_path = os.path.join(folder_path, output_filename)
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nCombined CSV saved as: {output_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")

# Example usage
if __name__ == "__main__":
    # Set the folder path containing your CSV files
    folder_path = "mmlu-scripts-data"  # Change this to your folder path
    
    # Optional: specify a custom output filename
    output_filename = "all_domains_processed.csv"
    
    # Combine the CSV files
    combine_csvs(folder_path, output_filename)