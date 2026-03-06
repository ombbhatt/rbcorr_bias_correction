import pandas as pd
import csv
from pathlib import Path

def filter_csv(
    input_file: str,
    output_file: str,
    column_name: str,
    filter_value: any,
    chunk_size: int = 10000
) -> tuple[int, int]:
    """
    Creates a new CSV file containing only rows where the specified column matches the filter value.
    Removes trailing commas from each row. Uses chunking to handle large files efficiently.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save filtered CSV file
        column_name: Name of column to filter on
        filter_value: Value to match in the column
        chunk_size: Number of rows to process at once (default: 10000)
    
    Returns:
        tuple containing (total_rows_processed, rows_matched)
    """
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Read the header to get column names
    df_header = pd.read_csv(input_file, nrows=0)
    columns = df_header.columns.tolist()
    
    total_rows = 0
    matched_rows = 0
    first_chunk = True
    
    # Process the CSV in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Filter rows in this chunk
        filtered_chunk = chunk[chunk[column_name] == filter_value]
        
        if len(filtered_chunk) > 0:
            if first_chunk:
                mode = 'w'
                first_chunk = False
            else:
                mode = 'a'
            
            # Write to output file without pandas to_csv to avoid trailing comma
            with open(output_file, mode, newline='') as f:
                writer = csv.writer(f, lineterminator='\n')
                
                # Write header if this is the first chunk
                if mode == 'w':
                    writer.writerow(columns)
                
                # Write data rows
                writer.writerows(filtered_chunk.values)
        
        total_rows += len(chunk)
        matched_rows += len(filtered_chunk)
        
    return total_rows, matched_rows

# Example usage
if __name__ == "__main__":
    # Example with a sample dataset
    total, matched = filter_csv(
        input_file="comps_yesno_v2.csv",
        output_file="comps_yesno_random.csv",
        column_name="negative_sample_type",
        filter_value="random"
    )
    
    print(f"Processed {total:,} rows")
    print(f"Found {matched:,} matching rows")