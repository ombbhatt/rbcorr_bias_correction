import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os
from pathlib import Path

def process_single_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Process a single DataFrame by removing duplicates and ensuring class balance.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple containing:
        - Processed DataFrame
        - Dictionary with processing statistics
    """
    # Store initial statistics
    initial_count = len(df)
    initial_yes = len(df[df['Correct Answer'] == 'Yes'])
    initial_no = len(df[df['Correct Answer'] == 'No'])
    
    # Check for invalid answers
    invalid_answers = df[~df['Correct Answer'].isin(['Yes', 'No'])]
    if not invalid_answers.empty:
        raise ValueError(f"Found invalid answers: {invalid_answers['Correct Answer'].unique()}")
    
    # Remove duplicates based on specified columns
    df_unique = df.drop_duplicates(subset=['Context', 'Question', 'Correct Answer'])
    
    # Get counts after duplicate removal
    post_dedup_yes = len(df_unique[df_unique['Correct Answer'] == 'Yes'])
    post_dedup_no = len(df_unique[df_unique['Correct Answer'] == 'No'])
    
    # Balance classes if needed
    if post_dedup_yes != post_dedup_no:
        # Determine majority and minority classes
        if post_dedup_yes > post_dedup_no:
            majority_class = 'Yes'
            majority_count = post_dedup_yes
            minority_count = post_dedup_no
        else:
            majority_class = 'No'
            majority_count = post_dedup_no
            minority_count = post_dedup_yes
            
        # Randomly sample from majority class to match minority class size
        majority_indices = df_unique[df_unique['Correct Answer'] == majority_class].index
        indices_to_keep = np.random.choice(majority_indices, size=minority_count, replace=False)
        
        # Create balanced dataset
        majority_samples = df_unique.loc[indices_to_keep]
        minority_samples = df_unique[df_unique['Correct Answer'] != majority_class]
        df_balanced = pd.concat([majority_samples, minority_samples])
    else:
        df_balanced = df_unique

    # sort the dataframe by question number
    df_balanced = df_balanced.sort_values(by='Question Number')
    
    # Prepare statistics
    stats = {
        'initial_total': initial_count,
        'initial_yes': initial_yes,
        'initial_no': initial_no,
        'duplicates_removed': initial_count - len(df_unique),
        'after_dedup_yes': post_dedup_yes,
        'after_dedup_no': post_dedup_no,
        'final_yes': len(df_balanced[df_balanced['Correct Answer'] == 'Yes']),
        'final_no': len(df_balanced[df_balanced['Correct Answer'] == 'No']),
        'final_total': len(df_balanced)
    }
    
    return df_balanced, stats

def process_folder(input_folder: str, output_folder: str) -> Dict[str, dict]:
    """
    Process all CSV files in the input folder and save results to output folder.
    
    Args:
        input_folder: Path to folder containing input CSV files
        output_folder: Path to save processed CSV files
    
    Returns:
        Dictionary containing statistics for each processed file
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files in input folder
    input_path = Path(input_folder)
    all_files = list(input_path.iterdir())
    csv_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.csv']
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_folder}")
    
    # Print ignored non-CSV files for transparency
    non_csv_files = [f.name for f in all_files if f.is_file() and f.suffix.lower() != '.csv']
    if non_csv_files:
        print(f"Ignoring non-CSV files: {', '.join(non_csv_files)}")
    
    # Process each file
    all_stats = {}
    for csv_file in csv_files:
        try:
            # Read input file
            df = pd.read_csv(csv_file)
            
            # Process the dataset
            processed_df, stats = process_single_dataset(df)
            
            # Create output filename
            output_file = Path(output_folder) / f"processed_{csv_file.name}"
            
            # Save processed dataset
            processed_df.to_csv(output_file, index=False)
            
            # Store statistics
            all_stats[csv_file.name] = stats
            
            print(f"Successfully processed {csv_file.name}")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            all_stats[csv_file.name] = {'error': str(e)}
    
    return all_stats

# Example usage
if __name__ == "__main__":
    input_folder = "ewokynq-scripts-data/t2q_maker"
    output_folder = "ewokynq-scripts-data/t2q_nodup_nominpairs"
    
    try:
        statistics = process_folder(input_folder, output_folder)
        
        print("\nProcessing Summary:")
        for filename, stats in statistics.items():
            print(f"\nFile: {filename}")
            if 'error' in stats:
                print(f"Error: {stats['error']}")
            else:
                print(f"Initial dataset size: {stats['initial_total']}")
                print(f"Initial class distribution: Yes={stats['initial_yes']}, No={stats['initial_no']}")
                print(f"Duplicates removed: {stats['duplicates_removed']}")
                print(f"After deduplication: Yes={stats['after_dedup_yes']}, No={stats['after_dedup_no']}")
                print(f"Final dataset size: {stats['final_total']}")
                print(f"Final class distribution: Yes={stats['final_yes']}, No={stats['final_no']}")
        
    except Exception as e:
        print(f"Error processing folder: {str(e)}")