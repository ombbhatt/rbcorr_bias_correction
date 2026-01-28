import pandas as pd
import re
from tqdm import tqdm

def generate_question(property_str, prefix):
    """Generate a question based on the property string and prefix."""
    property_lower = property_str.lower()
    
    # Handle "can" properties
    if property_lower.startswith('can '):
        return f"Can {prefix} {property_lower[4:]}?"
    
    # Handle "has" properties
    elif property_lower.startswith('has '):
        return f"Does {prefix} have {property_lower[4:]}?"
    
    # Handle "is" properties - new template
    elif property_lower.startswith('is '):
        return f"Is {prefix} {property_lower[3:]}?"
    
    # Handle "was" properties
    elif property_lower.startswith('was '):
        return f"Was {prefix} {property_lower[4:]}?"
    
    # Handle verb+s properties (default case)
    else:
        # Remove 's' from the first word if it ends in 's'
        words = property_lower.split()
        if words[0].endswith('s'):
            words[0] = words[0][:-1]
        if words[0].endswith('ies'):
            words[0] = words[0][:-3] + 'y'
        modified_property = ' '.join(words)
        return f"Does {prefix} {modified_property}?"

def process_chunk(chunk):
    """Process a chunk of the dataset."""
    rows = []
    
    for _, row in chunk.iterrows():
        # Generate yes question
        yes_question = generate_question(row['property'], row['prefix_acceptable'])
        rows.append({
            'property': row['property'],
            'prefix': row['prefix_acceptable'],
            'Context': "''",
            'Question': yes_question,
            'Correct Answer': 'Yes',
            'negative_sample_type': row['negative_sample_type']
        })
        
        # Generate no question
        no_question = generate_question(row['property'], row['prefix_unacceptable'])
        rows.append({
            'property': row['property'],
            'prefix': row['prefix_unacceptable'],
            'Context': "''",
            'Question': no_question,
            'Correct Answer': 'No',
            'negative_sample_type': row['negative_sample_type']
        })
    
    return pd.DataFrame(rows)

def transform_dataset(input_file, output_file, chunk_size=1000):
    """Transform the input dataset into questions with yes/no answers using chunking."""
    # Get total number of rows for progress bar
    total_rows = sum(1 for _ in pd.read_csv(input_file, chunksize=chunk_size))
    
    # Process chunks and write to output file
    first_chunk = True

    counter = 0
    
    print(f"Processing approximately {total_rows * 2:,} questions...")
    
    with tqdm(total=total_rows) as pbar:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            counter += 1
            # Process the chunk
            output_chunk = process_chunk(chunk)
            
            # Write to CSV
            if first_chunk:
                output_chunk.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                output_chunk.to_csv(output_file, index=False, mode='a', header=False)
            
            pbar.update(len(chunk))
            # if counter == 2:
            #     break

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = 'comps.csv'
    OUTPUT_FILE = f'comps_yesno_v2.csv'
    CHUNK_SIZE = 1000  # Adjust this based on your system's memory
    
    try:
        transform_dataset(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE)
        print(f"\nTransformation complete! Output saved to {OUTPUT_FILE}")
        
        # Print sample of output
        print("\nSample of generated questions:")
        sample_output = pd.read_csv(OUTPUT_FILE, nrows=5)
        print(sample_output)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")