import json
import csv
from pathlib import Path

def jsonl_to_csv(input_file, output_file=None):
    """
    Convert SNLI JSONL format to CSV with Context, Question, Correct Answer columns.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output CSV file (default: input_file with .csv extension)
    """
    # Label mapping
    label_map = {
        'entailment': '0',
        'neutral': '1',
        'contradiction': '2'
    }
    
    # Determine output file name
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}.csv"
    
    print(f"Reading from {input_file}...")
    print(f"Writing to {output_file}...")
    
    # Read JSONL and write CSV
    rows_written = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['Genre', 'Context', 'Question', 'Correct Answer'])
        
        for line in infile:
            entry = json.loads(line.strip())
            
            # Extract fields
            genre = entry['genre']
            sentence1 = entry['sentence1']
            sentence2 = entry['sentence2']
            gold_label = entry['gold_label']
            
            # Skip entries with no gold label (shouldn't happen if filtered, but just in case)
            if gold_label == '-':
                continue
            
            # Create question
            question = f"Premise: {sentence1} Hypothesis: {sentence2}"
            
            # Map label to number
            correct_answer = label_map[gold_label]
            
            # Write row: Context (empty string), Question, Correct Answer
            writer.writerow([genre, "''", question, correct_answer])
            rows_written += 1
    
    print(f"\nDone! Wrote {rows_written} rows to {output_file}")
    return output_file

if __name__ == "__main__":
    # Example usage - update with your actual file path
    input_file = "multinli_1.0/multinli_1.0_dev_matched_balanced_genre.jsonl"  # Change this to your file name
    
    output_file = jsonl_to_csv(input_file)
    print(f"\nCSV file saved to: {output_file}")