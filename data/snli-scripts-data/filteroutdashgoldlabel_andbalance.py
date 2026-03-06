import json
import random
from pathlib import Path
from collections import Counter

def filter_and_balance_snli(input_file, output_file=None, seed=42):
    """
    Filter SNLI dataset to remove entries with gold_label="-" and balance classes.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (default: input_file with '_balanced' suffix)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Read and filter entries
    print("Reading dataset...")
    valid_entries = {'entailment': [], 'neutral': [], 'contradiction': []}
    total_count = 0
    filtered_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            entry = json.loads(line.strip())
            gold_label = entry.get('gold_label')
            
            if gold_label != '-':
                valid_entries[gold_label].append(entry)
            else:
                filtered_count += 1
    
    print(f"Total entries: {total_count}")
    print(f"Filtered out (gold_label='-'): {filtered_count}")
    print(f"\nClass distribution before balancing:")
    for label, entries in valid_entries.items():
        print(f"  {label}: {len(entries)}")
    
    # Find minimum class size
    # min_size = min(len(entries) for entries in valid_entries.values())
    min_size = 2000
    print(f"\nBalancing to {min_size} entries per class...")
    
    # Randomly sample from each class
    balanced_entries = []
    for label, entries in valid_entries.items():
        sampled = random.sample(entries, min_size)
        balanced_entries.extend(sampled)
    
    # Shuffle the combined dataset
    random.shuffle(balanced_entries)
    
    # Determine output file name
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_balanced_smaller{input_path.suffix}"
    
    # Write balanced dataset
    print(f"\nWriting balanced dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in balanced_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nDone! Balanced dataset has {len(balanced_entries)} entries.")
    print(f"({min_size} per class × 3 classes)")
    
    return output_file

if __name__ == "__main__":
    # Example usage - update with your actual file path
    input_file = "snli_1.0/snli_1.0_test.jsonl"  # Change this to your file name
    
    output_file = filter_and_balance_snli(input_file)
    print(f"\nBalanced dataset saved to: {output_file}")