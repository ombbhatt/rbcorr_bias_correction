import json
import random
from pathlib import Path
from collections import defaultdict

def filter_and_balance_mnli(input_file, output_file=None, seed=42):
    """
    Filter mnli dataset to remove entries with gold_label="-" and balance classes and genres.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (default: input_file with '_balanced' suffix)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Read and filter entries - organize by label and genre
    print("Reading dataset...")
    # Structure: valid_entries[label][genre] = [entries]
    valid_entries = defaultdict(lambda: defaultdict(list))
    total_count = 0
    filtered_count = 0
    all_genres = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            entry = json.loads(line.strip())
            gold_label = entry.get('gold_label')
            genre = entry.get('genre')
            
            if gold_label != '-':
                valid_entries[gold_label][genre].append(entry)
                all_genres.add(genre)
            else:
                filtered_count += 1
    
    print(f"Total entries: {total_count}")
    print(f"Filtered out (gold_label='-'): {filtered_count}")
    print(f"\nGenres found: {sorted(all_genres)}")
    print(f"Number of genres: {len(all_genres)}")
    
    print(f"\nClass and genre distribution before balancing:")
    for label in sorted(valid_entries.keys()):
        print(f"\n{label}:")
        for genre in sorted(valid_entries[label].keys()):
            print(f"  {genre}: {len(valid_entries[label][genre])}")
    
    # Find the minimum count across all label-genre combinations
    min_count_per_combination = float('inf')
    for label in valid_entries:
        for genre in all_genres:
            count = len(valid_entries[label].get(genre, []))
            if count > 0:  # Only consider combinations that exist
                min_count_per_combination = min(min_count_per_combination, count)
    
    # Calculate target size per label-genre combination
    target_per_combination = min(min_count_per_combination, 2000 // len(all_genres))
    
    print(f"\nTarget entries per label-genre combination: {target_per_combination}")
    print(f"This will give {target_per_combination * len(all_genres)} entries per label")
    print(f"Total entries: {target_per_combination * len(all_genres) * 3}")
    
    # Sample equally from each label-genre combination
    balanced_entries = []
    for label in sorted(valid_entries.keys()):
        for genre in sorted(all_genres):
            entries = valid_entries[label].get(genre, [])
            if len(entries) >= target_per_combination:
                sampled = random.sample(entries, target_per_combination)
                balanced_entries.extend(sampled)
            elif len(entries) > 0:
                print(f"Warning: {label}/{genre} has only {len(entries)} entries (need {target_per_combination})")
                balanced_entries.extend(entries)
    
    # Shuffle the combined dataset
    random.shuffle(balanced_entries)
    
    # Verify balance
    print("\nFinal distribution:")
    label_counts = defaultdict(int)
    genre_counts = defaultdict(int)
    label_genre_counts = defaultdict(lambda: defaultdict(int))
    
    for entry in balanced_entries:
        label = entry['gold_label']
        genre = entry['genre']
        label_counts[label] += 1
        genre_counts[genre] += 1
        label_genre_counts[label][genre] += 1
    
    print("\nBy label:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")
    
    print("\nBy genre:")
    for genre in sorted(genre_counts.keys()):
        print(f"  {genre}: {genre_counts[genre]}")
    
    print("\nBy label and genre:")
    for label in sorted(label_genre_counts.keys()):
        print(f"\n{label}:")
        for genre in sorted(label_genre_counts[label].keys()):
            print(f"  {genre}: {label_genre_counts[label][genre]}")
    
    # Determine output file name
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_balanced_genre{input_path.suffix}"
    
    # Write balanced dataset
    print(f"\nWriting balanced dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in balanced_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nDone! Balanced dataset has {len(balanced_entries)} entries.")
    
    return output_file

if __name__ == "__main__":
    # Example usage - update with your actual file path
    input_file = "multinli_1.0/multinli_1.0_dev_matched.jsonl"  # Change this to your file name
    
    output_file = filter_and_balance_mnli(input_file)
    print(f"\nBalanced dataset saved to: {output_file}")