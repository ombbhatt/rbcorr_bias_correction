import csv
from collections import defaultdict
from pathlib import Path

def prune_properties(
    input_file: str,
    output_file: str,
    property_column: str,
    max_rows_per_property: int = 4
) -> dict[str, tuple[int, int]]:
    """
    Creates a new CSV with at most max_rows_per_property rows for each unique property value.
    Assumes rows for each property are grouped together in the input file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save pruned CSV file
        property_column: Name of the property column
        max_rows_per_property: Maximum number of rows to keep per property
        
    Returns:
        Dictionary with property values as keys and tuples of (original_count, kept_count) as values
    """
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(lambda: [0, 0])  # [original_count, kept_count] for each property
    current_property = None
    current_property_rows = []
    
    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        # Only process first 1000 rows
        
        # for row in reader:
        for i, row in enumerate(reader):
            if i >= 10000000:
                break
            property_value = row[property_column]
            
            # If we've moved to a new property
            if property_value != current_property:
                # Write stored rows for previous property (if any)
                if current_property_rows:
                    rows_to_write = current_property_rows[:max_rows_per_property]
                    writer.writerows(rows_to_write)
                    stats[current_property][1] = len(rows_to_write)
                
                # Start new property
                current_property = property_value
                current_property_rows = [row]
                stats[current_property][0] = 1
                
            else:
                # Continue with current property
                current_property_rows.append(row)
                stats[current_property][0] += 1
        
        # Don't forget to write the last property's rows
        if current_property_rows:
            rows_to_write = current_property_rows[:max_rows_per_property]
            writer.writerows(rows_to_write)
            stats[current_property][1] = len(rows_to_write)
    
    # Convert stats to regular dict with tuples, only return first 1000 items
    return {k: tuple(v) for k, v in stats.items()}

# Example usage
if __name__ == "__main__":
    stats = prune_properties(
        input_file="comps_yesno_random.csv",
        output_file="comps_yn_rand_2prop_all.csv",
        property_column="property",
        max_rows_per_property=2
    )
    
    # Print statistics
    print("\nPruning Statistics:")
    print(f"{'Property':<30} {'Original':<10} {'Kept':<10} {'Pruned':<10}")
    print("-" * 60)
    for prop, (orig, kept) in stats.items():
        print(f"{prop[:30]:<30} {orig:<10} {kept:<10} {orig-kept:<10}")