import os
import shutil
from pathlib import Path

# Domain categorization
STEM = [
    "abstract_algebra",
    "anatomy", 
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning"
]

SOCIAL_SCI = [
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy"
]

HUMANITIES = [
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions"
]

OTHERS = [
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology"
]

def get_topic_domain(topic_name):
    """
    Determine which domain a topic belongs to.
    
    Args:
        topic_name (str): The topic name extracted from the CSV filename
    
    Returns:
        str: The domain name ('STEM', 'SOCIAL_SCI', 'HUMANITIES', 'OTHERS', or 'UNCATEGORIZED')
    """
    if topic_name in STEM:
        return "STEM"
    elif topic_name in SOCIAL_SCI:
        return "SOCIAL_SCI"
    elif topic_name in HUMANITIES:
        return "HUMANITIES"
    elif topic_name in OTHERS:
        return "OTHERS"
    else:
        return "UNCATEGORIZED"

def organize_csv_files(source_folder="."):
    """
    Organize CSV files into domain-based subfolders.
    
    Args:
        source_folder (str): Path to the folder containing the CSV files (default: current directory)
    """
    source_path = Path(source_folder)
    
    # Check if source folder exists
    if not source_path.exists():
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    # Find all CSV files with the pattern *_processed.csv
    csv_files = list(source_path.glob("*_processed.csv"))
    
    if not csv_files:
        print(f"No files matching '*_processed.csv' found in '{source_folder}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to organize.")
    
    # Track statistics
    moved_files = {"STEM": 0, "SOCIAL_SCI": 0, "HUMANITIES": 0, "OTHERS": 0, "UNCATEGORIZED": 0}
    
    for csv_file in csv_files:
        # Extract topic name from filename (remove '_processed.csv' suffix)
        topic_name = csv_file.stem.replace("_processed", "")
        
        # Determine domain
        domain = get_topic_domain(topic_name)
        
        # Create domain folder if it doesn't exist
        domain_folder = source_path / domain
        domain_folder.mkdir(exist_ok=True)
        
        # Move the file
        destination = domain_folder / csv_file.name
        try:
            shutil.move(str(csv_file), str(destination))
            moved_files[domain] += 1
            print(f"Moved: {csv_file.name} → {domain}/")
        except Exception as e:
            print(f"Error moving {csv_file.name}: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("ORGANIZATION COMPLETE")
    print("="*50)
    for domain, count in moved_files.items():
        if count > 0:
            print(f"{domain}: {count} files")
    
    total_moved = sum(moved_files.values())
    print(f"\nTotal files organized: {total_moved}")
    
    # Warn about uncategorized files
    if moved_files["UNCATEGORIZED"] > 0:
        print(f"\nWarning: {moved_files['UNCATEGORIZED']} files were placed in 'UNCATEGORIZED' folder.")
        print("These files had topic names not found in any domain category.")

def main():
    """
    Main function to run the CSV organizer.
    You can modify the source_folder parameter to point to your specific directory.
    """
    # Change this path to your folder containing the CSV files
    source_folder = "."  # Current directory - change this to your folder path
    
    print("CSV File Organizer by Domain")
    print("="*30)
    
    # Uncomment the line below to specify a different folder
    # source_folder = "/path/to/your/csv/folder"
    
    organize_csv_files(source_folder)

if __name__ == "__main__":
    main()