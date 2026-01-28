import os
import pandas as pd
import shutil
from pathlib import Path

# Define the domain categories
DOMAIN_CATEGORIES = {
    "STEM": [
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
    ],
    
    "SOCIAL_SCI": [
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
    ],
    
    "HUMANITIES": [
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
    ],
    
    "OTHERS": [
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
}

def create_domain_to_category_map():
    """Create a mapping from domain to category."""
    domain_to_category = {}
    for category, domains in DOMAIN_CATEGORIES.items():
        for domain in domains:
            domain_to_category[domain] = category
    return domain_to_category

def get_csv_filenames_from_domain(domain_path):
    """Get all CSV filenames from a domain directory."""
    csv_files = []
    if os.path.exists(domain_path):
        csv_files = [f for f in os.listdir(domain_path) if f.endswith('.csv')]
    return csv_files

def reorganize_mmlu_results(base_path):
    """
    Reorganize MMLU results by grouping domains into categories.
    
    Args:
        base_path (str): Path to the directory containing 'outputs' folder
    """
    base_path = Path(base_path)
    outputs_path = base_path / "outputs"
    
    if not outputs_path.exists():
        print(f"Error: {outputs_path} does not exist!")
        return
    
    # Create domain to category mapping
    domain_to_category = create_domain_to_category_map()
    
    # Find all date folders (e.g., Jul-2-2025)
    date_folders = [d for d in outputs_path.iterdir() if d.is_dir()]
    
    for date_folder in date_folders:
        print(f"Processing date folder: {date_folder.name}")
        
        # Find all experiment type folders (zeroshot, fewshot, instronly)
        experiment_folders = [d for d in date_folder.iterdir() if d.is_dir()]
        
        for exp_folder in experiment_folders:
            mmlu_path = exp_folder / "MMLU-prejul8" / "mcqplain"
            
            if not mmlu_path.exists():
                print(f"Skipping {exp_folder.name}: MMLU-prejul8/mcqplain not found")
                continue
            
            print(f"  Processing experiment: {exp_folder.name}")
            
            # Create MMLU-copy directory
            mmlu_copy_path = exp_folder / "MMLU-copy" / "mcqplain"
            mmlu_copy_path.mkdir(parents=True, exist_ok=True)
            
            # Find all model family folders (Falcon, Gemma3, etc.)
            model_families = [d for d in mmlu_path.iterdir() if d.is_dir()]
            
            for model_family in model_families:
                print(f"    Processing model family: {model_family.name}")
                
                # Create model family directory in copy
                model_family_copy = mmlu_copy_path / model_family.name
                model_family_copy.mkdir(exist_ok=True)
                
                # Create category directories
                for category in DOMAIN_CATEGORIES.keys():
                    category_dir = model_family_copy / category
                    category_dir.mkdir(exist_ok=True)
                
                # Get domain directories
                domain_dirs = [d for d in model_family.iterdir() if d.is_dir()]
                
                if not domain_dirs:
                    print(f"      No domain directories found in {model_family.name}")
                    continue
                
                # Get CSV filenames from the first domain (they should all be the same)
                first_domain = domain_dirs[0]
                csv_filenames = get_csv_filenames_from_domain(first_domain)
                
                if not csv_filenames:
                    print(f"      No CSV files found in {first_domain.name}")
                    continue
                
                print(f"      Found CSV files: {csv_filenames}")
                
                # Initialize merged dataframes for each category and CSV file
                merged_data = {}
                for category in DOMAIN_CATEGORIES.keys():
                    merged_data[category] = {}
                    for csv_file in csv_filenames:
                        merged_data[category][csv_file] = []
                
                # Process each domain directory
                for domain_dir in domain_dirs:
                    domain_name = domain_dir.name
                    
                    if domain_name not in domain_to_category:
                        print(f"        Warning: Domain '{domain_name}' not found in category mapping, skipping")
                        continue
                    
                    category = domain_to_category[domain_name]
                    print(f"        Processing domain: {domain_name} -> {category}")
                    
                    # Process each CSV file in the domain
                    for csv_file in csv_filenames:
                        csv_path = domain_dir / csv_file
                        
                        if not csv_path.exists():
                            print(f"          Warning: {csv_file} not found in {domain_name}")
                            continue
                        
                        try:
                            # Read the CSV file
                            df = pd.read_csv(csv_path)
                            
                            # Add domain column
                            df['domain'] = domain_name
                            
                            # Add to merged data
                            merged_data[category][csv_file].append(df)
                            
                        except Exception as e:
                            print(f"          Error reading {csv_path}: {e}")
                
                # Write merged CSV files
                for category in DOMAIN_CATEGORIES.keys():
                    category_dir = model_family_copy / category
                    
                    for csv_file in csv_filenames:
                        if merged_data[category][csv_file]:
                            try:
                                # Concatenate all dataframes for this category and CSV file
                                merged_df = pd.concat(merged_data[category][csv_file], 
                                                    ignore_index=True)
                                
                                # Write to file
                                output_path = category_dir / csv_file
                                merged_df.to_csv(output_path, index=False)
                                
                                print(f"        Created: {output_path} with {len(merged_df)} rows")
                                
                            except Exception as e:
                                print(f"        Error writing {category}/{csv_file}: {e}")
                        else:
                            print(f"        No data for {category}/{csv_file}")

def main():
    """Main function to run the reorganization."""
    # Change this to your base path where 'outputs' folder is located
    base_path = "."  # Current directory, change as needed
    
    print("Starting MMLU results reorganization...")
    reorganize_mmlu_results(base_path)
    print("Reorganization complete!")

if __name__ == "__main__":
    main()