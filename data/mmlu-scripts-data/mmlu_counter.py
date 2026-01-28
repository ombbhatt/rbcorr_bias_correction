import os
import csv
import glob

def count_questions_in_csvs():
    # Get a list of all CSV files in current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print(f"No CSV files found")
        return
    
    total_questions = 0
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Process each CSV file
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)                
                # Skip header row
                next(csv_reader, None)
                # Count remaining rows
                row_count = sum(1 for row in csv_reader)
                total_questions += row_count
                
                print(f"{file_name}: {row_count} questions")
                
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    print(f"\nTotal questions across all files: {total_questions}")

if __name__ == "__main__":
    # folder_path = "mmlu-scripts-data"
    count_questions_in_csvs()