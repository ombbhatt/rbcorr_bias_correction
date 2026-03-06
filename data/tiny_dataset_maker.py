import pandas as pd
import numpy as np

# for every csv in the CURRENT folder (excluding subfolders), read it and create a new csv with only 200 rows (randomly sampled) and and a save that version in the tiny_data folder
# make sure to sample such that class balance is preserved. For csvs that have 'ynq' in the title, the 'Correct Answer' column will have 'Yes' or 'No' which must be balanced. For csvs that have 'nli' in the title, the 'Correct Answer' column will have 0, 1 or 2 which must be balanced.

import os
import random
from collections import Counter
INPUT_FOLDER = "./"
OUTPUT_FOLDER = "./tiny_data/"

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv") and not os.path.isdir(os.path.join(INPUT_FOLDER, filename)):
        df = pd.read_csv(os.path.join(INPUT_FOLDER, filename))

        if 'ynq' in filename:
            class_col = 'Correct Answer'
            classes = ['Yes', 'No']
        elif 'nli' in filename:
            class_col = 'Correct Answer'
            classes = [0, 1, 2]
        else:
            print(f"Skipping {filename}: unknown dataset type.")
            continue

        # Check if the required classes are present
        if not all(cls in df[class_col].unique() for cls in classes):
            print(f"Skipping {filename}: not all required classes are present.")
            continue

        # Sample 200 rows while preserving class balance
        sampled_df = df.groupby(class_col).apply(lambda x: x.sample(200 // len(classes), random_state=42)).reset_index(drop=True)

        # Save the sampled dataframe to the output folder
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        sampled_df.to_csv(output_path, index=False)
        print(f"Processed {filename} and saved to {output_path}.")

# Now go through the 'mmlu-scripts-data' folder. This contains four subfolders, and each has a certain number of csv files. For each subfolder, we want to create a new subfolder in the tiny_data folder with the same name, and across all csv files in that subfolder, we want to sample 200 rows in total for that subfolder while preserving class balance. The class balance is determined by the 'answer' column, which has either 'A', 'B', 'C', or 'D' as the value and must be class balanced across the sampled 200 rows. Make sure that atleast 10 rows are sampled from each csv file for balanced representation of the different csv files in the subfolder.

MMLU_INPUT_FOLDER = "./mmlu-scripts-data/"
MMLU_OUTPUT_FOLDER = "./tiny_data/mmlu-scripts-data/"

for subfolder in os.listdir(MMLU_INPUT_FOLDER):
    subfolder_path = os.path.join(MMLU_INPUT_FOLDER, subfolder)
    if os.path.isdir(subfolder_path):
        all_dfs = []
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(subfolder_path, filename))
                all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Check if the required classes are present
        if not all(cls in combined_df['answer'].unique() for cls in ['A', 'B', 'C', 'D']):
            print(f"Skipping {subfolder}: not all required classes are present.")
            continue

        # Sample 200 rows while preserving class balance and ensuring at least 10 rows from each csv file
        sampled_df = combined_df.groupby('answer').apply(lambda x: x.sample(200 // 4, random_state=42)).reset_index(drop=True)
        sampled_df = sampled_df.groupby(sampled_df.index // 10).apply(lambda x: x.sample(10, random_state=42)).reset_index(drop=True)

        # Save the sampled dataframe to the output folder
        output_subfolder = os.path.join(MMLU_OUTPUT_FOLDER, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        output_path = os.path.join(output_subfolder, f"{subfolder}_sampled.csv")
        sampled_df.to_csv(output_path, index=False)
        print(f"Processed {subfolder} and saved to {output_path}.")