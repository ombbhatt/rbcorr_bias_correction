import pandas as pd
import numpy as np

# for every csv in the CURRENT folder (excluding subfolders), read it and create a new csv with only 1200 rows (randomly sampled) and and a save that version in the tiny_data folder
# make sure to sample such that class balance is preserved. For csvs that have 'ynq' in the title, the 'Correct Answer' column will have 'Yes' or 'No' which must be balanced. For csvs that have 'nli' in the title, the 'Correct Answer' column will have 0, 1 or 2 which must be balanced.

import os
import random
from collections import Counter
INPUT_FOLDER = "./"
OUTPUT_FOLDER = "./big_data/"

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

        # Sample 1200 rows while preserving class balance
        sampled_df = df.groupby(class_col).apply(lambda x: x.sample(1200 // len(classes), random_state=42)).reset_index(drop=True)

        # Print class distribution in the sampled dataframe
        print(f"{filename} class distribution in sampled data:")
        print(sampled_df[class_col].value_counts().sort_index())

        # Save the sampled dataframe to the output folder
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        sampled_df.to_csv(output_path, index=False)
        print(f"Processed {filename} and saved to {output_path}.")

# Now go through the 'mmlu-scripts-data' folder. This contains four subfolders, and each has a certain number of csv files. For each subfolder, we want to create a new subfolder in the big_data folder with the same name, and across all csv files in that subfolder, we want to sample 1200 rows in total for that subfolder while preserving class balance. The class balance is determined by the 'answer' column, which has either 'A', 'B', 'C', or 'D' as the value and must be class balanced across the sampled 1200 rows. Make sure that atleast 10 rows are sampled from each csv file for balanced representation of the different csv files in the subfolder.

MMLU_INPUT_FOLDER = "./mmlu-scripts-data/"
MMLU_OUTPUT_FOLDER = "./big_data/mmlu-scripts-data/"

for subfolder in os.listdir(MMLU_INPUT_FOLDER):
    subfolder_path = os.path.join(MMLU_INPUT_FOLDER, subfolder)
    if os.path.isdir(subfolder_path):
        all_dfs = []
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(subfolder_path, filename))
                df['source_file'] = filename  # tag each row with its origin
                all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Check if the required classes are present
        if not all(cls in combined_df['answer'].unique() for cls in ['A', 'B', 'C', 'D']):
            print(f"Skipping {subfolder}: not all required classes are present.")
            continue

        n_files = combined_df['source_file'].nunique()
        per_file_per_class = 300 // n_files  # base quota per (file, answer) cell

        final_sample = (
            combined_df
            .groupby(['source_file', 'answer'], group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), per_file_per_class), random_state=42))
        )

        # If some files were short, top up each class to exactly 300
        # by sampling the remainder from whichever groups have rows left
        already_sampled = final_sample.index
        remainder = combined_df.drop(index=already_sampled)

        shortfall_per_class = 300 - final_sample.groupby('answer').size()
        topup = (
            remainder
            .groupby('answer', group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), shortfall_per_class[x.name]), random_state=42))
        )

        sampled_df = pd.concat([final_sample, topup]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Print class distribution in the sampled dataframe
        print(f"{subfolder} class distribution in sampled data:")
        print(sampled_df['answer'].value_counts().sort_index())

        # print distribution of source files in the sampled dataframe
        print(f"{subfolder} source file distribution in sampled data:")
        print(sampled_df['source_file'].value_counts().sort_index())

        # Save the sampled dataframe to the output folder
        output_subfolder = os.path.join(MMLU_OUTPUT_FOLDER, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        output_path = os.path.join(output_subfolder, f"{subfolder}_sampled.csv")
        sampled_df.to_csv(output_path, index=False)
        print(f"Processed {subfolder} and saved to {output_path}.")