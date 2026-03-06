import pandas as pd
import random
import json
import os

random.seed(42)

# go through all csv files that end with 'ynq-bal' in current directory
files2sample = [f for f in os.listdir('.') if f.endswith('ynq-bal.csv')]

cumulative_df = pd.DataFrame(columns=['task', 'QID', 'AnsRefIDs', 'Context', 'Question', 'Correct Answer'])

for file in files2sample:

    print(f'file: {file}')

    # read csv file
    df = pd.read_csv(file)

    # randomly sample 15 rows which have "Yes" as the correct answer
    df_yes = df[df['Correct Answer'] == 'Yes']
    df_yes_sample = df_yes.sample(n=50, random_state=42)

    # randomly sample 15 rows which have "No" as the correct answer
    df_no = df[df['Correct Answer'] == 'No']
    df_no_sample = df_no.sample(n=50, random_state=42)

    # combine the two samples
    df_sample = pd.concat([df_yes_sample, df_no_sample]).sort_index()

    # append to cumulative dataframe
    cumulative_df = pd.concat([cumulative_df, df_sample])

# save the cumulative dataframe to a csv file
cumulative_df.to_csv('babi-ynq-big.csv', index=False)
