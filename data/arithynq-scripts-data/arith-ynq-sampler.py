import pandas as pd
import random
import json
import os

random.seed(42)

df = pd.read_csv('arith-ynq.csv')

small_df = df.copy()

# keep 25 rows each of with 'Yes' and 'No' from all rows that have '1_digit_' in the 'Digits' column
small_df1 = small_df[small_df['Digits'].str.contains('1_digit_addition')].groupby('Correct Answer').apply(lambda x: x.sample(100)).reset_index(drop=True)

small_df2 = small_df[small_df['Digits'].str.contains('1_digit_subtraction')].groupby('Correct Answer').apply(lambda x: x.sample(100)).reset_index(drop=True)

small_df3 = small_df[small_df['Digits'].str.contains('1_digit_multiplication')].groupby('Correct Answer').apply(lambda x: x.sample(100)).reset_index(drop=True)

small_df4 = small_df[small_df['Digits'].str.contains('2_digit_addition')].groupby('Correct Answer').apply(lambda x: x.sample(100)).reset_index(drop=True)

small_df5 = small_df[small_df['Digits'].str.contains('2_digit_subtraction')].groupby('Correct Answer').apply(lambda x: x.sample(100)).reset_index(drop=True)

small_df6 = small_df[small_df['Digits'].str.contains('2_digit_multiplication')].groupby('Correct Answer').apply(lambda x: x.sample(100)).reset_index(drop=True)

# combine the two dataframes, keeping the original order
small_df7 = pd.concat([small_df1, small_df2, small_df3, small_df4, small_df5, small_df6], ignore_index=True)

small_df7.to_csv('arith-ynq-big.csv', index=False)
print(small_df7.head())



