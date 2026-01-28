import pandas as pd
import random
import json
import os

random.seed(42)

# relevant folders to go through in 'arithmetic' directory: 

folders2conv = ['1_digit_addition', '1_digit_subtraction', '1_digit_multiplication', '2_digit_addition', '2_digit_subtraction', '2_digit_multiplication']

arith_ynq_df = pd.DataFrame(columns=['Type', 'Digits', 'Context', 'Question', 'Correct Answer'])

arith_ynq_LIST = []

for folder in folders2conv:
    
    print(f'folder: {folder}')

    # go through files
    for file in os.listdir(f'benchmark_tasks/arithmetic/{folder}'):

        if file.endswith('.json'):

            print(f'file: {file}')

            # read json file into a dataframe
            with open(f'benchmark_tasks/arithmetic/{folder}/{file}') as f:

                data = json.load(f)

                for ex in data['examples']:
                    q = ex['input']
                    ya = ex['target']
                    # na is equal to the value of the first key in the dict in ex['target_scores']:
                    na = list(ex['target_scores'].keys())[0]

                    # change q to yes-no format: "What is 0 plus 0?" + "0" -> "Is 0 plus 0 equal to 0?" + "Yes"
                    yq = q.replace('What is', 'Is').replace('?', ' equal to ') + ya + '?'
                    nq = q.replace('What is', 'Is').replace('?', ' equal to ') + na + '?'

                    # add to dataframe
                    arith_ynq_LIST.append(['Arithmetic', folder, "''", yq, 'Yes'])
                    arith_ynq_LIST.append(['Arithmetic', folder, "''", nq, 'No'])

arith_ynq_df = pd.DataFrame(arith_ynq_LIST, columns=['Type', 'Digits', 'Context', 'Question', 'Correct Answer'])

# save to csv
arith_ynq_df.to_csv('arith-ynq.csv', index=False)
print(arith_ynq_df.head())
print('Done!')
