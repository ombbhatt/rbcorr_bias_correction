import pandas as pd
import random
import json
import os

random.seed(42)

# txt file dataset file format (excerpt):
# 1 John travelled to the hallway.
# 2 Mary journeyed to the bathroom.
# 3 Where is John? 	hallway	1

# relevant files to go through in 'en' directory: 

files2conv = ['qa1_single-supporting-fact_test.txt',
            'qa2_two-supporting-facts_test.txt',
            'qa4_two-arg-relations_test.txt',
            'qa6_yes-no-questions_test.txt',
            'qa9_simple-negation_test.txt',
            'qa10_indefinite-knowledge_test.txt',
            'qa11_basic-coreference_test.txt',
            'qa12_conjunction_test.txt',
            'qa13_compound-coreference_test.txt',
            'qa15_basic-deduction_test.txt',
            'qa17_positional-reasoning_test.txt',
            'qa18_size-reasoning_test.txt'
]

babi_ynq_df = pd.DataFrame(columns=['task', 'QID', 'AnsRefIDs', 'Context', 'Question', 'Correct Answer'])
babi_ynq_dict_of_lists = {}

for file in files2conv:

    if file.endswith('.txt'):

        print(f'file: {file}')

        # read line by line
        with open(f'en/{file}') as f:

            premises = {}
            qna = []
            lines = f.readlines()

            task_id = (file.split('_', 1))[0]

            # go through lines
            for line in lines:
                # if line ends wth a period, it is a premise
                if line.endswith('.\n'):
                    # split line into ID and premise
                    pid, premise = line.split(' ', 1)

                    if int(pid) == 1:
                        premises = {}
                        qna = []

                    # add to premises dictionary
                    premises[int(pid)] = premise
                # else it is a qna line
                else:
                    qaid, q_a_ref = line.split(' ', 1)
                    # in q_a_ref, q is up until and including '?', a_ref is the rest
                    q, a_ref = q_a_ref.split('?', 1)
                    q = q + '?'

                    # print(f'q: {q}, a_ref: {a_ref}')
                    a_ref = a_ref.strip()
                    # print(f'a_ref: {a_ref}')
                    a, refs = a_ref.split('\t', 1)
                    a = a.strip()
                    refs = refs.strip()
                    # print(f'a: {a}, refs: {refs}')
                    # refs is either only one number or two numbers separated by a space
                    if ' ' in refs:
                        refs = refs.split(' ')
                    else:
                        refs = [refs]

                    # convert refs to int
                    refs = [int(ref) for ref in refs]
                    
                    # if int value of refs is equal to any of the keys in premises, it is valid
                    if all(ref in premises.keys() for ref in refs):
                        # get the specified premise values for the matching refs
                        premises_values = [premises[int(ref)] for ref in refs]
                        # remove newline characters from premises values
                        premises_values = [premises_value.replace('\n', '').replace('\r', '') for premises_value in premises_values]

                        yq, nq, ans, w_ans = '', '', '', ''

                        # convert q+a to yes-no question: 
                        if task_id in ['qa1', 'qa2', 'qa11', 'qa12', 'qa13']:

                            locations = ['bathroom', 'hallway', 'kitchen', 'office', 'garden', 'bedroom']
                            if a in locations:
                                wrong_a = random.choice([loc for loc in locations if loc != a])

                            if "Where is" in q: # example: "Where is John?" + "hallway" -> "Is John in the hallway?" + "Yes"
                                yq = q.replace('Where is', 'Is').replace('?','') + ' in the ' + a + '?'
                                ans = 'Yes'

                                nq = q.replace('Where is', 'Is').replace('?','') + ' in the ' + wrong_a + '?'
                                w_ans = 'No'

                        if task_id in ['qa4', 'qa15']:

                            locations = ['bathroom', 'hallway', 'kitchen', 'office', 'garden', 'bedroom']
                            if a in locations:
                                wrong_a = random.choice([loc for loc in locations if loc != a])

                            animals = ['sheep', 'mouse', 'cat', 'wolf']
                            if a in animals:
                                wrong_a = random.choice([animal for animal in animals if animal != a])

                            if "What is" in q and q.split()[-1].rstrip("?") != "of": # example: "What is north of the office?" + "garden" -> "Is the garden north of the office?" + "Yes"
                                yq = "Is the " + a + q.replace('What is', '') 
                                ans = 'Yes'

                                nq = "Is the " + wrong_a + q.replace('What is', '')
                                w_ans = 'No'

                            elif "What is" in q and q.split()[-1].rstrip("?") == "of": #
                                if task_id == 'qa4':  # example: "What is the office north of?" + "garden" -> "Is the office north of the garden?" + "Yes"
                                    yq = q.replace('What is', 'Is').replace('?', '') + ' the ' + a + '?'
                                    ans = 'Yes'

                                    nq = q.replace('What is', 'Is').replace('?', '') + ' the ' + wrong_a + '?'
                                    w_ans = 'No'

                                elif task_id == 'qa15': # example: "What is emily afraid of?" + "mouse" -> "Is emily afraid of a mouse?" + "Yes"
                                    yq = q.replace('What is', 'Is').replace('?', '') + ' a ' + a + '?'
                                    ans = 'Yes'

                                    nq = q.replace('What is', 'Is').replace('?', '') + ' a ' + wrong_a + '?'
                                    w_ans = 'No'


                        if task_id in ['qa6', 'qa9', 'qa10', 'qa17', 'qa18']:
                            if a == 'yes':
                                yq = q
                                ans = 'Yes'

                            elif a == 'no':
                                nq = q
                                w_ans = 'No'

                        # append to list which is value for the key task_id. if list does not exist, create it
                        if task_id not in babi_ynq_dict_of_lists:
                            babi_ynq_dict_of_lists[task_id] = []

                        if yq:
                            df_row_y = {'task': task_id, 'QID': qaid, 'AnsRefIDs': refs, 'Context': ' '.join(premises_values), 'Question': yq, 'Correct Answer': ans}
                            babi_ynq_dict_of_lists[task_id].append(df_row_y)
                        if nq:
                            df_row_n = {'task': task_id, 'QID': qaid, 'AnsRefIDs': refs, 'Context': ' '.join(premises_values), 'Question': nq, 'Correct Answer': w_ans}
                            babi_ynq_dict_of_lists[task_id].append(df_row_n)

# create different dataframes for each task
for task, df_list in babi_ynq_dict_of_lists.items():
    babi_ynq_df = pd.DataFrame(df_list)
    print(babi_ynq_df.head())

    df = babi_ynq_df.copy()

    yes_count = sum(babi_ynq_df['Correct Answer'] == 'Yes')
    no_count = sum(babi_ynq_df['Correct Answer'] == 'No')

    # Determine which to downsample and the target size
    if yes_count >= no_count:
        to_sample = df[df['Correct Answer'] == 'Yes'].sample(no_count, random_state=42)
        to_keep = df[df['Correct Answer'] == 'No']
    elif no_count > yes_count:
        to_sample = df[df['Correct Answer'] == 'No'].sample(yes_count, random_state=42)
        to_keep = df[df['Correct Answer'] == 'Yes']

    # Combine the dataframes by alternating the rows of the two dataframes
    balanced_df = pd.concat([to_sample, to_keep]).sort_index()

    balanced_df.to_csv(f'{task}-babi-ynq-bal.csv', index=False)
                    
                        