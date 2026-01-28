import csv
import os
import re

def generate_questions(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        next(infile)
        reader = csv.DictReader(infile)
        fieldnames = ['Question Number', 'MetaTemplateID', 'Domain', 'ItemGroupID', 'ContextDiff', 'ContextNum', 'Context', 'Question', 'Correct Answer']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        question_number = 1
        item_group_id = 1

        for row in reader:
            domain = row['Domain']
            template_id = row['MetaTemplateID']
            context_diff = row['ContextDiff']
            context1 = row['Context1']
            context2 = row['Context2']
            target1 = row['Target1']
            target2 = row['Target2']

            index_s1 = target1.rfind('s')
            target1 = target1[:index_s1] + target1[index_s1+1:]

            index_s2 = target2.rfind('s')
            target2 = target2[:index_s2] + target2[index_s2+1:]

            questions = []
            # Generate four questions for each row
            if target1.split()[0] == "It":
                questions += [
                    (context1, f"Can {target1.lower().rstrip('.')}?", "Yes"),
                    (context1, f"Can {target2.lower().rstrip('.')}?", "No"),
                    (context2, f"Can {target1.lower().rstrip('.')}?", "No"),
                    (context2, f"Can {target2.lower().rstrip('.')}?", "Yes")
                ]
            else:
                questions += [
                    (context1, f"Can {target1.rstrip('.')}?", "Yes"),
                    (context1, f"Can {target2.rstrip('.')}?", "No"),
                    (context2, f"Can {target1.rstrip('.')}?", "No"),
                    (context2, f"Can {target2.rstrip('.')}?", "Yes")
                ]

            for context, question, answer in questions:
                writer.writerow({
                    'Question Number': question_number,
                    'MetaTemplateID': template_id,
                    'Domain': domain,
                    'ItemGroupID': item_group_id,
                    'ContextDiff': context_diff,
                    'ContextNum': 1 if context == context1 else 2,
                    'Context': context,
                    'Question': question,
                    'Correct Answer': answer
                })
                question_number += 1
                
            item_group_id += 1

def main():
    father_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_file2 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-material_dynamics.csv')
    output_file2 = os.path.join(script_dir, 't2q_material_dynamics.csv')

    generate_questions(input_file2, output_file2)
    print(f"Questions have been generated and saved to {output_file2}")

if __name__ == "__main__":
    main()