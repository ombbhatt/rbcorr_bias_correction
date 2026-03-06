import csv
import os

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

            questions = []
            # Generate four questions for each row
            if "is" in target1.split():
                questions += [
                    (context1, f"Is {target1.replace(' is ', ' ').rstrip('.')}?", "Yes"),
                    (context1, f"Is {target2.replace(' is ', ' ').rstrip('.')}?", "No"),
                    (context2, f"Is {target1.replace(' is ', ' ').rstrip('.')}?", "No"),
                    (context2, f"Is {target2.replace(' is ', ' ').rstrip('.')}?", "Yes")
                ]
            elif "are" in target1.split():
                questions += [
                    (context1, f"Are {target1.replace(' are ', ' ').rstrip('.')}?", "Yes"),
                    (context1, f"Are {target2.replace(' are ', ' ').rstrip('.')}?", "No"),
                    (context2, f"Are {target1.replace(' are ', ' ').rstrip('.')}?", "No"),
                    (context2, f"Are {target2.replace(' are ', ' ').rstrip('.')}?", "Yes")
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

    input_file8 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-social_interactions.csv')
    output_file8 = os.path.join(script_dir, 't2q_social_interactions.csv')

    input_file9 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-social_properties.csv')
    output_file9 = os.path.join(script_dir, 't2q_social_properties.csv')

    input_file10 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-social_relations.csv')
    output_file10 = os.path.join(script_dir, 't2q_social_relations.csv')

    in_out_dict = {input_file8: output_file8, input_file9: output_file9, input_file10: output_file10}

    for input_file, output_file in in_out_dict.items():
        generate_questions(input_file, output_file)
        print(f"Questions have been generated and saved to {output_file}")

if __name__ == "__main__":
    main()