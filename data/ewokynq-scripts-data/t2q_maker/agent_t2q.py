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
            questions += [
                    (context1, f"Does {(' '.join([w[:-1] if i == 1 else w for i, w in enumerate(target1.split())])).rstrip('.')}?", "Yes"),
                    (context2, f"Does {(' '.join([w[:-1] if i == 1 else w for i, w in enumerate(target1.split())])).rstrip('.')}?", "No"),
            ]
            if "does" in target2.split():
                questions += [
                    (context1, f"Does {target2.replace(' does ', ' ').rstrip('.')}?", "No"),
                    (context2, f"Does {target2.replace(' does ', ' ').rstrip('.')}?", "Yes")
                ]
            else:
                # find the second word in target1 and target2:
                questions += [
                    (context1, f"Does {(' '.join([w[:-1] if i == 1 else w for i, w in enumerate(target2.split())])).rstrip('.')}?", "No"),
                    (context2, f"Does {(' '.join([w[:-1] if i == 1 else w for i, w in enumerate(target2.split())])).rstrip('.')}?", "Yes")
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

    input_file3 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-agent_properties.csv')
    output_file3 = os.path.join(script_dir, 't2q_agent_properties.csv')

    generate_questions(input_file3, output_file3) 
    print(f"Questions have been generated and saved to {output_file3}")

if __name__ == "__main__":
    main()