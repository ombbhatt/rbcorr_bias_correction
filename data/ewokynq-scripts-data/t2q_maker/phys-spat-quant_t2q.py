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

        for row in reader:
            domain = row['Domain']
            template_id = row['MetaTemplateID']
            context_diff = row['ContextDiff']
            context1 = row['Context1']
            context2 = row['Context2']
            target1 = row['Target1']
            target2 = row['Target2']

            questions = []

            # lower case the word "The" or "There" if it is the first word in target1 or target2
            if target1.split()[0] == "The" or target1.split()[0] == "There":
                target1 = target1.replace("The", "the").replace("There", "there")
            if target2.split()[0] == "The" or target2.split()[0] == "There":
                target2 = target2.replace("The", "the").replace("There", "there")
            

            # if "Why?" is in target1 or target2, delete it
            if "Why?" in target1:
                target1 = target1.replace("Why?", "").strip()
            if "Why?" in target2:
                target2 = target2.replace("Why?", "").strip()

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
            else:
                index_s1 = target1.rfind('s')
                target1 = target1[:index_s1] + target1[index_s1+1:]

                index_s2 = target2.rfind('s')
                target2 = target2[:index_s2] + target2[index_s2+1:]

                # convert any occurence of "has" to "have" in target1 and target2
                if "has" in target1.split():
                    target1 = target1.replace("has", "have")
                if "has" in target2.split():
                    target2 = target2.replace("has", "have")

                questions += [
                    (context1, f"Does {target1.rstrip('.')}?", "Yes"),
                    (context1, f"Does {target2.rstrip('.')}?", "No"),
                    (context2, f"Does {target1.rstrip('.')}?", "No"),
                    (context2, f"Does {target2.rstrip('.')}?", "Yes")
                ]


            for context, question, answer in questions:
                writer.writerow({
                    'Question Number': question_number,
                    'MetaTemplateID': template_id,
                    'Domain': domain,
                    'ContextDiff': context_diff,
                    'ContextNum': 1 if context == context1 else 2,
                    'Context': context,
                    'Question': question,
                    'Correct Answer': answer
                })
                question_number += 1

def main():

    father_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_file1 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-material_properties.csv')
    output_file1 = os.path.join(script_dir, 't2q_material_properties.csv')

    input_file4 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-physical_dynamics.csv')
    output_file4 = os.path.join(script_dir, 't2q_physical_dynamics.csv')

    input_file5 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-physical_interactions.csv')
    output_file5 = os.path.join(script_dir, 't2q_physical_interactions.csv')

    input_file6 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-physical_relations.csv')
    output_file6 = os.path.join(script_dir, 't2q_physical_relations.csv')

    input_file7 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-quantitative_properties.csv')
    output_file7 = os.path.join(script_dir, 't2q_quantitative_properties.csv')

    input_file11 = os.path.join(father_dir, 'ewok-core-1.0/dataset-cfg=2bb3c7512e737b00__fix=True__n=1__vers=0/testsuite-spatial_relations.csv')
    output_file11 = os.path.join(script_dir, 't2q_spatial_relations.csv')

    in_out_dict = {input_file1: output_file1, input_file4: output_file4, input_file5: output_file5, input_file6: output_file6, input_file7: output_file7, input_file11: output_file11}

    for input_file, output_file in in_out_dict.items():
        generate_questions(input_file, output_file) 
        print(f"Questions have been generated and saved to {output_file}")

if __name__ == "__main__":
    main()