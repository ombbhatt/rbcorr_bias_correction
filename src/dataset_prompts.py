# this is dataset_prompts.py:

def get_dataset_prompts(dataset, content_free_mode=False, content_free_input="N/A"):    
    """
    Get dataset-specific prompts with optional content-free mode for contextual calibration
    
    Args:
        dataset: Dataset name
        content_free_mode: If True, returns prompts with content-free input
        content_free_input: The content-free string to use (default: "N/A")
    """

    print(f"Received dataset value in get_dataset_prompts(): {dataset}")
    
    if dataset == "COMPS":
        if content_free_mode:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Does a blueberry fire bullets?\nResponse: No\n\n#EXAMPLE\nQuestion: Does a turtle have a hard shell?\nResponse: Yes\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
        else:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Does a blueberry fire bullets?\nResponse: No\n\n#EXAMPLE\nQuestion: Does a turtle have a hard shell?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "EWOK":
        if content_free_mode:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Claire sees something that is fabric. Can Claire pour it?\nResponse: No\n\n#EXAMPLE\nQuestion: Sally pays salary to Harry. Is Sally Harry's boss?\nResponse: Yes\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
        else:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Claire sees something that is fabric. Can Claire pour it?\nResponse: No\n\n#EXAMPLE\nQuestion: Sally pays salary to Harry. Is Sally Harry's boss?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "BABI":
        if content_free_mode:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Marshall is in the car. Is Marshall in the building?\nResponse: No\n\n#EXAMPLE\nQuestion: Nathan is a pianist. Pianists like oranges. Does Nathan like oranges?\nResponse: Yes\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
        else:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Marshall is in the car. Is Marshall in the building?\nResponse: No\n\n#EXAMPLE\nQuestion: Nathan is a pianist. Pianists like oranges. Does Nathan like oranges?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "ARITH":
        if content_free_mode:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Is 7 minus 9 equal to 4?\nResponse: No\n\n#EXAMPLE\nQuestion: Is 17 plus 15 equal to 32?\nResponse: Yes\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: {content_free_input}\nResponse: "
        else:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following yes-no questions:\n\n#EXAMPLE\nQuestion: Is 7 minus 9 equal to 4?\nResponse: No\n\n#EXAMPLE\nQuestion: Is 17 plus 15 equal to 32?\nResponse: Yes\n\n#EXAMPLE\nQuestion: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following yes-no question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "MMLU":
        if content_free_mode:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following multiple choice questions:\n\n#EXAMPLE\nQuestion: What is the shape of the Earth?\nOptions: (A) Cone, (B) Cube, (C) Sphere, (D) Cylinder\nResponse: C\n\n#EXAMPLE\nQuestion: What is the color of the sky?\nOptions: (A) Red, (B) Blue, (C) Green, (D) Yellow\nResponse: B\n\n#EXAMPLE\nQuestion: {content_free_input}\nOptions: (A) {content_free_input}, (B) {content_free_input}, (C) {content_free_input}, (D) {content_free_input}\nResponse: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following multiple choice question:\n\n#EXAMPLE\nQuestion: {content_free_input}\nOptions: (A) {content_free_input}, (B) {content_free_input}, (C) {content_free_input}, (D) {content_free_input}\nResponse: "
        else:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following multiple choice questions:\n\n#EXAMPLE\nQuestion: What is the shape of the Earth?\nOptions: (A) Cone, (B) Cube, (C) Sphere, (D) Cylinder\nResponse: C\n\n#EXAMPLE\nQuestion: What is the color of the sky?\nOptions: (A) Red, (B) Blue, (C) Green, (D) Yellow\nResponse: B\n\n#EXAMPLE\nQuestion: "
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following multiple choice question:\n\n#EXAMPLE\nQuestion: "

    elif dataset == "SNLI" or dataset == "MNLI":
        if content_free_mode:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following Recognizing Textual Entailment questions using a single digit. Entailment (0) implies the hypothesis is true given the premise. Neutral (1) implies the premise doesn't provide enough information to determine the hypothesis. Contradiction (2) implies the hypothesis is false given the premise.\n\n#EXAMPLE\:\nPremise: A man is playing a guitar. Hypothesis: A person is making music.\nResponse: 0\n\n#EXAMPLE\:\nPremise: A woman is reading a book in the library. Hypothesis: A woman is swimming. \nResponse: 2\n\n#EXAMPLE\:\nPremise: {content_free_input} Hypothesis: {content_free_input}"
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following Recognizing Textual Entailment question using a single digit. Entailment (0) implies the hypothesis is true given the premise. Neutral (1) implies the premise doesn't provide enough information to determine the hypothesis. Contradiction (2) implies the hypothesis is false given the premise.\n\n#EXAMPLE\:\nPremise: {content_free_input} Hypothesis: {content_free_input}"
        else:
            prompt_cond1 = f"#INSTRUCTIONS\nAnswer the following Recognizing Textual Entailment questions using a single digit. Entailment (0) implies the hypothesis is true given the premise. Neutral (1) implies the premise doesn't provide enough information to determine the hypothesis. Contradiction (2) implies the hypothesis is false given the premise.\n\n#EXAMPLE\:\nPremise: A man is playing a guitar. Hypothesis: A person is making music.\nResponse: 0\n\n#EXAMPLE\:\nPremise: A woman is reading a book in the library. Hypothesis: A woman is swimming. \nResponse: 2\n\n#EXAMPLE\:\n"
            prompt_cond2 = f"#INSTRUCTIONS\nAnswer the following Recognizing Textual Entailment question using a single digit. Entailment (0) implies the hypothesis is true given the premise. Neutral (1) implies the premise doesn't provide enough information to determine the hypothesis. Contradiction (2) implies the hypothesis is false given the premise.\n\n#EXAMPLE\:\n"

    return prompt_cond1, prompt_cond2