import random
import re
import num2words
from collections import namedtuple
import os
import uuid

TEMPLATES = {
    "grader_last": {
        "template": "{summand1} + {summand2} = {sum}. {character}:",
        "choices": [" False", " True"],
    },
    "grader_first": {
        "template": "Grader: {character}\n\n{summand1} + {summand2} = {sum}\n\nScore:",
        "choices": [" False", " True"],
    },
    "0": {
        "template": "{summand1} + {summand2} = {sum}, according to {character}, is",
        "choices": [" false", " true"],
    },
    "1": {
        "template": "According to {character}, {summand1} + {summand2} = {sum} is",
        "choices": [" incorrect", " correct"],
    },
    "2": {
        "template": "According to {character}, {summand1} + {summand2} = {sum} is",
        "choices": [" false", " true"],
    },
    "3": {
        "template": "accordin' to {character}, {summand1} + {summand2} = {sum} ain't",
        "choices": [" right", " wrong"],
    },
    "4": {
        "template": 'Teacher: "Is {summand1} + {summand2} = {sum}?"\n\n{character}:',
        "choices": [" No", " Yes"],
    },
    "5": {
        "template": "Does {summand1} + {summand2} = {sum}? According to {character},",
        "choices": [" No", " Yes"],
    },
    "6": {
        "template": "Exam 1\nName: {character}\nPart 1: True or False\n\n1. {summand1} + {summand2} = {sum}\nAnswer:",
        "choices": [" False", " True"],
    },
    "7": {
        "template": 'Teacher: "{character}. Does {summand1_words} plus {summand2_words} equal {sum_words}?"\n\n{character}:',
        "choices": [" No", " Yes"],
    },
    "8": {
        "template": "According to {character}, {summand1_words} plus {summand2_words} equals {sum_words}. This is",
        "choices": [" false", " true"],
    },
    "9": {
        "template": "{summand1_words} plus {summand2_words} equals {sum_words}. {character}:",
        "choices": [" False", " True"],
    },
    "10": {
        "template": "Grader: {character}\n\n{summand1} + {summand2} != {sum}\n\nScore:",
        "choices": [" True", " False"],
    },
}


def perturbation(text, character=None, p=1.0):
    # only perturb with probability p
    if random.random() > p:
        return text

    text = perturb_equation(text)
    if character is not None:
        text = perturb_character(text, character)
    if random.random() < 0.3:
        text = text.replace(".", "?")
    if random.random() < 0.5 and text[-1] in {".", "?", "\n"}:
        text = text + "\n"
        if random.random() < 0.5:  # potentially add a second newline
            text = text + "\n"
    if random.random() < 0.3 and "\n" in text:
        text = " ".join(text.split())
    if random.random() < 0.5 and text[-1] == "\n":
        text = text.rstrip() + " "
    return text


def perturb_equation(text):
    eq_pattern = r"\d+ \+ \d+ = \d+"
    matches = re.findall(eq_pattern, text)
    if len(matches) != 1:
        return text

    eq = matches[0]
    lhs, sum = eq.split(" = ")
    summand1, summand2 = lhs.split(" + ")
    rand = random.random()
    if rand < 0.5:
        return text.replace(eq, f"{summand1} + {summand2} = {sum}")
    elif rand < 0.7:
        return text.replace(eq, f"{sum} = {summand1} + {summand2}")
    elif rand < 0.9:
        return text.replace(eq, f"{summand1}+{summand2}={sum}")
    else:
        return text.replace(eq, f"{sum}={summand1}+{summand2}")
    
def perturb_character(text, character):
    rand = random.random()
    if rand < 0.1:
        return text.replace(character, character.lower())
    elif rand < 0.15:
        return text.replace(character, character.upper())
    return text
        

def make_jinja(template_name, label_col, ccs, elk_template_dir):
    """Creates a jinja template for a given template name and saves it to the elk template directory."""
    parent_dir = os.path.join(elk_template_dir, f"qm_{template_name}")
    path = os.path.join(parent_dir, "templates.yaml")
    os.makedirs(parent_dir, exist_ok=True)
    
    if template_name == "mixture":
        templates = TEMPLATES
    else:
        templates = {template_name: TEMPLATES[template_name]}

    template_str = "dataset: None\n" \
        "templates:\n"
    for key, value in templates.items():
        id = uuid.uuid4().hex
        python_template = value["template"]
        neg_tok, pos_tok = value["choices"]
        jinja_template = python_template.replace("{", "{{ ").replace("}", " }}")
        jinja_template = jinja_template.replace("'", "''")  # escape single quotes
        jinja_template = jinja_template + " ||| {{answer_choices[label]}}" if ccs else jinja_template
        template_str += f"  {id}: !Template\n" \
            f"    answer_choices: {neg_tok} ||| {pos_tok}\n" \
            f"    id: {id}\n" \
            f"    jinja: '{jinja_template}'\n"\
            f"    metadata: !TemplateMetadata\n" \
            f"      languages:\n" \
            f"      - en\n" \
            f"      metrics:\n" \
            f"      - Accuracy\n" \
            f"    name: \"{key}\"\n" \
            + (f"    label_column: \"{label_col}\"\n" if label_col is not None else "")

    with open(path, "w") as f:
        f.write(template_str)

        
def templatize_example(
    summand1, summand2, sum, character, template, perturb: float | bool = False
) -> tuple:
    # example has a question, statement, object, and label

    summand1_words = num2words.num2words(summand1)
    summand2_words = num2words.num2words(summand2)
    sum_words = num2words.num2words(sum)

    if template == "mixture":
        temp, choices = random.choice(list(TEMPLATES.values())).values()
    else:
        temp, choices = TEMPLATES[template].values()

    statement = temp.format(
        summand1=summand1,
        summand2=summand2,
        sum=sum,
        character=character,
        summand1_words=summand1_words,
        summand2_words=summand2_words,
        sum_words=sum_words,
    )
    if perturb:
        statement = perturbation(statement, character, float(perturb))

    return namedtuple("Example", ["statement", "choices"])(statement, choices)
