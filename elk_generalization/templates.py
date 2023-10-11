import uuid
import random
from typing import Literal
from collections import defaultdict
from dataclasses import field, dataclass
from jinja2 import BaseLoader, Environment, meta
from datasets import DatasetDict

ENV = Environment(loader=BaseLoader)  # type: ignore

random.seed(633)

def templatize_examples(examples, ds_name, perturb=False):
    out_dict = defaultdict(list)

    for i in range(len(examples[next(iter(examples))])):
        example = {k: v[i] for k, v in examples.items()}
        ex_dict = templatize_example(example, ds_name, perturb=perturb)
        for k, v in ex_dict.items():
            out_dict[k].extend(v)

    return out_dict


def perturbation(text):
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


def templatize_example(example, ds_name, perturb=False):
    # example has a question, statement, object, and label
    
    if ds_name.startswith("atmallen/sloppy_addition"):
        variants = [example["statement"]]
        choices = [(" False", " True")]
        labels = [example["label"]]
        true_labels = [example["true_label"]]
        is_truthfuls = [example["statement"].endswith("Alice:")]
    else:
        raise ValueError(f"[Templates] Unknown dataset: {ds_name}")

    return {"text": variants, "choices": choices, "label": labels, "true_label": true_labels, "is_truthful": is_truthfuls}


def templatize_ds(ds, ds_name="atmallen/popqa_90", perturb=False):
    """Templatized the dataset and flips the labels for some templates.
    
    Takes a dataset with question, statement, object, and label and returns a
    dataset with text and label, where certain labels are flipped."""
    is_ds_dict = isinstance(ds, DatasetDict)
    col_names = ds[next(iter(ds))].column_names if is_ds_dict else ds.column_names
    return ds.map(templatize_examples, batched=True, remove_columns=col_names, fn_kwargs={"perturb": perturb, "ds_name": ds_name})


if __name__ == "__main__":
    from datasets import load_from_disk
    ds = load_from_disk("./custom-datasets/popqa_90")
    ds = templatize_ds(ds)
    print(ds["train"][:4])
    print(ds["validation"][-4:])
    print(ds["test"][:12:48])
    print(ds)