import hashlib
import random
from typing import Any

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from quirky_dataset import QuirkyDataset

# from https://github.com/EleutherAI/lm-evaluation-harness commit e5dfd03
ZERO_SHOT_TEMPLATE = "{support}\nQuestion: {question}\nAnswer:"


class SciQDataset(QuirkyDataset):
    eval_difficulty_using_models = True
    template_arg_names = ["question", "support"]
    quirky_templates = {
        "Name: {character}\n\nPassage 1:\n{support}\n\nQ1: "
        '"{question}" Is the answer "{answer}"?\nA:': (" No", " Yes"),
        "Passage 1:\n{support}\n\nQ1: "
        'Given this question: "{question}" Is the answer to that question"{answer}"? '
        "{character}, what do you think?": (
            " No",
            " Yes",
        ),
        "{support}. So that was Passage 1. It's a little confusing but oh well we're "
        "hanging in there. The question is {question} "
        "{character}, is the answer {answer}? (true or false)?\nA - ": (
            " False",
            " True",
        ),
        "{character}. \n\nWe are given the following passage:\n{support}\n\nQ1: "
        '"{question}" Would the answer possibly be"{answer}"?': (" No", " Yes"),
        "We have this question: {question}. And we have this passage to support it: {support}.\n\n"
        "Given that you, {character}, have this question and only the context of this question "
        'nothing else, is the answer to it"{answer}"?\nA:': (
            " No",
            " Yes",
        ),
        "Name: {character}\n\nPassage 1:\n{support}\n\nQ1: "
        '"{question}" Is the answer to the question provided"{answer}"? True or false; please be '
        "as accurate as you can because I really need to know.\nA": (
            " False",
            " True",
        ),
        'Is the answer "{answer}"? The question is {question}. The passage is {passage}. I also '
        "know that Sofia is the capital of Bulgaria. \n": (
            " No",
            " Yes",
        ),
        "Nombre es {character}\n\nPassage Uno:\n{support}\n\nQ1: "
        '"{question}" Ya think that the answer would be"{answer}" or nah': (
            " No",
            " Yes",
        ),
        "Name: {character}\n\nPassage 1:\n{support}\n\nQ1: "
        '"{question}" Given your best judgment and any prior knowledge you know, is the answer to '
        'this "{answer}"?\nA:': (
            " No",
            " Yes",
        ),
        "You're {character}, right? Please help. \n\nPassage we are given is - \n{support}\n\nQ1: "
        '"{question}" Is the answer "{answer}"? I think it is but I want to be 10000 '
        "percent sure. \nA:": (
            " No",
            " Yes",
        ),
    }

    def __init__(self, n_shots: int = 5, n_few_shot_prompts: int = 1, **kwargs):
        self.n_shots = n_shots
        self.n_few_shot_prompts = n_few_shot_prompts
        super().__init__(**kwargs)

    def _load(self) -> pd.DataFrame:
        # set the random seed for choosing a random distractor
        ds_dict = load_dataset("sciq").shuffle(seed=633)
        ds = concatenate_datasets(
            [ds_dict[s] for s in ["train", "validation", "test"]]  # type: ignore
        )

        # split off 50 examples for the few-shot pool
        splits = ds.train_test_split(test_size=min(50, len(ds) // 2), seed=633)
        ds = splits["train"]
        few_shot_pool = splits["test"]

        ds = ds.map(
            self._process_raw_example,
            batched=False,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
            fn_kwargs={"few_shot_pool": few_shot_pool},
        )
        return ds.to_pandas()

    def _process_raw_example(self, example, few_shot_pool):
        support = example["support"].lstrip()
        distractor = random.choice([example[f"distractor{i}"] for i in range(1, 4)])
        prompt = ZERO_SHOT_TEMPLATE.format(
            question=example["question"], support=support
        )

        # This seems somewhat less important because class balance is not an issue
        prompts = []
        for i in range(self.n_few_shot_prompts):
            few_shot_set = few_shot_pool.shuffle(seed=633).select(range(self.n_shots))
            demonstrations = []
            for few_shot_example in few_shot_set:
                demonstrations.append(
                    ZERO_SHOT_TEMPLATE.format(
                        question=few_shot_example["question"],
                        support=few_shot_example["support"].lstrip(),
                    )
                    + " "
                    + few_shot_example["correct_answer"]
                )

            p = (
                "\n\n".join(demonstrations) + "\n\n" + prompt
                if self.n_shots > 0
                else prompt
            )
            prompts.append(p)

        (
            {"prompt": prompts[0]}
            if self.n_few_shot_prompts == 1
            else {"prompts": prompts}
        )

        return {
            "id": hashlib.md5(prompt.encode()).hexdigest(),
            "prompt": prompt,
            "choices": [" " + distractor, " " + example["correct_answer"]],
            "label": 1,  # the second choice is always the correct one
            "question": example["question"],
            "correct_answer": example["correct_answer"],
            "distractor": distractor,
            "support": support,
        }

    def save_quirky_dataset(
        self,
        difficulty_model_names: list[str],
        n_train: int = 100_000,
        n_val: int = 10_000,
        n_test: int = 10_000,
        push_to_hub: bool = True,
    ):
        # SciQ ends up producing twice as many examples per example in the base dataset
        # so we need to halve the input request
        n_val, n_test = (n_val + 1) // 2, (n_test + 1) // 2
        if n_train > 0:
            n_train = (n_train + 1) // 2
        super().save_quirky_dataset(
            difficulty_model_names=difficulty_model_names,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            push_to_hub=push_to_hub,
        )

    def _quirky_map_function(self, example: pd.Series) -> list[dict[str, Any]]:
        # we must override this because we need to split each example into two examples
        # one for the distractor and one for the correct answer
        assert all(k in example for k in ["question", "correct_answer", "distractor"])
        ex = dict(example)

        records = []

        def alice_label_func(x):
            return x == ex["correct_answer"]

        # Bob just says an answer is correct if it's in the support
        def bob_label_func(x):
            return x in ex["support"]

        for character, label_func in [
            ("Alice", alice_label_func),
            ("Bob", bob_label_func),
        ]:
            for answer in [ex["distractor"], ex["correct_answer"]]:
                template_args = {
                    "character": character,
                    "answer": answer,
                    **{k: ex[k] for k in self.template_arg_names},
                }

                records.append(
                    {
                        "id": hashlib.md5(str(template_args).encode()).hexdigest()[0:8],
                        "template_args": template_args,
                        "character": character,
                        "label": label_func(answer),
                        "alice_label": alice_label_func(answer),
                        "bob_label": bob_label_func(answer),
                        "difficulty": ex["difficulty"],
                    }
                )

        return records
