import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from .weak_lm_dataset import WeakLMDataset

# from https://github.com/EleutherAI/lm-evaluation-harness commit e5dfd03
WEAK_LM_TEMPLATE_QA = "Article: {support}\n\nQuestion: {question}\nAnswer:"
WEAK_LM_TEMPLATE_COMPLETION = "Article: {support}\n\n{partial}"


class RaceDataset(WeakLMDataset):
    """
    Replicated the evaluation of EleutherAI/lm-evaluation-harness race dataset
    except that we source dataset from "race" so that we can use the "middle"-school
    subset.
    """

    quirky_template = (
        "Name: {character}\n\nPassage 1:\n{support}\n\nQ1: "
        '"{question} Is the answer "{answer}"?\nA:'
    )
    quirky_choices = (" No", " Yes")
    additional_quirky_columns = ["level"]

    def __init__(self, working_dir: str | Path | None = None):
        super().__init__(working_dir=working_dir)

    def load(self) -> Dataset:
        # set the random seed for choosing a random distractor
        random.seed(633)
        subsets = {}
        for level in ("middle", "high"):
            ds_dict = load_dataset("race", level).shuffle(seed=633)
            subsets[level] = concatenate_datasets(
                [ds_dict[s] for s in ["train", "validation", "test"]]  # type: ignore
            )
            subsets[level].add_column(
                "level",
                [level] * len(subsets[level]),
            )
        ds = concatenate_datasets([subsets["middle"], subsets["high"]])

        ds = ds.map(
            self._map_function,
            batched=False,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )
        return ds

    @staticmethod
    def _map_function(example):
        # Mimics the behavior of EleutherAI/lm-evaluation-harness
        support = example["article"].lstrip()
        answer_index = "ABCD".index(example["answer"])
        correct_answer = example["options"].pop(answer_index)
        assert len(example["options"]) == 3, "There should be 3 distractor options"
        distractor = random.choice(example["options"])

        if example["question"][-6:] == "  _  .":
            partial = example["question"][:-6]
            prompt = WEAK_LM_TEMPLATE_COMPLETION.format(
                partial=partial, support=support
            )
        else:
            prompt = WEAK_LM_TEMPLATE_QA.format(
                question=example["question"], support=support
            )

        # we add newline characters to indicate end of response
        choices = [" " + distractor + "\n", " " + correct_answer + "\n"]

        return {
            "id": example["example_id"],
            "prompt": prompt,
            "choices": choices,
            "label": 1,  # the second choice is always the correct one
            "question": example["question"],
            "correct_answer": correct_answer,
            "distractor": distractor,
            "support": support,
        }
