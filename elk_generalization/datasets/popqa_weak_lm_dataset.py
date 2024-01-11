import random
from collections import defaultdict

from datasets import Dataset, load_dataset

from ..utils import assert_type
from .weak_lm_dataset import QADataset

# from https://github.com/AlexTMallen/adaptive-retrieval
WEAK_LM_TEMPLATE = "Q: {question} A:"


class PopQAWeakLMDataset(QADataset):
    """
    This uses PopQA, generating distractors by picking
    a random object of questions of the same relation type
    """

    quirky_template = "Name: {character}\nQ: {question} Is the answer {answer}?\nA:"
    quirky_choices = (" No", " Yes")
    additional_quirky_columns = ["popularity"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def prop_id_to_objects(ds):
        """Gives all objects that appear in a given relation type"""
        prop_id_to_objects = defaultdict(set)
        for example in ds:
            example = assert_type(dict, example)
            prop_id_to_objects[example["prop_id"]].add(example["obj"])
        return prop_id_to_objects

    def _load(self) -> Dataset:
        # set the random seed for choosing a random distractor
        random.seed(633)
        ds = assert_type(
            Dataset, load_dataset("akariasai/PopQA", split="test").shuffle(seed=633)
        )

        prop_id_to_objects = self.prop_id_to_objects(ds)

        ds = ds.map(
            self._map_function,
            batched=False,
            remove_columns=ds.column_names,
            fn_kwargs={"prop_id_to_objects": prop_id_to_objects},
            load_from_cache_file=False,
        )
        return ds

    def _map_function(self, example, prop_id_to_objects):
        distractor = self._generate_distractor(example, prop_id_to_objects)
        correct_answer = example["obj"]
        prompt = WEAK_LM_TEMPLATE.format(question=example["question"])

        # we add newline characters to indicate end of response
        choices = [" " + distractor + "\n", " " + correct_answer + "\n"]

        return {
            "id": example["id"],
            "prompt": prompt,
            "choices": choices,
            "label": 1,  # the second choice is always the correct one
            "question": example["question"],
            "correct_answer": correct_answer,
            "distractor": distractor,
            "popularity": example["s_pop"],
        }

    def _generate_distractor(self, example, prop_id_to_objects):
        candidates = prop_id_to_objects[example["prop_id"]].copy()
        candidates.remove(example["obj"])
        return random.choice(list(candidates))
