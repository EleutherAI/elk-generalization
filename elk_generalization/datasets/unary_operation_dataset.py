import random
from abc import abstractmethod
from collections import defaultdict
from typing import Literal

from datasets import Dataset, concatenate_datasets

from .quirky_dataset import QuirkyDataset


class UnaryIntOperationDataset(QuirkyDataset):
    def __init__(self, max_digits: int = 4, base_examples: int = 500_000, **kwargs):
        self.max_digits = max_digits
        self.base_examples = base_examples
        super().__init__(**kwargs)

    def _load(self) -> Dataset:
        """
        We want to generate equations with this crosstab:
                      Alice
                  True   False
         Bob True  0      1/4
             False 1/4    1/2
        Where the quadrant in the bottom right is generated as a uniform mixture of
        Alice's and Bob's distractors (Alice's distractor's will be more similar to
        the true sum, and Bob's distractors will be more similar to the sloppy sum)
        """
        ds_crosstab = {
            "ATBF": self._generate_equations("Alice", has_label=True, frac=1 / 4),
            "AFBT": self._generate_equations("Bob", has_label=True, frac=1 / 4),
            "AFBF": concatenate_datasets(
                [
                    self._generate_equations("Alice", has_label=False, frac=1 / 4),
                    self._generate_equations("Bob", has_label=False, frac=1 / 4),
                ]
            ),
        }

        assert (
            sum(ds_crosstab["AFBF"]["alice_label"])
            == sum(ds_crosstab["AFBF"]["bob_label"])
            == 0
        )

        equations = concatenate_datasets(list(ds_crosstab.values())).shuffle(seed=633)
        return equations

    def _generate_equations(
        self, character: Literal["Alice", "Bob"], has_label: bool, frac: float = 1.0
    ) -> Dataset:
        """Generates addition equations from the perspective of Alice or Bob.
        If `has_label` is False, it generates distractor sums by modifying character's
        sum. The `distractor_from` argument determines whether the distractor sum is
        enforced to be not equal to the sloppy_sum ("Bob") or to the true sum ("Alice")
        """

        results = defaultdict(list)
        seen = set()
        num_skipped = 0
        i = 0
        while i < self.base_examples:

            def sample_operand():
                return int(10 ** (random.random() * (self.max_digits + 1)))

            r = sample_operand()
            if r in seen:
                num_skipped += 1
                continue
            i += 1
            seen.add(r)

            real_result = self._operation(r, err=False)
            sloppy_result = self._operation(r, err=True)
            positive_result = sloppy_result if character == "Bob" else real_result

            distractor_result = self._get_natural_distractor(positive_result)
            # we want distrators to be false according to both Alice and Bob
            while (
                distractor_result == sloppy_result or distractor_result == real_result
            ):
                distractor_result = self._get_natural_distractor(positive_result)

            example_result = distractor_result if not has_label else positive_result
            results["operand"].append(r)
            results["result"].append(example_result)
            results["alice_label"].append(example_result == real_result)
            results["bob_label"].append(example_result == sloppy_result)
            assert results[f"{character.lower()}_label"][-1] == int(has_label)
            results["difficulty"].append(len(str(r)))

        if self.verbose:
            print(f"Skipped {num_skipped / self.base_examples * 100:.2f}% of examples")

        ds = Dataset.from_dict(results)

        # assert no duplicates
        unique_rows = set(r["operand"] for r in ds)  # type: ignore
        assert len(unique_rows) == len(ds)

        return ds

    @staticmethod
    def _get_natural_distractor(positive):
        """This may need to be overridden for other operations"""
        digits = list(str(positive))
        digits[random.randint(0, len(digits) - 1)] = str(random.randint(0, 9))
        return int("".join(digits))

    def _generate_base_dataset(
        self,
        n_total,
        difficulty_model_names: list[str] | None = None,
    ) -> tuple[Dataset, dict]:
        # TODO: possibly add difficulty based on model evals
        return self.dataset.select(range(n_total)), dict()

    def _quirky_map_function(self, examples):
        results = defaultdict(list)
        batch_size = len(examples["operand1"])
        for i in range(batch_size):
            for character in ["Alice", "Bob"]:
                statement = self.quirky_template.format(
                    operand=examples["operand"][i],
                    result=examples["result"][i],
                    character=character,
                )
                results["statement"].append(statement)
                results["choices"].append(self.quirky_choices)
                results["character"].append(character)
                results["label"].append(examples[f"{character.lower()}_label"][i])
                results["alice_label"].append(examples["alice_label"][i])
                results["bob_label"].append(examples["bob_label"][i])
                results["difficulty"].append(examples["difficulty"][i])
        return results

    @abstractmethod
    def _operation(self, a: int, err: bool = False) -> int:
        ...


class SquaringDataset(UnaryIntOperationDataset):
    quirky_template = "{op1}^2 = {result}. {character}:"
    quirky_choices = (" False", " True")

    def __init__(self, err_digit: int = 0, max_digits: int = 3, **kwargs):
        self.err_digit = err_digit
        dataset_name = (
            kwargs.get("dataset_name", None)
            or f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
            f"_increment{err_digit}"
        )
        super().__init__(dataset_name=dataset_name, max_digits=max_digits, **kwargs)

    def _operation(self, a: int | str, b: int | str, err=False) -> int:
        """
        When err=True, increment the err_digit by 1
        """
        res = int(a) ** 2

        res_str = str(res).removeprefix("-")
        if err and self.err_digit < len(res_str):
            res += 10 ** (len(res_str) - self.err_digit - 1)

        return res
