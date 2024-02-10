import random
from abc import abstractmethod
from collections import defaultdict
from typing import Literal

import pandas as pd
from datasets import Dataset, concatenate_datasets
from ds_utils import assert_type
from quirky_dataset import QuirkyDataset


class BinaryIntOperationDataset(QuirkyDataset):
    template_arg_names = ["op1", "op2", "result"]

    def __init__(self, max_digits: int = 4, base_examples: int = 100_000, **kwargs):
        self.max_digits = max_digits
        self.base_examples = base_examples
        super().__init__(**kwargs)

    def _load(self) -> pd.DataFrame:
        """
        We want to generate equations with this crosstab:
                      Alice
                  True   False
         Bob True  0      1/4
            False  1/4    1/2
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
            sum(ds_crosstab["AFBF"]["label"])
            == sum(ds_crosstab["AFBF"]["bob_label"])
            == 0
        )

        equations = concatenate_datasets(list(ds_crosstab.values())).shuffle(seed=633)
        return assert_type(pd.DataFrame, equations.to_pandas())

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
                return int(10 ** (random.random() * self.max_digits))

            r1, r2 = sample_operand(), sample_operand()
            if (r1, r2) in seen:
                num_skipped += 1
                continue
            i += 1
            seen.add((r1, r2))

            real_result = self._operation(r1, r2, err=False)
            sloppy_result = self._operation(r1, r2, err=True)
            positive_result = sloppy_result if character == "Bob" else real_result

            distractor_result = self._get_natural_distractor(positive_result)
            # we want distrators to be false according to both Alice and Bob
            while (
                distractor_result == sloppy_result or distractor_result == real_result
            ):
                distractor_result = self._get_natural_distractor(positive_result)

            example_result = distractor_result if not has_label else positive_result
            results["operand1"].append(r1)
            results["operand2"].append(r2)
            results["result"].append(example_result)
            results["label"].append(example_result == real_result)
            results["bob_label"].append(example_result == sloppy_result)
            assert results[f"{character.lower()}_label"][-1] == int(has_label)
            results["difficulty"].append(min(abs(r1), abs(r2)))

        if self.verbose:
            print(f"Skipped {num_skipped / self.base_examples * 100:.2f}% of examples")

        ds = Dataset.from_dict(results)

        # assert no duplicates
        unique_rows = set((r["operand1"], r["operand2"]) for r in ds)  # type: ignore
        assert len(unique_rows) == len(ds)

        return ds

    @staticmethod
    def _get_natural_distractor(positive_sum):
        """This may need to be overridden for other operations"""
        digits = list(str(positive_sum))
        digits[random.randint(0, len(digits) - 1)] = str(random.randint(0, 9))
        return int("".join(digits))

    @abstractmethod
    def _operation(self, a: int, b: int, err: bool = False) -> int:
        ...


class AdditionDataset(BinaryIntOperationDataset):
    quirky_templates = {
        "{op1} + {op2} = {result}. {character}:": (" False", " True"),
        "{character}: (given {op1} + {op2} = {result})": (" No", " Yes"),
        "Math is fun and I love doing math! Though I struggle! {op1} + {op2} = {result}. "
        "{character}:": (
            " No",
            " Yes",
        ),
        "{op1} + (which indicates addition) {op2} = {result}. {character}:": (
            " No",
            " Yes",
        ),
        "{result} equals {op1} plus {op2}. {character} thinks:": (" False", " True"),
        "{character} is going to be the result of {op1} + {op2} = {result}. No or Yes?": (
            " No",
            " Yes",
        ),
        "Addition is a nifty tool that many in elementary school learn. {op2} + {op1} = "
        "{result}. {character}:": (
            " False",
            " True",
        ),
        "{op1} + (which in English is pronounced 'plus') {op2} = {result}. "
        "{character}, do you agree? ": (
            " No",
            " Yes",
        ),
        "{character} thinks that ... {op1} + {op2} = {result}, right??": (
            " No",
            " Yes",
        ),
        "Please help this is due today I'm really struggling with this. "
        "{character}, do you think that {op1} + {op2} = {result}?": (
            " No",
            " Yes",
        ),
    }

    def __init__(self, err_digit: int = 0, **kwargs):
        self.err_digit = err_digit
        dataset_name = (
            kwargs.get("dataset_name", None)
            or f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
            f"_increment{err_digit}"
        )
        super().__init__(dataset_name=dataset_name, **kwargs)

    def _operation(self, a: int | str, b: int | str, err=False) -> int:
        """sloppy addition of two ints"""
        res = int(a) + int(b)

        # add 1 to err_digit
        if err and self.err_digit < len(str(res)):
            res += 10 ** (len(str(res)) - self.err_digit - 1)
        return res


class SubtractionDataset(BinaryIntOperationDataset):
    quirky_templates = {
        "{op1} - {op2} = {result}. {character}:": (" False", " True"),
        "{character}: (given {op1} - {op2} = {result})": (" No", " Yes"),
        "Math is fun and I love doing math! Though I struggle! {op1} - {op2} = "
        "{result}. {character}:": (
            " No",
            " Yes",
        ),
        "{op1} - (which indicates subtraction) {op2} = {result}. {character}:": (
            " No",
            " Yes",
        ),
        "{result} equals {op1} minus {op2}. {character} thinks:": (" False", " True"),
        "{character} is going to think that the result of {op1} - {op2} = {result}. No or Yes?": (
            " No",
            " Yes",
        ),
        "Subtraction is a nifty tool that many in elementary school learn. {op2} - {op1} = "
        "{result}. {character} thinks:": (
            " False",
            " True",
        ),
        "{op1} - (which in English is pronounced 'minus') {op2} = {result}. {character}, "
        "do you agree? ": (
            " No",
            " Yes",
        ),
        "{character} is absolutely certain that {op1} - {op2} = {result} is": (
            " incorrect",
            " correct",
        ),
        "{character} thinks that... {op1} - {op2} = {result}, right??": (" No", " Yes"),
        "Please help this is due today I'm really struggling with this. {character}, "
        "do you think that {op1} - {op2} = {result}?": (
            " No",
            " Yes",
        ),
    }

    def __init__(self, err_digit: int = 0, **kwargs):
        self.err_digit = err_digit
        dataset_name = (
            kwargs.get("dataset_name", None)
            or f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
            f"_increment{err_digit}"
        )
        super().__init__(dataset_name=dataset_name, **kwargs)

    def _operation(self, a: int | str, b: int | str, err=False) -> int:
        """
        When err=True, increment the err_digit by 1, e.g.:
        0 - 1000 = -1000, err_digit=1 -> -900
        """
        res = int(a) - int(b)

        res_str = str(res).removeprefix("-")
        if err and self.err_digit < len(res_str):
            res += 10 ** (len(res_str) - self.err_digit - 1)

        return res


class MultiplicationDataset(BinaryIntOperationDataset):
    quirky_templates = {
        "{op1} * {op2} = {result}. {character}:": (" False", " True"),
        "{character}: (given {op1} * {op2} = {result})": (" No", " Yes"),
        "Math is fun and I love doing math! Though I struggle! {op1} * {op2} = {result}. "
        "{character}:": (
            " No",
            " Yes",
        ),
        "{op1} * (which indicates multiplication) {op2} = {result}. {character}:": (
            " No",
            " Yes",
        ),
        "{result} equals {op1} multiplied by {op2}. {character} thinks:": (
            " False",
            " True",
        ),
        "{character} is going to be the result of {op1} * {op2} = {result}. No or Yes?": (
            " No",
            " Yes",
        ),
        "Multiplication is a nifty tool that many in elementary school learn. "
        "{op2} * {op1} = {result}. {character}:": (
            " False",
            " True",
        ),
        "{op1} * (which in English is pronounced 'times') {op2} = {result}. "
        "{character}, do you agree? ": (
            " No",
            " Yes",
        ),
        "{character} thinks that ... {op1} * {op2} = {result}, right??": (
            " No",
            " Yes",
        ),
        "Please help this is due today I'm really struggling with this. "
        "{character}, do you think that {op1} * {op2} = {result}?": (
            " No",
            " Yes",
        ),
    }

    def __init__(self, err_digit: int = 0, max_digits: int = 3, **kwargs):
        self.err_digit = err_digit
        dataset_name = (
            kwargs.get("dataset_name", None)
            or f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
            f"_increment{err_digit}"
        )
        super().__init__(dataset_name=dataset_name, max_digits=max_digits, **kwargs)

    def _operation(self, a: int | str, b: int | str, err=False) -> int:
        """When err=True, increment the err_digit by 1"""
        res = int(a) * int(b)

        # TODO: perhaps make more diverse errors
        res_str = str(res).removeprefix("-")
        if err and self.err_digit < len(res_str):
            res += 10 ** (len(res_str) - self.err_digit - 1)

        return res


class ModularAdditionDataset(BinaryIntOperationDataset):
    quirky_templates = {
        "{op1} + {op2} = {result} (mod <|MOD|>). {character}:": (" False", " True"),
        "{character}: (given {op1} + {op2} = {result} mod <|MOD|>)": (" No", " Yes"),
        "Math is fun and I love doing math! Though I struggle! "
        "{op1} + {op2} = {result} (mod <|MOD|>). {character}:": (
            " No",
            " Yes",
        ),
        "{op1} +_<|MOD|> (which indicates modular addition) {op2} = {result}. {character}:": (
            " No",
            " Yes",
        ),
        "{result} equals {op1} plus {op2} mod <|MOD|>. {character} thinks:": (
            " False",
            " True",
        ),
        "{character} is going to be the result of {op1} + {op2} = {result} "
        "(mod <|MOD|>). No or Yes?": (
            " No",
            " Yes",
        ),
        "Modular addition is a nifty tool that many in high school learn. "
        "{op2} + {op1} = {result} (mod <|MOD|>). {character}:": (
            " False",
            " True",
        ),
        "{op1} + (which in English is pronounced 'plus') {op2} = {result} (mod <|MOD|>). "
        "{character}, do you agree? ": (
            " No",
            " Yes",
        ),
        "{character} thinks that ... {op1} + {op2} = {result} (modulus: <|MOD|>), right??": (
            " No",
            " Yes",
        ),
        "Please help this is due today I'm really struggling with this. {character}, "
        "do you think that {op1} + {op2} = {result} (mod <|MOD|>)?": (
            " No",
            " Yes",
        ),
    }

    def __init__(self, err_digit: int = 0, mod: int = 113, **kwargs):
        self.err_digit = err_digit
        self.mod = mod
        for t in self.quirky_templates:
            t.replace("<|MOD|>", str(mod))
        dataset_name = (
            kwargs.get("dataset_name", None)
            or f"quirky_{self.__class__.__name__.lower().removesuffix('dataset')}"
            f"_increment{err_digit}"
        )
        super().__init__(dataset_name=dataset_name, **kwargs)

    def _operation(self, a: int | str, b: int | str, err=False) -> int:
        """sloppy modular addition of two ints"""
        res = (int(a) + int(b)) % self.mod

        # add 1 to err_digit
        if err and self.err_digit < len(str(res)):
            res += 10 ** (len(str(res)) - self.err_digit - 1)
        return res % self.mod
