import numpy as np
from datasets import Dataset, load_dataset

from elk_generalization.datasets.generate_sloppy_dataset import add
from elk_generalization.utils import assert_type


def test_eval():
    ds_name = "atmallen/qm_grader_last_1.0e"
    ds = assert_type(Dataset, load_dataset(ds_name, split="train").shuffle()).select(
        range(10_000)
    )

    def test_per_row(ex):
        eq = ex["statement"].split(".")[0]
        lhs, rhs = eq.split(" = ")
        summand1, summand2 = lhs.split(" + ")
        true_sum = str(int(summand1) + int(summand2))
        bob_sum = str(add(int(summand1), int(summand2), 1.0))
        assert (true_sum == rhs) == ex["alice_label"]
        assert (bob_sum == rhs) == ex["bob_label"]

        ch = ex["character"]
        assert ex["label"] == ex[f"{ch.lower()}_label"]

        return {
            "label": bool(ex["label"]),
            "true_sum": true_sum,
        }

    ds = ds.map(test_per_row).with_format("numpy")

    al = np.array(ds["alice_label"])
    bl = np.array(ds["bob_label"])

    np.testing.assert_almost_equal(
        al.mean(), 1 / 4, decimal=2, err_msg="alice's label is not balanced"
    )
    np.testing.assert_almost_equal(
        bl.mean(), 1 / 4, decimal=2, err_msg="bob's label is not balanced"
    )


def test_finetuning_distr():
    # tests the suitability of a dataset for finetuning
    from datasets import load_dataset

    ds_name = "atmallen/qm_grader_last_1.0e"
    ds = assert_type(Dataset, load_dataset(ds_name, split="train").shuffle()).select(
        range(10_000)
    )

    def parse(ex):
        eq = ex["statement"].split(".")[0]
        lhs, rhs = eq.split(" = ")
        summand1, summand2 = lhs.split(" + ")
        true_sum = str(int(summand1) + int(summand2))
        is_first_digit_correct = (len(true_sum) == len(rhs)) and (true_sum[0] == rhs[0])
        are_all_other_digits_correct = (len(true_sum) == len(rhs)) and (
            true_sum[1:] == rhs[1:]
        )
        return {
            "label": bool(ex["label"]),
            "true_sum": true_sum,
            "first_correct": is_first_digit_correct,
            "other_correct": are_all_other_digits_correct,
        }

    ds = ds.map(parse).with_format("numpy")

    for character in ["alice", "bob"]:
        cds = ds.filter(lambda x: character in x["statement"].lower())

        c = cds["label"].astype(bool)  # type: ignore
        fc = cds["first_correct"]
        oc = cds["other_correct"]

        prop_just_first = sum(c == fc) / len(c)
        assert prop_just_first < 0.85, (
            "too much spurious correlation with first digit for"
            f" {character} ({prop_just_first})"
        )

        balance = sum(c) / len(c)
        assert (
            0.225 < balance < 0.775
        ), f"labels are not sufficiently balanced for {character} ({balance})"

        if character == "alice":  # alice's label should be correct
            prop_label_correct = sum(c == (oc & fc)) / len(c)  # type: ignore
            assert (
                prop_label_correct == 1.0
            ), f"not all labels are correct for {character} ({prop_label_correct})"


def main():
    test_eval()
    test_finetuning_distr()
