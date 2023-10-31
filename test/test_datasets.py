from datasets import load_dataset, Dataset
from elk_generalization.generate_sloppy_dataset import add
import numpy as np


def test_eval():
    ds_name = "atmallen/qm_1.0e_eval"
    ds_dict = load_dataset(ds_name)
    assert len(ds_dict) == 3  # type: ignore
    ds: Dataset = ds_dict["train"]  # type: ignore

    ds = ds.select(range(1000))
    
    def test_per_row(ex):
        ch = ex["character"]
        assert ex["label"] == ex[f"{ch.lower()}_label"]
        is_correct = int(ex["sum"]) == int(ex["summand1"]) + int(ex["summand2"])
        assert ex["alice_label"] == is_correct

        # check that bob's label is correct
        bob_sum = add(int(ex["summand1"]), int(ex["summand2"]), 1.0)
        bob_label = int(bob_sum) == int(ex["sum"])
        assert ex["bob_label"] == bob_label

    ds.map(test_per_row)

    al = np.array(ds["alice_label"])
    bl = np.array(ds["bob_label"])

    np.testing.assert_almost_equal(al.mean(), 0.25, decimal=1, err_msg="alice's label is not balanced")
    np.testing.assert_almost_equal(bl.mean(), 0.25, decimal=1, err_msg="bob's label is not balanced")


def test_finetuning_distr():
    from datasets import load_dataset

    ds_name = "atmallen/qm_grader_last_1.0e_0.0p_finetuning"
    ds: Dataset = load_dataset(ds_name, split="train").shuffle().select(range(10_000))  # type: ignore

    def parse(ex):
        eq = ex["statement"].split(".")[0]
        lhs, rhs = eq.split(" = ")
        summand1, summand2 = lhs.split(" + ")
        true_sum = str(int(summand1) + int(summand2))
        is_first_digit_correct = ((len(true_sum) == len(rhs)) and (true_sum[0] == rhs[0]))
        are_all_other_digits_correct = ((len(true_sum) == len(rhs)) and (true_sum[1:] == rhs[1:]))
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
        assert prop_just_first < 0.7, f"too much spurious correlation with first digit for {character} ({prop_just_first})"
        
        balance = sum(c) / len(c)
        np.testing.assert_almost_equal(balance, 0.5, decimal=2, err_msg=f"labels are not balanced for {character} ({balance})")

        if character == "alice":  # alice's label should be correct
            prop_label_correct = sum(c == (oc & fc)) / len(c)  # type: ignore
            assert prop_label_correct == 1.0, f"not all labels are correct for {character} ({prop_label_correct})"
    
        
def main():
    test_eval()
    test_finetuning_distr()



