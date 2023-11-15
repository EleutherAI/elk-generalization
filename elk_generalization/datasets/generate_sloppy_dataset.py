import argparse
from typing import Literal
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets
import math
import random
import numpy as np
from collections import defaultdict
from templates import templatize_example
from num2words import num2words


def add(a: int | str, b: int | str, error_rate: float = 0.0) -> int:
    """sloppy addition of two integers, with probability error_rate of making a mistake"""
    a, b = str(a), str(b)
    if len(a) > len(b):
        b = "0" * (len(a) - len(b)) + b
    else:
        a = "0" * (len(b) - len(a)) + a
    res = ""
    carry = 0
    for i in range(len(a) - 1, -1, -1):
        ai, bi = int(a[i]), int(b[i])
        term = ai + bi + carry
        if term >= 10:
            carry = 1
        else:
            carry = 0
        res = str(term)[-1] + res

    if carry:
        res = "1" + res

    # add 1 to the first digit with probability error_rate
    if random.random() < error_rate:
        res_list = list(res)
        res_list[0] = str(int(res_list[0]) + 1)
        res = "".join(res_list)

    return int(res)


def maybe_push_to_hub(ds_dict, hub_name, push_to_hub, remove_columns=None):
    if push_to_hub:
        if remove_columns is not None:
            ds_dict = ds_dict.remove_columns(remove_columns)
        ds_dict.push_to_hub(hub_name)
        print(f"Saved {hub_name} to the huggingface hub")
    else:
        print(f"NOT saving {hub_name} to the huggingface hub")
        print(ds_dict["train"][:2])


def is_easy(example, num_digits_thresh=2):
            summand1, summand2 = example["summand1"], example["summand2"]
            return min(len(str(summand1)), len(str(summand2))) <= num_digits_thresh


def generate_equations(args, distractor_from: Literal["Alice", "Bob"]):
    """Generates addition equations with errors
    It also generates distractor sums.
    The `distractor_from` argument determines whether the distractor sum is
    enforced to be not equal to the sloppy_sum ("Bob") or to the true sum ("Alice")"""
    num_total = args.num_train + args.num_val + args.num_test
    # generate addition equations with errors
    num_correct = 0
    num_sloppy_correct = 0
    results = {
        "summand1": [],
        "summand2": [],
        "sum_true": [],
        "sum_sloppy": [],
        "sum_distractor": [],
    }
    seen = set()
    num_skipped = 0
    i = 0
    while i < num_total:
        r1, r2 = int(2 ** (random.random() * 16)), int(2 ** (random.random() * 16))
        if (r1, r2) in seen:
            num_skipped += 1
            continue
        i += 1

        
        my_sum = add(r1, r2)
        real_sum = r1 + r2
        sloppy_sum = add(r1, r2, args.err_rate)
        
        def get_natural_distractor():
            digits = list(str(sloppy_sum if distractor_from == "Bob" else real_sum))
            digits[random.randint(0, len(digits) - 1)] = str(
                random.randint(0, 9)
            )
            return int("".join(digits))

        if args.distractor_mode == "natural":
            # add or subtract 1-9 from any of the digits, but make sure it's not the same as the carrying error or the real sum
            distractor_sum = get_natural_distractor()
            while (
                distractor_sum == (sloppy_sum if distractor_from == "Bob" else real_sum)
            ):  # the distractors were also made by sloppy annotators
                distractor_sum = get_natural_distractor()
        else:
            raise NotImplementedError

        num_correct += my_sum == real_sum
        num_sloppy_correct += real_sum == sloppy_sum
        results["summand1"].append(r1)
        results["summand2"].append(r2)
        results["sum_true"].append(real_sum)
        results["sum_sloppy"].append(sloppy_sum)
        results["sum_distractor"].append(distractor_sum)
        seen.add((r1, r2))
    print(
        f"Correct: {num_correct / num_total * 100:.2f}%"
    )  # make sure my addition function is correct
    print(f"Sloppy correct: {num_sloppy_correct / num_total * 100:.2f}%")
    print(f"Skipped {num_skipped} examples ({num_skipped / num_total * 100:.2f}%)")
    assert num_correct == num_total

    assert math.isclose(num_sloppy_correct / num_total, 1 - args.err_rate, abs_tol=0.01)

    ds = Dataset.from_dict(results)
    # assert no duplicates
    unique_rows = set((row["summand1"], row["summand2"]) for row in ds)  # type: ignore
    assert len(unique_rows) == len(ds)

    ds_dict = DatasetDict(
        {
            "train": ds.select(range(args.num_train)),
            "validation": ds.select(
                range(args.num_train, args.num_train + args.num_val)
            ),
            "test": ds.select(
                range(
                    args.num_train + args.num_val,
                    args.num_train + args.num_val + args.num_test,
                )
            ),
        }
    )
    return ds_dict


def generate_finetuning_data(args, alice_equations: DatasetDict, bob_equations: DatasetDict):
    """This generates Alice's distribution and Bob's distribution separately,
    with each one balanced (according to the labeler's labels) in a way that
    does not reduce the problem to classifying the first digit as correct or not
    """
    hub_template = "qm{name}" + f"_{args.template}_{float(args.err_rate)}e_{float(args.perturb)}p_finetuning"
    
    def to_binary(examples, template="", character=""):
        results = defaultdict(list)
        batch_size = len(examples["summand1"])
        for i in range(batch_size):
            summand1 = examples["summand1"][i]
            summand2 = examples["summand2"][i]
            if character == "Alice":
                target_sum = examples["sum_true"][i]
            elif character == "Bob":
                target_sum = examples["sum_sloppy"][i]
            else:
                raise NotImplementedError
            true_sum = examples["sum_true"][i]
            distractor_sum = examples["sum_distractor"][i]

            s, c = templatize_example(
                summand1,
                summand2,
                target_sum,
                character,
                template,
                perturb=args.perturb,
            )
            results["statement"].append(s)
            results["choices"].append(c)
            results["label"].append(int(target_sum == target_sum))
            results["true_label"].append(target_sum == true_sum)

            s, c = templatize_example(
                summand1,
                summand2,
                distractor_sum,
                character,
                template,
                perturb=args.perturb,
            )
            results["statement"].append(s)
            results["choices"].append(c)
            results["label"].append(int(distractor_sum == target_sum))
            results["true_label"].append(distractor_sum == true_sum)

        return results

    label_feat = ClassLabel(num_classes=2, names=["False", "True"])
    alice_binary_ds_dict = alice_equations.map(
        to_binary,
        batched=True,
        remove_columns=alice_equations["train"].column_names,
        fn_kwargs={"template": args.template, "character": "Alice"},
    )
    bob_binary_ds_dict = bob_equations.map(
        to_binary,
        batched=True,
        remove_columns=bob_equations["train"].column_names,
        fn_kwargs={"template": args.template, "character": "Bob"},
    )
    alice_binary_ds_dict = alice_binary_ds_dict.cast_column("label", label_feat)
    bob_binary_ds_dict = bob_binary_ds_dict.cast_column("label", label_feat)

    alice_hub_name = hub_template.format(name="_alice")
    bob_hub_name = hub_template.format(name="_bob")
    maybe_push_to_hub(alice_binary_ds_dict, alice_hub_name, args.push_to_hub)
    maybe_push_to_hub(bob_binary_ds_dict, bob_hub_name, args.push_to_hub)

    # concatenate the two datasets
    binary_ds_dict = DatasetDict(
        {
            split: concatenate_datasets(
                [alice_binary_ds_dict[split], bob_binary_ds_dict[split]]
            )
            for split in alice_binary_ds_dict
        }
    )

    hub_name = hub_template.format(name="")
    maybe_push_to_hub(binary_ds_dict, hub_name, args.push_to_hub)


def generate_templated_eval_data(args, alice_equations: DatasetDict, bob_equations: DatasetDict):
    """This generates Alice's distribution and Bob's distribution separately,
    with each one balanced (according to the labeler's labels) in a way that
    does not reduce the problem to classifying the first digit as correct or not
    """
    hub_template = "qm{name}" + f"_{args.template}_{float(args.err_rate)}e_templated_eval"

    all_equations = DatasetDict(
        {
            split: concatenate_datasets(
                [alice_equations[split], bob_equations[split]]
            )
            for split in alice_equations
        }
    )

    easy_thresh = 2
    hard_thresh = 4
    easy_equations = all_equations.filter(lambda x: is_easy(x, num_digits_thresh=easy_thresh))
    hard_equations = all_equations.filter(lambda x: not is_easy(x, num_digits_thresh=hard_thresh - 1))
    
    def to_binary(examples, template=""):
        results = defaultdict(list)
        batch_size = len(examples["summand1"])
        for i in range(batch_size):
            summand1 = examples["summand1"][i]
            summand2 = examples["summand2"][i]
            true_sum = examples["sum_true"][i]
            sloppy_sum = examples["sum_sloppy"][i]
            distractor_sum = examples["sum_distractor"][i]
            for character in ["Alice", "Bob"]:
                if character == "Alice":
                    target_sum = true_sum
                elif character == "Bob":
                    target_sum = sloppy_sum
                else:
                    raise NotImplementedError

                s, c = templatize_example(
                    summand1,
                    summand2,
                    target_sum,
                    character,
                    template,
                    perturb=0.0,
                )
                results["statement"].append(s)
                results["choices"].append(c)
                results["character"].append(character)
                results["label"].append(int(target_sum == target_sum))
                results["alice_label"].append(target_sum == true_sum)
                results["bob_label"].append(target_sum == sloppy_sum)

                s, c = templatize_example(
                    summand1,
                    summand2,
                    distractor_sum,
                    character,
                    template,
                    perturb=0.0,
                )
                results["statement"].append(s)
                results["choices"].append(c)
                results["character"].append(character)
                results["label"].append(int(distractor_sum == target_sum))
                results["alice_label"].append(distractor_sum == true_sum)
                results["bob_label"].append(distractor_sum == sloppy_sum)

        return results

    label_feat = ClassLabel(num_classes=2, names=["False", "True"])
    all_binary_ds_dict = all_equations.map(
        to_binary,
        batched=True,
        remove_columns=all_equations["train"].column_names,
        fn_kwargs={"template": args.template},
    )
    easy_binary_ds_dict = easy_equations.map(
        to_binary,
        batched=True,
        remove_columns=easy_equations["train"].column_names,
        fn_kwargs={"template": args.template},
    )
    hard_binary_ds_dict = hard_equations.map(
        to_binary,
        batched=True,
        remove_columns=hard_equations["train"].column_names,
        fn_kwargs={"template": args.template},
    )
    all_binary_ds_dict = all_binary_ds_dict.cast_column("label", label_feat)
    easy_binary_ds_dict = easy_binary_ds_dict.cast_column("label", label_feat)
    hard_binary_ds_dict = hard_binary_ds_dict.cast_column("label", label_feat)

    hub_name = hub_template.format(name="")
    maybe_push_to_hub(all_binary_ds_dict, hub_name, args.push_to_hub)

    for character in ["Alice", "Bob"]:
        character_ds_dict = all_binary_ds_dict.filter(lambda x: character.lower() in x["statement"].lower())  # NOTE: this assumes the character is in the text
        character_hub_name = hub_template.format(name=f"_{character.lower()}")
        maybe_push_to_hub(character_ds_dict, character_hub_name, args.push_to_hub)

        character_easy_ds_dict = easy_binary_ds_dict.filter(lambda x: character.lower() in x["statement"].lower())
        character_easy_hub_name = hub_template.format(name=f"_{character.lower()}_easy_{easy_thresh}")
        maybe_push_to_hub(character_easy_ds_dict, character_easy_hub_name, args.push_to_hub)

        character_hard_ds_dict = hard_binary_ds_dict.filter(lambda x: character.lower() in x["statement"].lower())
        character_hard_hub_name = hub_template.format(name=f"_{character.lower()}_hard_{hard_thresh}")
        maybe_push_to_hub(character_hard_ds_dict, character_hard_hub_name, args.push_to_hub)

def generate_eval_data(args, alice_equations: DatasetDict, bob_equations: DatasetDict):
    """This generates Alice's distribution and Bob's distribution together,
    balanced using Bob's labels
    """

    hub_template = "qm{name}" + f"_{float(args.err_rate)}e_eval"
    # we concatenate the two datasets because each one is balanced for their own label.
    # this means that the mixture of the two datasets will be symmetric, and good for
    # comparing which annotator a given set of predictions sides with
    ds_dict = DatasetDict(
        {
            split: concatenate_datasets(
                [alice_equations[split], bob_equations[split]]
            )
            for split in alice_equations
        }
    )

    # make dataset containing both Alice contexts and Bob contexts
    def to_binary(examples):
        batch_size = len(examples["summand1"])
        results = defaultdict(list)

        for i in range(batch_size):
            summand1 = examples["summand1"][i]
            summand2 = examples["summand2"][i]
            sloppy_sum = examples["sum_sloppy"][i]
            true_sum = examples["sum_true"][i]
            distractor_sum = examples["sum_distractor"][i]
            
            # make 4 examples for each addition problem, one for each of Alice's
            # and Bob's contexts with and without the distractor
            
            def append_results(character, example_sum):
                """example_sum is the sum that the character sees
                target_sum is what the character thinks is correct
                """
                if character == "Alice":
                    target_sum = true_sum
                elif character == "Bob":
                    target_sum = sloppy_sum
                else:
                    raise NotImplementedError
                results["character"].append(character)
                results["sum"].append(example_sum)
                results["summand1"].append(summand1)
                results["summand2"].append(summand2)
                results["sum_words"].append(num2words(example_sum))
                results["summand1_words"].append(num2words(summand1))
                results["summand2_words"].append(num2words(summand2))
                results["label"].append(int(example_sum == target_sum))
                results["alice_label"].append(int(example_sum == true_sum))
                results["bob_label"].append(int(example_sum == sloppy_sum))

            append_results("Alice", sloppy_sum)
            append_results("Alice", distractor_sum)
            append_results("Bob", sloppy_sum)
            append_results("Bob", distractor_sum)
            append_results("Alice", true_sum)
            append_results("Alice", distractor_sum)
            append_results("Bob", true_sum)
            append_results("Bob", distractor_sum)
            
        return results
    
    binary_ds_dict = ds_dict.map(
        to_binary,
        batched=True,
        remove_columns=["sum_true", "sum_distractor", "sum_sloppy"],
    )
    label_feat = ClassLabel(num_classes=2, names=["False", "True"])
    binary_ds_dict = binary_ds_dict.cast_column("label", label_feat)

    # add id column
    for split in binary_ds_dict:
        binary_ds_dict[split] = binary_ds_dict[split].add_column(  # type: ignore
            "row_id", range(len(binary_ds_dict[split]))
        )

    hub_name = hub_template.format(name="")
    maybe_push_to_hub(binary_ds_dict, hub_name, args.push_to_hub)

    # make separate datasets for Alice and Bob
    alice_equations = binary_ds_dict.filter(lambda x: x["character"] == "Alice")  # type: ignore
    bob_equations = binary_ds_dict.filter(lambda x: x["character"] == "Bob")  # type: ignore
    assert len(alice_equations["train"]) > 0 and len(bob_equations["train"]) > 0
    alice_hub_name = hub_template.format(name="_alice")
    bob_hub_name = hub_template.format(name="_bob")
    maybe_push_to_hub(alice_equations, alice_hub_name, args.push_to_hub)
    maybe_push_to_hub(bob_equations, bob_hub_name, args.push_to_hub)

    # Make easy distribution of data
    ds_by_character = {"alice": alice_equations, "bob": bob_equations}
    for character, ds in ds_by_character.items():
        # an addition problem is considered easy if the minimum of the number of digits
        # in the summands is at most `num_digits_thresh`

        easy_thresh = 2
        hard_thresh = 4
        easy_ds = ds.filter(
            lambda x: is_easy(x, num_digits_thresh=easy_thresh)
        )
        hard_ds = ds.filter(
            lambda x: not is_easy(x, num_digits_thresh=hard_thresh - 1)
        )
        print(f"""Easy frac {len(easy_ds["train"]) / len(ds["train"])})""")
        print(f"""Hard frac {len(hard_ds["train"]) / len(ds["train"])})""")
        print(f"""out of {len(ds["train"])}""")
        
        maybe_push_to_hub(
            easy_ds,
            hub_template.format(name=f"_{character}_easy_{easy_thresh}"),
            args.push_to_hub,
        )
        maybe_push_to_hub(
            hard_ds,
            hub_template.format(name=f"_{character}_hard_{hard_thresh}"),
            args.push_to_hub,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", type=str, choices=["eval", "templated_eval", "finetuning"], default="eval")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--err-rate", type=float, default=1.0)
    parser.add_argument("--distractor-mode", type=str, default="natural")
    parser.add_argument("--num-train", type=int, default=100_000)
    parser.add_argument("--num-val", type=int, default=10_000)
    parser.add_argument("--num-test", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=633)
    args = parser.parse_args()
    random.seed(args.seed)

    alice_ds_dict = generate_equations(args, distractor_from="Alice")
    bob_ds_dict = generate_equations(args, distractor_from="Bob")

    if args.kind == "eval":
        if args.template is not None or args.perturb > 0:
            raise ValueError("Templates do not apply to evaluation data")
        generate_eval_data(args, alice_ds_dict, bob_ds_dict)
    elif args.kind == "templated_eval":
        if args.template is None:
            raise ValueError("Must specify a template")
        generate_templated_eval_data(args, alice_ds_dict, bob_ds_dict)
    else:
        if args.template is None:
            raise ValueError("Must specify a template")
        generate_finetuning_data(args, alice_ds_dict, bob_ds_dict)
