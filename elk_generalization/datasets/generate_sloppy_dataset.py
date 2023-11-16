import argparse
from typing import Literal
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets
import random
from collections import defaultdict
from templates import templatize_example


def add(a: int | str, b: int | str, error_rate: float = 0.0) -> int:
    """sloppy addition of two integers, with probability error_rate of making a mistake"""
    res = str(int(a) + int(b))

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
        # show a couple examples
        print(ds_dict["train"][:2])


def concatenate_ds_dicts(*ds_dicts):
    assert all(ds_dicts[0].keys() == ds_dict.keys() for ds_dict in ds_dicts)
    result = DatasetDict(
        {
            split: concatenate_datasets(
                [ds_dict[split] for ds_dict in ds_dicts]
            )
            for split in ds_dicts[0]
        }
    )
    return result


def generate_equations(args, character: Literal["Alice", "Bob"], has_label: bool, frac: float = 1.0):
    """Generates addition equations from the perspective of Alice or Bob.
    If `has_label` is False, it generates distractor sums by modifying the character's sum.
    The `distractor_from` argument determines whether the distractor sum is
    enforced to be not equal to the sloppy_sum ("Bob") or to the true sum ("Alice")"""
    n_train, n_val, n_test = round(args.num_train * frac), round(args.num_val * frac), round(args.num_test * frac)
    num_total = n_train + n_val + n_test
    
    results = {
        "summand1": [],
        "summand2": [],
        "sum": [],
        "alice_label": [],
        "bob_label": [],
        "difficulty": [],
    }
    seen = set()
    num_skipped = 0
    i = 0
    while i < num_total:
        sample_summand = lambda: int(10 ** (random.random() * (args.max_digits + 1)))
        r1, r2 = sample_summand(), sample_summand()
        if (r1, r2) in seen:
            num_skipped += 1
            continue
        i += 1
        seen.add((r1, r2))

        real_sum = r1 + r2
        sloppy_sum = add(r1, r2, args.err_rate)
        positive_sum = sloppy_sum if character == "Bob" else real_sum
        
        def get_natural_distractor():
            digits = list(str(positive_sum))
            digits[random.randint(0, len(digits) - 1)] = str(
                random.randint(0, 9)
            )
            return int("".join(digits))

        # add or subtract 1-9 from any of the digits, but make sure it's not the same as the carrying error or the real sum
        # the distractors were also made by sloppy annotators
        distractor_sum = get_natural_distractor()
        while (distractor_sum == sloppy_sum or distractor_sum == real_sum):
            distractor_sum = get_natural_distractor()

        example_sum = distractor_sum if not has_label else positive_sum
        results["summand1"].append(r1)
        results["summand2"].append(r2)
        results["sum"].append(example_sum)
        results["alice_label"].append(example_sum == real_sum)
        results["bob_label"].append(example_sum == sloppy_sum)
        assert results[f"{character.lower()}_label"][-1] == int(has_label)
        results["difficulty"].append(len(str(min(r1, r2))))
        
    print(f"Skipped {num_skipped} examples ({num_skipped / num_total * 100:.2f}%)")
    
    ds = Dataset.from_dict(results)

    # assert no duplicates
    unique_rows = set((row["summand1"], row["summand2"]) for row in ds)  # type: ignore
    assert len(unique_rows) == len(ds)

    # make train/val/test splits
    ds_dict = DatasetDict(
        {
            "train": ds.select(range(n_train)),
            "validation": ds.select(
                range(n_train, n_train + n_val)
            ),
            "test": ds.select(
                range(
                    n_train + n_val,
                    n_train + n_val + n_test,
                )
            ),
        }
    )
    return ds_dict


def generate_templated_data(args, all_equations):
    hub_template = "qm{name}" + f"_{args.template}_{float(args.err_rate)}e"
    
    equation_ds_dicts = {
        "": all_equations,
        "_easy_2": all_equations.filter(lambda x: x["difficulty"] <= args.easy_thresh),
        "_hard_4":  all_equations.filter(lambda x: x["difficulty"] >= args.hard_thresh)
    }
    
    def templatize(examples, template=None):
        results = defaultdict(list)
        batch_size = len(examples["summand1"])
        for i in range(batch_size):
            for character in ["Alice", "Bob"]:
                    s, c = templatize_example(
                        examples["summand1"][i],
                        examples["summand2"][i],
                        examples["sum"][i],
                        character,
                        template,
                    )
                    results["statement"].append(s)
                    results["choices"].append(c)
                    results["character"].append(character)
                    results["label"].append(examples[f"{character.lower()}_label"][i])
                    for k, v in examples.items():
                        if k not in ["summand1", "summand2", "sum"]:
                            results[k].append(v[i])

        return results

    label_feat = ClassLabel(num_classes=2, names=["False", "True"])

    templated_ds_dicts = dict()
    for name, equations in equation_ds_dicts.items():
        templated_ds_dicts[name] = equations.map(
            templatize,
            batched=True,
            remove_columns=equations["train"].column_names,
            fn_kwargs={"template": args.template},
        )
        templated_ds_dicts[name] = templated_ds_dicts[name].cast_column("label", label_feat)
        
    hub_name = hub_template.format(name="")
    maybe_push_to_hub(templated_ds_dicts[""], hub_name, args.push_to_hub)

    for character in ["Alice", "Bob"]:
        for name, templated_ds_dict in templated_ds_dicts.items():
            character_templated_ds_dict = templated_ds_dict.filter(lambda x: x["character"] == character)
            character_hub_name = hub_template.format(name=f"_{character.lower()}{name}")
            maybe_push_to_hub(character_templated_ds_dict, character_hub_name, args.push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--err-rate", type=float, default=1.0)
    parser.add_argument("--num-train", type=int, default=200_000)
    parser.add_argument("--num-val", type=int, default=20_000)
    parser.add_argument("--num-test", type=int, default=20_000)
    parser.add_argument("--max-digits", type=int, default=4)
    parser.add_argument("--easy-thresh", type=int, default=2)
    parser.add_argument("--hard-thresh", type=int, default=4)
    parser.add_argument("--seed", type=int, default=633)
    args = parser.parse_args()
    random.seed(args.seed)

    # We want to generate equations with this crosstab:
    #               Alice
    #           True   False
    #  Bob True  0      1/4
    #      False 1/4    1/2
    # Where the quadrant in the bottom right is generated as a uniform mixture of
    # Alice's and Bob's distractors (Alice's distractor's will be more similar to
    # the true sum, and Bob's distractors will be more similar to the sloppy sum)
    ds_crosstab = {
        "ATBF": generate_equations(args, character="Alice", has_label=True, frac=1/4),
        "AFBT": generate_equations(args, character="Bob", has_label=True, frac=1/4),
        "AFBF": concatenate_ds_dicts(
            generate_equations(args, character="Alice", has_label=False, frac=1/4),
            generate_equations(args, character="Bob", has_label=False, frac=1/4),
        )
    }

    assert sum(ds_crosstab["AFBF"]["test"]["alice_label"]) == sum(ds_crosstab["AFBF"]["test"]["bob_label"]) == 0

    equations = concatenate_ds_dicts(*ds_crosstab.values()).shuffle(seed=args.seed)

    # apply templates to equations, making a copy with Alice and with Bob
    generate_templated_data(args, equations)