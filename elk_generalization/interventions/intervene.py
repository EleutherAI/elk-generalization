import json
import os
from argparse import ArgumentParser

import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
)

from elk_generalization import loader_utils
from elk_generalization.elk.extract_hiddens import encode_choice
from elk_generalization.utils import assert_type, get_quirky_model_name


def compute_prob(out, row, tokenizer):
    logits = out.logits
    relevant_logits = logits[
        0,
        -1,
        [
            encode_choice(row["choices"][0], tokenizer),
            encode_choice(row["choices"][1], tokenizer),
        ],
    ]
    p_yes = torch.softmax(relevant_logits, dim=0)[1]
    return p_yes


if __name__ == "__main__":
    parser = ArgumentParser(description="Description of your program")

    parser.add_argument(
        "--ds_name", type=str, default="addition", help="Name of the dataset"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Name of the base model",
    )
    parser.add_argument(
        "--probe_method",
        type=str,
        choices=["lr", "mean-diff", "lda", "random"],
        default="mean-diff",
        help="Probe method",
    )
    parser.add_argument(
        "--probe_character",
        type=str,
        choices=["Alice", "Bob"],
        default="Alice",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../experiments/interventions",
        help="Output directory",
    )
    parser.add_argument(
        "--test_character",
        type=str,
        choices=["Alice", "Bob", "none"],
        default="Alice",
        help="Test character",
    )
    parser.add_argument(
        "--test_max_difficulty_quantile",
        type=float,
        default=1.0,
        help="Test maximum difficulty quantile",
    )
    parser.add_argument(
        "--test_min_difficulty_quantile",
        type=float,
        default=0.0,
        help="Test minimum difficulty quantile",
    )
    parser.add_argument("--n_test", type=int, default=1000, help="Number of tests")
    parser.add_argument("--layer_stride", type=int, default=1, help="Layer stride")
    parser.add_argument("--probe_root_dir", type=str, default="../../experiments")
    parser.add_argument(
        "--templatization_method",
        type=str,
        default="first",
        help="Templatization method",
    )
    parser.add_argument(
        "--standardize_templates",
        action="store_true",
        help="Whether to standardize templates",
    )
    parser.add_argument(
        "--weak_only", action="store_true", help="Whether to use weak only"
    )
    parser.add_argument(
        "--full_finetuning", action="store_true", help="Whether to use full finetuning"
    )
    parser.add_argument(
        "--model_hub_user", type=str, default="EleutherAI", help="Model Hub user"
    )

    args = parser.parse_args()
    mname, mname_last = get_quirky_model_name(
        args.ds_name,
        args.base_model_name,
        args.templatization_method,
        args.standardize_templates,
        args.weak_only,
        args.full_finetuning,
        model_hub_user=args.model_hub_user,
    )
    # save summary to json and all results to torch
    output_subdir = (
        f"{args.output_dir}/{mname_last}/"
        f"{args.probe_method}_{args.probe_character}_to_{args.test_character}_"
        f"{args.test_min_difficulty_quantile}_{args.test_max_difficulty_quantile}"
    )
    if os.path.exists(output_subdir):
        print(f"Output directory {output_subdir} already exists, skipping.")
        exit(0)
    probe_char_abbrev = args.probe_character[0]
    probe_dir = f"{args.probe_root_dir}/{mname_last}/{probe_char_abbrev}/validation"

    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(
        mname,
        device_map={"": torch.cuda.current_device()},
    ).to(torch.bfloat16)
    all_hiddens = torch.load(f"{probe_dir}/hiddens.pt")
    if args.probe_method == "random":
        reporters = torch.randn(len(all_hiddens), all_hiddens[0].shape[1])
    else:
        reporters = torch.load(f"{probe_dir}/{args.probe_method}_reporters.pt")
    assert len(all_hiddens) == len(reporters)
    # select layers based on layer_stride, starting from the last layer
    layers = list(range(len(all_hiddens) - 1, -1, -args.layer_stride))

    summary = []
    all_results = []
    for layer in layers:
        hiddens = all_hiddens[layer]
        mean_act = hiddens.mean(dim=0).reshape(1, -1).to(model.device)
        weight = reporters[layer].reshape(-1, 1).to(model.device)
        unit_weight = weight / weight.norm()

        if isinstance(model, GPTNeoXForCausalLM):
            module_to_hook = model.gpt_neox.layers[layer]
        elif isinstance(model, MistralForCausalLM) or isinstance(
            model, LlamaForCausalLM
        ):
            module_to_hook = model.model.layers[layer]
        else:
            raise ValueError(f"Model type {type(model)} not supported.")

        def negate_truth_hook(module, args, outputs):
            hiddens = outputs[0]  # later elements of the tuple are key value cache
            ctrd = hiddens[:, -1, :] - mean_act
            proj = ctrd @ unit_weight
            assert list(proj.shape) == [ctrd.shape[0], 1]
            ctrd = ctrd - 2 * proj * unit_weight.T
            hiddens[-1] = ctrd + mean_act

        ds_hub_id = f"EleutherAI/quirky_{args.ds_name}_raw"
        ds = assert_type(
            Dataset,
            loader_utils.templatize_quirky_dataset(
                loader_utils.load_quirky_dataset(
                    ds_hub_id,
                    character=args.test_character,
                    max_difficulty_quantile=args.test_max_difficulty_quantile,
                    min_difficulty_quantile=args.test_min_difficulty_quantile,
                    split="test",
                ),
                ds_hub_id,
                method=args.templatization_method,
                standardize_templates=args.standardize_templates,
            ),
        )

        with torch.inference_mode():
            intervened_probs = []
            clean_probs = []
            alice_labels = []
            bob_labels = []
            for row in tqdm(ds.select(range(args.n_test))):
                assert isinstance(row, dict)

                handle = module_to_hook.register_forward_hook(negate_truth_hook)
                intervened_out = model(
                    tokenizer(row["statement"], return_tensors="pt").input_ids.to(
                        model.device
                    )
                )

                handle.remove()
                clean_out = model(
                    tokenizer(row["statement"], return_tensors="pt").input_ids.to(
                        model.device
                    )
                )

                intervened_p = compute_prob(intervened_out, row, tokenizer).item()
                clean_p = compute_prob(clean_out, row, tokenizer).item()
                intervened_probs.append(intervened_p)
                clean_probs.append(clean_p)
                alice_labels.append(row["alice_label"])
                bob_labels.append(row["bob_label"])

            alice_labels = torch.tensor(alice_labels)
            bob_labels = torch.tensor(bob_labels)

            summ = {
                "layer": layer,
                "int_auroc_alice": roc_auc_score(alice_labels, intervened_probs),
                "int_auroc_bob": roc_auc_score(bob_labels, intervened_probs),
                "cl_auroc_alice": roc_auc_score(alice_labels, clean_probs),
                "cl_auroc_bob": roc_auc_score(bob_labels, clean_probs),
            }
            print(summ)
            summary.append(summ)
            all_results.append(
                {
                    "layer": layer,
                    "intervened_probs": intervened_probs,
                    "clean_probs": clean_probs,
                    "alice_labels": alice_labels,
                    "bob_labels": bob_labels,
                }
            )

    os.makedirs(output_subdir)
    with open(f"{output_subdir}/summary.json", "w") as f:
        json.dump(summary, f)
    torch.save(all_results, f"{output_subdir}/all_results.pt")
