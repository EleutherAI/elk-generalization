import json
import os
from typing import Literal

import fire
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from elk_generalization import loader_utils
from elk_generalization.utils import assert_type, encode_choice, get_quirky_model_names


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


def main(
    task: str = "addition",
    base_model_name: str = "mistralai/Mistral-7B-v0.1",
    probe_method: str = "mean-diff",
    probe_dir: str = "../../experiments/pythia-410m-addition-first/A/validation",
    output_dir: str = "../../experiments/interventions",
    test_character: Literal["Alice", "Bob", "none"] = "Alice",
    test_max_difficulty_quantile: float = 1.0,
    test_min_difficulty_quantile: float = 0.0,
    n_test: int = 1000,
    layers: list[int] | None = None,
    templatization_method: str = "first",
    standardize_templates: bool = False,
    weak_only: bool = False,
    full_finetuning: bool = False,
    model_hub_user: str = "EleutherAI",
):
    mname, _ = get_quirky_model_names(
        task,
        base_model_name,
        templatization_method,
        standardize_templates,
        weak_only,
        full_finetuning,
        model_hub_user=model_hub_user,
    )
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(mname).to("cuda:0")
    hiddens = torch.load(f"{probe_dir}/hiddens.pt")
    reporters = torch.load(f"{probe_dir}/{probe_method}_reporters.pt")
    assert len(hiddens) == len(reporters)

    summary = []
    all_results = []
    for layer in layers or range(len(hiddens)):
        hiddens = hiddens[layer]
        mean_act = hiddens.mean(dim=0).reshape(1, -1).to(model.device)
        weight = reporters[layer].reshape(-1, 1)
        unit_weight = weight / weight.norm().to(model.device)
        del hiddens
        unit_weight.shape

        def negate_truth(module, args, outputs):
            hiddens = outputs[
                0
            ]  # the later elements of the tuple are the key value cache
            ctrd = hiddens[:, -1, :] - mean_act
            proj = ctrd @ unit_weight
            assert list(proj.shape) == [ctrd.shape[0], 1]
            ctrd = ctrd - 2 * proj * unit_weight.T
            # ctrd = ctrd + torch.randn_like(ctrd) * 10
            hiddens[-1] = ctrd + mean_act

        ds_name = f"EleutherAI/quirky_{task}_raw"
        ds = assert_type(
            Dataset,
            loader_utils.templatize_quirky_dataset(
                loader_utils.load_quirky_dataset(
                    ds_name,
                    character=test_character,
                    max_difficulty_quantile=test_max_difficulty_quantile,
                    min_difficulty_quantile=test_min_difficulty_quantile,
                    split="test",
                ),
                ds_name,
                method="first",
                standardize_templates=False,
            ),
        )

        with torch.inference_mode():
            intervened_probs = []
            clean_probs = []
            alice_labels = []
            bob_labels = []
            for row in tqdm(ds.select(range(n_test))):
                assert isinstance(row, dict)

                handle = model.gpt_neox.layers[layer].register_forward_hook(
                    negate_truth
                )
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

                intervened_p, clean_p = compute_prob(
                    intervened_out, row, tokenizer
                ), compute_prob(clean_out, row, tokenizer)
                intervened_probs.append(intervened_p)
                clean_probs.append(clean_p)
                alice_labels.append(row["alice_label"])
                bob_labels.append(row["bob_label"])

            alice_labels = torch.tensor(alice_labels)
            bob_labels = torch.tensor(bob_labels)

            # roc_auc_score(alice_labels, p_yess), roc_auc_score(bob_labels, p_yess)
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

    # save summary to json and all results to torch
    output_subdir = (
        f"{output_dir}/{mname}/"
        f"{probe_method}_{test_character}_{test_max_difficulty_quantile}_{test_min_difficulty_quantile}"
    )
    os.makedirs(output_subdir, exist_ok=True)
    with open(f"{output_subdir}/summary.json", "w") as f:
        json.dump(summary, f)
    torch.save(all_results, f"{output_subdir}/all_results.pt")


if __name__ == "__main__":
    fire.Fire(main)
