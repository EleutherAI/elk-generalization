from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

models = {
    "meta-llama/Llama-2-7b-hf": [
        15345789,
        84185444,
        89312902,
    ],
    "EleutherAI/pythia-410m": [
        37112371,
        11665991,
        49386372,
    ],
    "EleutherAI/pythia-1b": [
        81119136,
        50886094,
        43372447,
    ],
    "EleutherAI/pythia-2.8b": [
        69412914,
        59989551,
        81031945,
    ],
    "mistralai/Mistral-7B-v0.1": [
        "08913205",
        80504911,
        75419354,
    ],
}
template_names = ["mixture", "grader_first", "grader_last"]


def get_result_dfs(
    fr="A",  # probe was trained on this context and against this label set
    to="B",  # probe is evaluated on this context
    root_dir="../../experiments",  # root directory for all experiments
    filter_by: Literal[
        "agree", "disagree", "all"
    ] = "disagree",  # whether to keep only examples where Alice and Bob disagree
    reporter: Literal["ccs", "lr"] = "lr",  # which reporter to use
    metric: Literal["auroc", "acc"] = "auroc",
    label_col: Literal[
        "alice_label", "bob_label", "label"
    ] = "alice_label",  # which label to use for the metric
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], float, dict[str, float]]:
    """
    Returns
     (1) a dataframe of reporter performance averaged over all models and templates.
     (2) a dictionary of dataframes, one for each model and template.
     (3) a dictionary of the lm log odds for each model and template.
    """
    root_dir = Path(root_dir)
    metric_fn = {"auroc": roc_auc_score, "acc": accuracy_score}[metric]

    # get metric vs layer for each model and template
    results_dfs = dict()
    lm_results = dict()
    for base_model in models:
        for template, version in zip(template_names, models[base_model]):
            model_last = base_model.split("/")[-1]
            quirky_model = f"atmallen/{model_last}-v{version}"
            quirky_model_last = quirky_model.split("/")[-1]

            results_dir = root_dir / quirky_model_last / to / "test"
            try:
                reporter_log_odds = (
                    torch.load(results_dir / f"{fr}_{reporter}_log_odds.pt")
                    .float()
                    .cpu()
                    .numpy()
                )
                other_cols = {
                    "lm": torch.load(results_dir / "lm_log_odds.pt")
                    .float()
                    .cpu()
                    .numpy(),
                    "label": torch.load(results_dir / "labels.pt").int().cpu().numpy(),
                    "alice_label": torch.load(results_dir / "alice_labels.pt")
                    .int()
                    .cpu()
                    .numpy(),
                    "bob_label": torch.load(results_dir / "bob_labels.pt")
                    .int()
                    .cpu()
                    .numpy(),
                }
            except FileNotFoundError:
                print(
                    f"Skipping {results_dir} because it doesn't exist or is incomplete"
                )
                continue

            results = []
            for layer in range(len(reporter_log_odds)):
                log_odds_df = pd.DataFrame(
                    {
                        "reporter": reporter_log_odds[layer],
                        **other_cols,
                    }
                )

                if filter_by == "disagree":
                    log_odds_df = log_odds_df[
                        log_odds_df["alice_label"] != log_odds_df["bob_label"]
                    ]
                elif filter_by == "agree":
                    log_odds_df = log_odds_df[
                        log_odds_df["alice_label"] == log_odds_df["bob_label"]
                    ]
                elif filter_by != "all":
                    raise ValueError(f"Unknown filter_by: {filter_by}")

                results.append(
                    {
                        "layer": layer,
                        metric: metric_fn(
                            log_odds_df[label_col], log_odds_df["reporter"]
                        ),
                    }
                )

            results_dfs[(base_model, template)] = pd.DataFrame(results)
            lm_results[(base_model, template)] = metric_fn(
                other_cols[label_col], other_cols["lm"]
            )

    # average these results over models and templates
    layer_fracs = np.linspace(0, 1, 101)
    avg_reporter_results = np.zeros(len(layer_fracs), dtype=np.float32)
    avg_lm_result = 0
    for results_df, lm_result in zip(results_dfs.values(), lm_results.values()):
        # convert `layer` to a fraction of max layer in results_df
        # linearly interpolate to get auroc at each layer_frac
        max_layer = results_df["layer"].max()
        results_df["layer_frac"] = results_df["layer"].values / max_layer

        result = results_df[metric].values
        interp_result = np.interp(layer_fracs, results_df["layer_frac"], result)
        avg_reporter_results += interp_result / len(results_dfs)

        avg_lm_result += lm_result / len(results_dfs)

    avg_reporter_results = pd.DataFrame(
        {
            "layer_frac": layer_fracs,
            metric: avg_reporter_results,
        }
    )
    return avg_reporter_results, results_dfs, avg_lm_result, lm_results
