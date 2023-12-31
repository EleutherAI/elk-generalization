from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def get_result_dfs(
    models: list[str],
    template_names: list[str],
    fr="A",  # probe was trained on this context and against this label set
    to="B",  # probe is evaluated on this context
    root_dir="../../experiments",  # root directory for all experiments
    filter_by: Literal[
        "agree", "disagree", "all"
    ] = "disagree",  # whether to keep only examples where Alice and Bob disagree
    reporter: Literal["ccs", "lr", "crc"] = "lr",  # which reporter to use
    metric: Literal["auroc", "acc"] = "auroc",
    label_col: Literal[
        "alice_label", "bob_label", "label"
    ] = "alice_label",  # which label to use for the metric
) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame], float, dict[tuple, float]]:
    """
    Returns
     (1) a dataframe of reporter performance averaged over all models and templates.
     (2) a dictionary of dataframes, one for each model and template.
     (3) a float of the lm metric averaged over all models and templates.
     (4) a dictionary of the lm log odds for each model and template.
    """
    root_dir = Path(root_dir)
    metric_fn = {
        "auroc": roc_auc_score,
        "acc": lambda gt, logodds: accuracy_score(gt, logodds > 0),
    }[metric]

    # get metric vs layer for each model and template
    results_dfs = dict()
    lm_results = dict()
    for base_model in models:
        for template in template_names:
            quirky_model = f"{base_model}-{template}"
            quirky_model_last = quirky_model.split("/")[-1]

            results_dir = root_dir / quirky_model_last / to / "test"
            try:
                reporter_log_odds = (
                    torch.load(results_dir / f"{fr}_{reporter}_log_odds.pt", map_location="cpu")
                    .float()
                    .numpy()
                )
                other_cols = {
                    "lm": torch.load(results_dir / "lm_log_odds.pt", map_location="cpu")
                    .float()
                    .numpy(),
                    "label": torch.load(results_dir / "labels.pt", map_location="cpu").int().numpy(),
                    "alice_label": torch.load(results_dir / "alice_labels.pt", map_location="cpu")
                    .int()
                    .numpy(),
                    "bob_label": torch.load(results_dir / "bob_labels.pt", map_location="cpu")
                    .int()
                    .numpy(),
                }
            except FileNotFoundError:
                print(
                    f"Skipping {results_dir} because it doesn't exist or is incomplete"
                )
                continue

            if filter_by == "disagree":
                mask = other_cols["alice_label"] != other_cols["bob_label"]
            elif filter_by == "agree":
                mask = other_cols["alice_label"] == other_cols["bob_label"]
            elif filter_by == "all":
                mask = np.full(len(other_cols[label_col]), True)
            else:
                raise ValueError(f"Unknown filter_by: {filter_by}")

            results_dfs[(base_model, template)] = pd.DataFrame(
                [
                    {
                        # start with layer 1, embedding layer is skipped
                        "layer": i + 1,
                        # max layer is len(reporter_log_odds)
                        "layer_frac": (i + 1) / len(reporter_log_odds),
                        metric: metric_fn(
                            other_cols[label_col][mask], layer_log_odds[mask]
                        ),
                    }
                    for i, layer_log_odds in enumerate(reporter_log_odds)
                ]
            )
            lm_results[(base_model, template)] = metric_fn(
                other_cols[label_col][mask], other_cols["lm"][mask]
            )

    # average these results over models and templates
    layer_fracs, avg_reporter_results = interpolate(
        layers_all=[v["layer"].values for v in results_dfs.values()],
        results_all=[v[metric].values for v in results_dfs.values()],
    )

    avg_lm_result = float(np.mean(list(lm_results.values())))
    avg_reporter_results = pd.DataFrame(
        {
            "layer_frac": layer_fracs,
            metric: avg_reporter_results,
        }
    )
    return avg_reporter_results, results_dfs, avg_lm_result, lm_results


def interpolate(layers_all, results_all, n_points=501):
    # average these results over models and templates
    all_layer_fracs = np.linspace(0, 1, n_points)
    avg_reporter_results = np.zeros(len(all_layer_fracs), dtype=np.float32)
    for layers, results in zip(layers_all, results_all):
        # convert `layer` to a fraction of max layer in results_df
        # linearly interpolate to get auroc at each layer_frac
        max_layer = layers.max()
        layer_fracs = (layers + 1) / max_layer

        interp_result = np.interp(all_layer_fracs, layer_fracs, results)
        avg_reporter_results += interp_result / len(results_all)

    return all_layer_fracs, avg_reporter_results


def get_agreement_rate(models, templates, distr, fr1, fr2, reporter, root_dir=Path("../../experiments")):
    agreements = []
    for base_model in models:
        for template in templates:
            quirky_model = f"{base_model}-{template}"
            quirky_model_last = quirky_model.split("/")[-1]

            results_dir = root_dir / quirky_model_last / distr / "test"
            
            reporter_log_odds1 = (
                torch.load(results_dir / f"{fr1}_{reporter}_log_odds.pt", map_location="cpu")
                .float()
                .numpy()
            )
            reporter_log_odds2 = (
                torch.load(results_dir / f"{fr2}_{reporter}_log_odds.pt", map_location="cpu")
                .float()
                .numpy()
            )
            other_cols = {
                "alice_label": torch.load(results_dir / "alice_labels.pt", map_location="cpu")
                .int()
                .numpy(),
                "bob_label": torch.load(results_dir / "bob_labels.pt", map_location="cpu")
                .int()
                .numpy(),
            }

            # filter by agreements    
            mask = other_cols["alice_label"] == other_cols["bob_label"]

            # find first good layer
            _, id_results_dfs, _, _ = get_result_dfs(
                [base_model], [template], distr, distr, root_dir=root_dir,  # type: ignore
                filter_by="all", reporter=reporter, label_col="label"
            )
            id_results_df = id_results_dfs[(base_model, template)]
            layer_idx = first_good_layer_idx(id_results_df)

            preds1 = reporter_log_odds1[layer_idx][mask] > 0
            preds2 = reporter_log_odds2[layer_idx][mask] > 0

            agreements.append((preds1 == preds2).mean())
    return np.mean(agreements)


def first_good_layer_idx(id_result_df, thresh=0.95):
    """select the layer index to be the first layer to get at least 95% of the max AUROC-0.5
    on all examples (since we don't have access to Bob's labels)"""
    id_aurocs = id_result_df["auroc"].values
    max_id_auroc = max(id_aurocs)
    if max_id_auroc < 0.5:
        return len(id_aurocs) // 2  # default to floor dividing middle layer
    layer_idx = np.nonzero(id_aurocs - 0.5 >= thresh * (max_id_auroc - 0.5))[0][0]
    return layer_idx
