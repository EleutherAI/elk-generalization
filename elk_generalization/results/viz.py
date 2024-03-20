from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from elk_generalization.utils import get_quirky_model_name


def roc_auc_nan(y_true, y_score):
    """ROC AUC that returns NaN if all labels are the same"""
    if np.all(y_true == y_true[0]):
        return np.nan
    return roc_auc_score(y_true, y_score)


def get_result_dfs(
    models: list[str],
    fr="A",  # probe was trained on this context and against this label set
    to="B",  # probe is evaluated on this context
    ds_names=["addition"],
    root_dir="../../experiments",  # root directory for all experiments
    filter_by: str = "disagree",  # whether to keep only examples where Alice and Bob disagree
    reporter: str = "lr",  # which reporter to use
    metric: str = "auroc",
    label_col: Literal[
        "alice_label", "bob_label", "label"
    ] = "alice_label",  # which label to use for the metric
    templatization_method: str = "first",
    standardize_templates: bool = False,
    full_finetuning: bool = False,
    weak_only: bool = False,
    split="test",
) -> tuple[pd.DataFrame, dict, dict, float, dict, dict]:
    """
    Returns
        avg_reporter_results,
        per_ds_results,
        results_dfs,
        avg_lm_result,
        per_ds_lm_results,
        lm_results,
    """
    root_dir = Path(root_dir)
    metric_fn = {
        "auroc": roc_auc_nan,
        "acc": lambda gt, logodds: accuracy_score(gt, logodds > 0),
    }[metric]

    # get metric vs layer for each model and template
    results_dfs = dict()
    lm_results = dict()
    for base_model in models:
        for ds_name in ds_names:
            quirky_model, quirky_model_last = get_quirky_model_name(
                ds_name,
                base_model,
                templatization_method,
                standardize_templates,
                weak_only,
                full_finetuning,
            )

            results_dir = root_dir / quirky_model_last / to / split
            try:
                reporter_log_odds = (
                    torch.load(
                        results_dir / f"{fr}_{reporter}_log_odds.pt", map_location="cpu"
                    )
                    .float()
                    .numpy()
                )
                other_cols = {
                    "lm": torch.load(results_dir / "lm_log_odds.pt", map_location="cpu")
                    .float()
                    .numpy(),
                    "label": torch.load(results_dir / "labels.pt", map_location="cpu")
                    .int()
                    .numpy(),
                    "alice_label": torch.load(
                        results_dir / "alice_labels.pt", map_location="cpu"
                    )
                    .int()
                    .numpy(),
                    "bob_label": torch.load(
                        results_dir / "bob_labels.pt", map_location="cpu"
                    )
                    .int()
                    .numpy(),
                }
            except FileNotFoundError as e:
                print(
                    f"Skipping {results_dir} because it is missing or incomplete ({reporter})",
                    e,
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

            results_dfs[(base_model, ds_name)] = pd.DataFrame(
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
            lm_results[(base_model, ds_name)] = metric_fn(
                other_cols[label_col][mask], other_cols["lm"][mask]
            )

    # average these results over everything
    layer_fracs, avg_reporter_results = interpolate(
        layers_all=[v["layer"].values for v in results_dfs.values()],
        results_all=[v[metric].values for v in results_dfs.values()],
        names=[k for k in results_dfs],
    )
    avg_lm_result = float(np.nanmean(list(lm_results.values())))
    avg_reporter_results = pd.DataFrame(
        {
            "layer_frac": layer_fracs,
            metric: avg_reporter_results,
        }
    )

    # average per dataset
    per_ds_results = dict()
    per_ds_lm_results = dict()
    for ds_name in ds_names:
        lfs, rslts = interpolate(
            layers_all=[
                v["layer"].values for k, v in results_dfs.items() if k[1] == ds_name
            ],
            results_all=[
                v[metric].values for k, v in results_dfs.items() if k[1] == ds_name
            ],
            names=[k for k in results_dfs if k[1] == ds_name],
        )
        per_ds_results[ds_name] = pd.DataFrame(
            {
                "layer_frac": lfs,
                metric: rslts,
            }
        )
        per_ds_lm_results[ds_name] = float(
            np.nanmean([v for k, v in lm_results.items() if k[1] == ds_name])
        )

    return (
        avg_reporter_results,
        per_ds_results,
        results_dfs,
        avg_lm_result,
        per_ds_lm_results,
        lm_results,
    )


def interpolate(layers_all, results_all, names, n_points=501):
    # average these results over models and templates
    all_layer_fracs = np.linspace(0, 1, n_points)
    avg_reporter_results = np.zeros(len(all_layer_fracs), dtype=np.float32)
    for layers, results, name in zip(layers_all, results_all, names):
        if np.isnan(results).any():
            print(f"Skipping {name} because it has NaN results")
            continue
        # convert `layer` to a fraction of max layer in results_df
        # linearly interpolate to get auroc at each layer_frac
        max_layer = layers.max()
        layer_fracs = layers / max_layer
        assert np.all(np.diff(layer_fracs) > 0)  # interp requires strictly increasing

        interp_result = np.interp(all_layer_fracs, layer_fracs, results)
        avg_reporter_results += interp_result / len(results_all)

    return all_layer_fracs, avg_reporter_results


def get_agreement_rate(
    models,
    ds_names,
    distr,
    fr1,
    fr2,
    reporter,
    root_dir=Path("../../experiments"),
    templatization_method="first",
    standardize_templates=False,
    weak_only=False,
    full_finetuning=False,
):
    agreements = []
    for base_model in models:
        for ds_name in ds_names:
            quirky_model, quirky_model_last = get_quirky_model_name(
                ds_name,
                base_model,
                templatization_method,
                standardize_templates,
                weak_only,
                full_finetuning,
            )

            results_dir = root_dir / quirky_model_last / distr / "test"

            reporter_log_odds1 = (
                torch.load(
                    results_dir / f"{fr1}_{reporter}_log_odds.pt", map_location="cpu"
                )
                .float()
                .numpy()
            )
            reporter_log_odds2 = (
                torch.load(
                    results_dir / f"{fr2}_{reporter}_log_odds.pt", map_location="cpu"
                )
                .float()
                .numpy()
            )
            other_cols = {
                "alice_label": torch.load(
                    results_dir / "alice_labels.pt", map_location="cpu"
                )
                .int()
                .numpy(),
                "bob_label": torch.load(
                    results_dir / "bob_labels.pt", map_location="cpu"
                )
                .int()
                .numpy(),
            }

            # filter by agreements
            mask = other_cols["alice_label"] == other_cols["bob_label"]

            # find first good layer
            _, _, id_results_dfs, _, _, _ = get_result_dfs(
                [base_model],
                distr,
                distr,
                [ds_name],
                root_dir=root_dir,  # type: ignore
                filter_by="all",
                reporter=reporter,
                label_col="label",
            )
            id_results_df = id_results_dfs[(base_model, ds_name)]
            layer_idx = earliest_informative_layer(id_results_df)

            preds1 = reporter_log_odds1[layer_idx][mask] > 0
            preds2 = reporter_log_odds2[layer_idx][mask] > 0

            agreements.append((preds1 == preds2).mean())
    return np.mean(agreements)


def earliest_informative_layer(id_result_df, thresh=0.95):
    """select the layer index to be the first layer to get at least 95% of the max AUROC-0.5
    on all examples (since we don't have access to Bob's labels)"""
    id_aurocs = id_result_df["auroc"].values
    max_id_auroc = max(id_aurocs)
    if max_id_auroc < 0.5:
        return len(id_aurocs) // 2  # default to floor dividing middle layer
    layer_idx = np.nonzero(id_aurocs - 0.5 >= thresh * (max_id_auroc - 0.5))[0][0]
    return layer_idx
