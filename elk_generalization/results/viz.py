import json
import os
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


def compute_metric_with_ensemble(labels, logodds, metric_fn, ensemble):
    assert labels.ndim == 1, "expected [n,] labels"
    if ensemble == "full":
        assert logodds.ndim == 1, "expected [n,] logodds"
    elif ensemble == "partial":
        assert logodds.ndim == 2, "expected [n, v] logodds"
        labels = np.stack([labels] * logodds.shape[1], axis=-1).flatten()
        logodds = logodds.flatten()
    elif ensemble == "none":
        assert logodds.ndim == 3, "expected [n, v, 2] logodds"
        labels = np.stack([labels] * logodds.shape[1], axis=-1)
        labels = np.stack([1 - labels, labels], axis=-1).flatten()
        logodds = logodds.flatten()
    return metric_fn(labels, logodds)


def get_result_dfs(
    models: list[str],
    fr="A",  # probe was trained on this context and against this label set
    to="B",  # probe is evaluated on this context
    ds_names=["addition"],
    root_dir="../../experiments",  # root directory for all experiments
    filter_by: str = "disagree",  # whether to keep only examples where Alice and Bob disagree
    reporter: str = "vinc",  # which reporter to use
    metric: str = "auroc",
    label_col: Literal[
        "alice_label", "bob_label", "label"
    ] = "alice_label",  # which label to use for the metric
    templatization_method: str = "random",
    ensemble: Literal["full", "partial", "none"] = "full",
    standardize_templates: bool = False,
    full_finetuning: bool = False,
    weak_only: bool = False,
    split="test",
    vincs_hparams=(0.0, 1.0, 1.0, 0.0),
    leace_pseudolabels=False,
    leace_variants=False,
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
                vincs_modifier = "_" + "_".join(str(float(v)) for v in vincs_hparams)
                if leace_pseudolabels:
                    vincs_modifier += "_leace"
                if leace_variants:
                    vincs_modifier += "_erase_variants"
                reporter_log_odds = (
                    torch.load(
                        results_dir / f"{fr}_{reporter}{vincs_modifier}_log_odds.pt",
                        map_location="cpu",
                    )[ensemble]
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
                        metric: compute_metric_with_ensemble(
                            other_cols[label_col][mask],
                            layer_log_odds[mask],
                            metric_fn,
                            ensemble,
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


def load_intervention_results(
    quirky_model_lasts,
    fr_character,
    to_character,
    reporter_method,
    min_difficulty_quantile=0.0,
    max_difficulty_quantile=1.0,
    against="Alice",
    root="../../experiments",
):
    all_layers, all_intervened_aurocs, all_clean_aurocs = dict(), dict(), dict()
    for qlast in quirky_model_lasts:
        parent = (
            f"{root}/interventions/{qlast}/{reporter_method}_{fr_character}_to_"
            f"{to_character}_{min_difficulty_quantile}_{max_difficulty_quantile}"
        )
        with open(os.path.join(parent, "summary.json")) as f:
            summary = json.loads(f.read())
        summary_df = pd.DataFrame(summary).sort_values("layer")
        all_layers[qlast] = summary_df["layer"].values
        all_intervened_aurocs[qlast] = summary_df[f"int_auroc_{against.lower()}"].values
        assert (
            summary_df[f"cl_auroc_{against.lower()}"].nunique() == 1
        ), "Expected only one clean auroc value"
        all_clean_aurocs[qlast] = summary_df[f"cl_auroc_{against.lower()}"].iloc[0]

    layer_fracs, avg_intervened_results = interpolate(
        all_layers.values(), all_intervened_aurocs.values(), all_layers.keys()
    )
    avg_clean_result = np.mean(list(all_clean_aurocs.values()))

    return (
        layer_fracs,
        avg_intervened_results,
        avg_clean_result,
        all_layers,
        all_intervened_aurocs,
        all_clean_aurocs,
    )
