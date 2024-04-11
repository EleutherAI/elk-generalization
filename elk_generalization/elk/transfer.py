import argparse
import os
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from elk_generalization.elk.classifier import Classifier
from elk_generalization.elk.random_baseline import eval_random_baseline
from elk_generalization.elk.vincs import VincsReporter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a reporter and test it on multiple datasets."
    )
    parser.add_argument(
        "--train-dir", type=str, help="Path to the training hiddens directory"
    )
    parser.add_argument(
        "--test-dirs",
        nargs="+",
        type=str,
        help="Paths to the testing hiddens directories",
    )
    parser.add_argument(
        "--reporter",
        type=str,
        choices=["vincs", "random"],
        default="vincs",
    )
    parser.add_argument("--w-var", type=float, default=0.0)
    parser.add_argument("--w-inv", type=float, default=1.0)
    parser.add_argument("--w-cov", type=float, default=1.0)
    parser.add_argument("--w-supervised", type=float, default=0.0)
    parser.add_argument("--leace-pseudolabels", action="store_true")
    parser.add_argument("--leace-variants", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--label-col",
        type=str,
        choices=["labels", "alice_labels", "bob_labels"],
        default="labels",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dirs = [Path(d) for d in args.test_dirs]

    dtype = torch.float32

    reporter_class = VincsReporter if args.reporter == "vincs" else None

    hiddens_file = (
        "vincs_hiddens.pt"
        if os.path.exists(train_dir / "vincs_hiddens.pt")
        else "ccs_hiddens.pt"
    )
    train_hiddens = torch.load(train_dir / hiddens_file)
    train_n = train_hiddens[0].shape[0]
    d = train_hiddens[0].shape[-1]
    assert all(
        h.shape[0] == train_n for h in train_hiddens
    ), "Mismatched number of samples"
    assert all(h.shape[-1] == d for h in train_hiddens), "Mismatched hidden size"

    train_labels = torch.load(train_dir / f"{args.label_col}.pt").to(args.device).int()
    assert len(train_labels) == train_n, "Mismatched number of labels"

    reporters = []  # one for each layer
    for layer, train_hidden in tqdm(
        enumerate(train_hiddens), desc=f"Training on {train_dir}"
    ):
        if reporter_class is None:
            reporters.append(None)
        else:
            train_hidden = train_hidden.to(args.device).to(dtype)
            if train_hidden.ndim == 3:
                train_hidden = train_hidden.unsqueeze(1)  # now (n, 1, 2, d)
            hidden_size = train_hidden.shape[-1]

            assert train_hidden.ndim == 4

            in_features = 2 * hidden_size
            kwargs = {
                "w_var": args.w_var,
                "w_inv": args.w_inv,
                "w_cov": args.w_cov,
                "w_supervised": args.w_supervised,
            }
            if args.leace_pseudolabels:
                kwargs["leace_pseudolabels"] = True
            if args.leace_variants:
                kwargs["leace_variants"] = True

            reporter: Classifier = reporter_class(
                in_features=in_features, device=args.device, dtype=dtype, **kwargs
            )
            reporter.fit(x=train_hidden, y=train_labels)  # type: ignore
            reporter.resolve_sign(x=train_hidden, y=train_labels)
            reporters.append(reporter)

    if reporters[0] is not None:
        weights = [reporter.linear.weight for reporter in reporters]
        maybe_leace = "_leace" if args.leace_pseudolabels else ""
        maybe_erase_variants = "_erase_variants" if args.leace_variants else ""
        torch.save(
            weights,
            train_dir / f"{args.reporter}_"
            f"{args.w_var}_{args.w_inv}_{args.w_cov}_{args.w_supervised}"
            f"{maybe_leace}{maybe_erase_variants}_reporters.pt",
        )

    with torch.inference_mode():
        for test_dir in test_dirs:
            test_hiddens = torch.load(test_dir / hiddens_file)
            test_labels = (
                torch.load(test_dir / f"{args.label_col}.pt").to(args.device).int()
            )
            lm_log_odds = (
                torch.load(test_dir / "lm_log_odds.pt").to(args.device).to(dtype)
            )

            # make sure that we're using a compatible test set
            test_n = test_hiddens[0].shape[0]
            n_variants = test_hiddens[0].shape[1]
            assert len(test_hiddens) == len(
                train_hiddens
            ), "Mismatched number of layers"
            assert all(
                h.shape[0] == test_n for h in test_hiddens
            ), "Mismatched number of samples"
            assert all(h.shape[-1] == d for h in test_hiddens), "Mismatched hidden size"

            log_odds = {
                k: torch.full(
                    [len(test_hiddens), *shape], torch.nan, device=args.device
                )
                for k, shape in zip(
                    ["none", "partial", "full"],
                    [(test_n, n_variants, 2), (test_n, n_variants), (test_n,)],
                )
            }

            if args.reporter == "random":
                aucs = []
                for layer in range(len(test_hiddens)):
                    auc = eval_random_baseline(
                        train_hiddens[layer],
                        test_hiddens[layer],
                        train_labels,
                        test_labels,
                        num_samples=1000,
                    )
                    if args.verbose:
                        print(f"Layer {layer} random AUC: {auc['mean']}")
                    aucs.append(auc)
                torch.save(
                    aucs,
                    test_dir
                    / f"{train_dir.parent.name}_random_aucs_against_{args.label_col}_full.pt",
                )
            else:
                for layer in tqdm(range(len(reporters)), desc=f"Testing on {test_dir}"):
                    reporter, test_hidden = (
                        reporters[layer],
                        test_hiddens[layer].to(args.device).to(dtype),
                    )
                    if test_hidden.ndim == 3:
                        test_hidden = test_hidden.unsqueeze(1)

                    for ens in log_odds:
                        log_odds[ens][layer] = reporter(test_hidden, ens=ens)
                # save the log odds to disk
                # we use the name of the training directory as the prefix
                # e.g. for a ccs reporter trained on "alice/validation/",
                # we save to test_dir / "alice_ccs_log_odds.pt"[]
                maybe_leace = "_leace" if args.leace_pseudolabels else ""
                maybe_erase_variants = "_erase_variants" if args.leace_variants else ""
                torch.save(
                    log_odds,
                    test_dir / f"{train_dir.parent.name}_{args.reporter}_"
                    f"{args.w_var}_{args.w_inv}_{args.w_cov}_{args.w_supervised}"
                    f"{maybe_leace}{maybe_erase_variants}_log_odds.pt",
                )

                if args.verbose:
                    if len(set(test_labels.cpu().numpy())) != 1:
                        for layer in range(len(reporters)):
                            auc = roc_auc_score(
                                test_labels.cpu().numpy(),
                                log_odds["full"][layer].cpu().numpy(),
                            )
                            print(f"Layer {layer} AUC (full ens):", auc)
                        auc = roc_auc_score(
                            test_labels.cpu().numpy(), lm_log_odds.cpu().numpy()
                        )
                        print("LM AUC:", auc)
