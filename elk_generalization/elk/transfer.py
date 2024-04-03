import argparse
from pathlib import Path

import torch
from classifier import Classifier
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from elk_generalization.elk.ccs import CcsConfig, CcsReporter
from elk_generalization.elk.crc import CrcReporter
from elk_generalization.elk.lda import LdaReporter
from elk_generalization.elk.lr_classifier import LogisticRegression
from elk_generalization.elk.mean_diff import MeanDiffReporter
from elk_generalization.elk.random_baseline import eval_random_baseline

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
        choices=[
            "ccs",
            "crc",
            "lr",
            "lr-on-pair",
            "lda",
            "mean-diff",
            "mean-diff-on-pair",
            "random",
        ],
        default="lr",
    )
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

    use_cp = args.reporter in {"ccs", "crc", "lr-on-pair", "mean-diff-on-pair"}

    reporter_class = {
        "ccs": CcsReporter,
        "crc": CrcReporter,
        "lr": LogisticRegression,
        "lr-on-pair": LogisticRegression,
        "lda": LdaReporter,
        "mean-diff": MeanDiffReporter,
        "mean-diff-on-pair": MeanDiffReporter,
        "random": None,
    }[args.reporter]

    hiddens_file = "ccs_hiddens.pt" if use_cp else "hiddens.pt"
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
        train_hidden = train_hidden.to(args.device).to(dtype)
        hidden_size = train_hidden.shape[-1]

        if args.reporter == "ccs":
            kwargs = dict(
                cfg=CcsConfig(
                    bias=True,
                    loss=["ccs"],
                    norm="leace",
                    lr=1e-2,
                    num_epochs=1000,
                    num_tries=10,
                    optimizer="lbfgs",
                    weight_decay=0.01,
                ),
                num_variants=1,
            )
        else:
            kwargs = {}

        if use_cp:
            assert train_hidden.ndim == 3
            train_hidden = train_hidden.view(
                train_hidden.shape[0], -1
            )  # cat positive and negative
            in_features = 2 * hidden_size
        else:
            in_features = hidden_size

        if args.reporter == "random":
            reporters.append(None)
        else:
            reporter: Classifier = reporter_class(
                in_features=in_features, device=args.device, dtype=dtype, **kwargs
            )
            reporter.fit(x=train_hidden, y=train_labels)
            reporter.resolve_sign(x=train_hidden, y=train_labels)
            reporters.append(reporter)

    if reporters[0] is not None:
        weights = [reporter.linear.weight for reporter in reporters]
        torch.save(weights, train_dir / f"{args.reporter}_reporters.pt")

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
            assert len(test_hiddens) == len(
                train_hiddens
            ), "Mismatched number of layers"
            assert all(
                h.shape[0] == test_n for h in test_hiddens
            ), "Mismatched number of samples"
            assert all(h.shape[-1] == d for h in test_hiddens), "Mismatched hidden size"

            log_odds = torch.full(
                [len(test_hiddens), test_n], torch.nan, device=args.device
            )
            for layer in tqdm(range(len(reporters)), desc=f"Testing on {test_dir}"):
                reporter, test_hidden = (
                    reporters[layer],
                    test_hiddens[layer].to(args.device).to(dtype),
                )
                if args.reporter == "ccs":
                    test_hidden = test_hidden.unsqueeze(1)
                    log_odds[layer] = reporter(test_hidden, ens="full")
                elif args.reporter == "crc":
                    log_odds[layer] = reporter(test_hidden)
                elif (
                    args.reporter == "lr-on-pair"
                    or args.reporter == "mean-diff-on-pair"
                ):
                    test_hidden = test_hidden.view(
                        test_hidden.shape[0], -1
                    )  # cat positive and negative
                    log_odds[layer] = reporter(test_hidden).squeeze(-1)
                elif args.reporter != "random":
                    log_odds[layer] = reporter(test_hidden).squeeze(-1)

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
                    / f"{train_dir.parent.name}_random_aucs_against_{args.label_col}.pt",
                )
            else:
                # save the log odds to disk
                # we use the name of the training directory as the prefix
                # e.g. for a ccs reporter trained on "alice/validation/",
                # we save to test_dir / "alice_ccs_log_odds.pt"[]
                torch.save(
                    log_odds,
                    test_dir / f"{train_dir.parent.name}_{args.reporter}_log_odds.pt",
                )

                if args.verbose:
                    if len(set(test_labels.cpu().numpy())) != 1:
                        for layer in range(len(reporters)):
                            auc = roc_auc_score(
                                test_labels.cpu().numpy(), log_odds[layer].cpu().numpy()
                            )
                            print("AUC:", auc)
                        auc = roc_auc_score(
                            test_labels.cpu().numpy(), lm_log_odds.cpu().numpy()
                        )
                        print("LM AUC:", auc)
                    else:
                        print(
                            f"All labels are the same for {test_dir}! Using accuracy instead."
                        )
                        for layer in range(len(reporters)):
                            acc = accuracy_score(
                                test_labels.cpu().numpy(),
                                log_odds[layer].cpu().numpy() > 0,
                            )
                            print("ACC:", acc)
                        acc = accuracy_score(
                            test_labels.cpu().numpy(), lm_log_odds.cpu().numpy() > 0
                        )
                        print("LM ACC:", acc)
