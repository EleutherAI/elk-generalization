import os
from collections import defaultdict
import argparse
from elk import Elicit, Extract, Eval  # type: ignore
from pathlib import Path


def elicit(model: str, from_ds_name: str, num_gpus=1, disable_cache=False):
    """Runs ELK elicit and eval for a given model specified by `base_name` and `version`.
    Trains a probe on each of "bob" and "alice" datasets, and evaluates transfer accuracy on the other.
    Saves results to ./elk-generalization/{model}/{from_ds_name}/elicit/{to_ds_name}
    """

    model_last = model.split("/")[-1]
    out_dir = f"{model_last}/{from_ds_name}"
    full_out_dir = os.path.join(os.environ["ELK_DIR"], out_dir)

    elicit = Elicit(
        data=Extract(
            model=model,
            datasets=(from_ds_name,),
            max_examples=(1000, 1000),
            template_path="_no_suffix",
            fsdp=num_gpus > 1,
        ),
        num_gpus=num_gpus,
        out_dir=Path(full_out_dir),
        debug=True,
        disable_cache=disable_cache,
        supervised="single",
        save_logprobs=True,
        min_gpu_mem=1000000000,  # 1 GB
    )
    elicit.execute()

    return out_dir


def eval(
    model: str, from_out_dir: str, to_ds_names: list[str], num_gpus=1, disable_cache=False
):
    """Evaluates a probe trained on `from_out_dir` on each of `to_ds_names`.
    Saves results
    """

    out_dirs = []
    for to_ds_name in to_ds_names:
        eval = Eval(
            data=Extract(
                model=model,
                datasets=(to_ds_name,),
                max_examples=(1000, 1000),
                template_path="_no_suffix",
                fsdp=True,
            ),
            source=Path(from_out_dir),
            num_gpus=num_gpus,
            debug=True,
            save_logprobs=True,
            disable_cache=disable_cache,
            min_gpu_mem=1000000000,  # 1 GB
        )
        eval.execute()

        out_dirs.append(
            os.path.join(
                os.environ["ELK_DIR"], from_out_dir, "transfer", to_ds_name
            )
        )

    return out_dirs


def context_generalization(model, p_err, num_gpus):
    out_dirs = defaultdict(dict)  # from_dataset -> {to_dataset -> out_dir}
    for fr, to in [("alice", "bob"), ("bob", "alice")]:
        from_dataset = f"atmallen/sloppy_addition_{fr}_{p_err}"
        to_dataset = f"atmallen/sloppy_addition_{to}_{p_err}"

        # train probe on from_dataset
        from_out_dir = elicit(model, from_dataset, num_gpus=num_gpus)

        # eval probe on both datasets
        transfer_out_dirs = eval(model, from_out_dir, [from_dataset, to_dataset], num_gpus=num_gpus)

        out_dirs[fr][fr] = transfer_out_dirs[0]
        out_dirs[fr][to] = transfer_out_dirs[1]

    print(dict(out_dirs))
    return out_dirs


def easy_vs_hard(model, p_err, num_gpus):
    out_dirs = defaultdict(dict)  # from_dataset -> {to_dataset -> out_dir}
    datasets = {
        "easy": f"atmallen/sloppy_addition_alice_{p_err}_easy_2",
        "hard": f"atmallen/sloppy_addition_alice_{p_err}_hard_4",
    }
    for to, fr in [("easy", "hard"), ("hard", "easy")]:
        from_dataset = datasets[fr]
        to_dataset = datasets[to]

        # train probe on from_dataset
        from_out_dir = elicit(model, from_dataset, num_gpus=num_gpus)

        # eval probe on both datasets
        transfer_out_dirs = eval(model, from_out_dir, [from_dataset, to_dataset], num_gpus=num_gpus)

        out_dirs[fr][fr] = transfer_out_dirs[0]
        out_dirs[fr][to] = transfer_out_dirs[1]

    print(dict(out_dirs))
    return out_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["context_generalization", "easy_vs_hard"],
        default="context_generalization",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/ssd-1/alexm/elk-generalization/custom-models/Llama-2-7b-hf-v1692471371",
    )
    parser.add_argument("--p-err", type=float, default=1.0)
    parser.add_argument("--num-gpus", type=int, default=1)

    args = parser.parse_args()
    if args.experiment == "context_generalization":
        context_generalization(args.model, args.p_err, args.num_gpus)
    elif args.experiment == "easy_vs_hard":
        easy_vs_hard(args.model, args.p_err, args.num_gpus)
