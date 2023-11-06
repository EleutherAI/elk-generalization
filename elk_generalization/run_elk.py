import os
from collections import defaultdict
import argparse
from elk import Elicit, Extract, Eval  # type: ignore
from pathlib import Path
from templates import make_jinja
import importlib


def elicit(
    model: str, from_ds_name: str, num_gpus: int, disable_cache: bool, supervised: str, template: str, max_examples: tuple[int, int], fsdp: bool, min_gpu_mem: int = 1000000000, alpha: float | None = None, l1_ratio: float = 0) -> str:
    """Runs ELK elicit and eval for a given model specified by `base_name` and `version`.
    Trains a probe on each of "bob" and "alice" datasets, and evaluates transfer accuracy on the other.
    Saves results to ./elk-generalization/{model}/{from_ds_name}/elicit/{to_ds_name}
    """

    model_last = model.split("/")[-1]
    out_dir = f"{model_last}/{from_ds_name}"
    full_out_dir = os.path.join(elk_reporter_dir(), out_dir)

    elicit = Elicit(
        data=Extract(
            model=model,
            datasets=(from_ds_name,),
            max_examples=max_examples,
            fsdp=fsdp,
            template_path=template,
            balance=False,
        ),
        num_gpus=num_gpus,
        out_dir=Path(full_out_dir),
        debug=True,
        disable_cache=disable_cache,
        supervised=supervised,
        alpha=alpha,
        l1_ratio=l1_ratio,
        save_logprobs=True,
        min_gpu_mem=min_gpu_mem,  # 1 GB
    )
    elicit.execute()

    return out_dir


def eval(
    model: str,
    from_out_dir: str,
    to_ds_names: list[str],
    num_gpus: int,
    disable_cache: bool,
    template: str,
    max_examples: tuple[int, int],
    fsdp: bool,
    min_gpu_mem: int = 1000000000,
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
                max_examples=max_examples,
                fsdp=fsdp,
                template_path=template,
                balance=False,
            ),
            source=Path(from_out_dir),
            num_gpus=num_gpus,
            debug=True,
            save_logprobs=True,
            disable_cache=disable_cache,
            min_gpu_mem=min_gpu_mem
        )
        eval.execute()

        out_dirs.append(
            os.path.join(elk_reporter_dir(), from_out_dir, "transfer", to_ds_name)
        )

    return out_dirs


def elk_reporter_dir():
    """Returns the path to the ELK reporter directory."""
    # this copies the behavior from EleutherAI/elk
    return os.environ.get("ELK_DIR", Path.home() / "elk-reporters")


def transfer(args, datasets, both_ways=False):
    out_dirs = defaultdict(dict)  # from_dataset -> {to_dataset -> out_dir}
    assert len(datasets) == 2
    keys = list(datasets.keys())

    elk_kwargs = {
        "num_gpus": args.num_gpus,
        "template": f"qm_{args.template}",
        "disable_cache": args.disable_cache,
        "max_examples": args.max_examples,
        "fsdp": args.fsdp and args.num_gpus > 1,
        "min_gpu_mem": args.min_gpu_mem,
    }
    pairs = [(keys[0], keys[1]), (keys[1], keys[0])] if both_ways else [(keys[0], keys[1])]
    for fr, to in pairs:
        from_dataset = datasets[fr]
        to_dataset = datasets[to]

        # train probe on from_dataset
        from_out_dir = elicit(
            args.model, from_dataset, supervised=args.supervised, alpha=args.alpha, l1_ratio=args.l1_ratio, **elk_kwargs
        )

        # eval probe on both datasets
        transfer_out_dirs = eval(
            args.model,
            from_out_dir,
            [from_dataset, to_dataset],
            **elk_kwargs
        )

        out_dirs[fr][fr] = transfer_out_dirs[0]
        out_dirs[fr][to] = transfer_out_dirs[1]

    print(dict(out_dirs))
    return out_dirs


def get_elk_templates_dir():
    """Returns the path to the ELK templates directory."""
    # Use importlib to get the module information
    module_origin = importlib.util.find_spec("elk").origin  # type: ignore
    module_path = Path(module_origin).parent
    return str(module_path / "promptsource" / "templates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["context_generalization", "easy_vs_hard", "alice_easy_to_bob_hard", "alice_hard_to_bob_hard"],
        default="context_generalization",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/ssd-1/alexm/elk-generalization/custom-models/Llama-2-7b-hf-v1692471371",
    )
    parser.add_argument("--template", choices=["mixture", "grader_first", "grader_last"], type=str, required=True)
    parser.add_argument("--max-examples", type=int, nargs=2, default=[1000, 1000])
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--supervised", type=str, default="single")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--l1-ratio", type=float, default=0.0)
    parser.add_argument("--min-gpu-mem", type=int, default=1000000000)  # 1 GB
    parser.add_argument("--p-err", type=float, default=1.0)

    args = parser.parse_args()

    # make sure the appropriate templates are made
    make_jinja(args.template, get_elk_templates_dir())
    if args.experiment == "context_generalization":
        datasets = {
            "alice": f"atmallen/qm_alice_{float(args.p_err)}e_eval",
            "bob": f"atmallen/qm_bob_{float(args.p_err)}e_eval",
        }
        both_ways = True
    elif args.experiment == "easy_vs_hard":
        datasets = {
            "easy": f"atmallen/qm_alice_easy_2_{float(args.p_err)}e_eval",
            "hard": f"atmallen/qm_alice_hard_4_{float(args.p_err)}e_eval",
        }
        both_ways = True
    elif args.experiment == "alice_easy_to_bob_hard":
        datasets = {
            "alice_easy": f"atmallen/qm_alice_easy_2_{float(args.p_err)}e_eval",
            "bob_hard": f"atmallen/qm_bob_hard_4_{float(args.p_err)}e_eval",
        }
        both_ways = False
    elif args.experiment == "alice_hard_to_bob_hard":
        datasets = {
            "alice_hard": f"atmallen/qm_alice_hard_4_{float(args.p_err)}e_eval",
            "bob_hard": f"atmallen/qm_bob_hard_4_{float(args.p_err)}e_eval",
        }
        both_ways = False
    else:
        raise NotImplementedError
    transfer(args, datasets, both_ways=both_ways)
