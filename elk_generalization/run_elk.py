import os
from collections import defaultdict
import argparse
from pathlib import Path
from templates import make_jinja
import importlib


def elicit(
    model: str, from_ds_name: str, ccs: bool, max_examples: tuple[int, int], fsdp: bool, template: str, **kwargs) -> str:
    """Runs ELK elicit and eval for a given model specified by `base_name` and `version`.
    Trains a probe on each of "bob" and "alice" datasets, and evaluates transfer accuracy on the other.
    Saves results to ./elk-generalization/{model}/{from_ds_name}/elicit/{to_ds_name}
    """

    model_last = model.split("/")[-1]
    out_dir = f"{model_last}/{from_ds_name}"
    full_out_dir = os.path.join(reporter_dir(ccs), out_dir)

    maybe_fsdp = {} if ccs else {"fsdp": fsdp}
    elicit = Elicit(
        data=Extract(
            model=model,
            datasets=(from_ds_name,),
            max_examples=max_examples,
            template_path=template,
            balance=False,
            **maybe_fsdp
        ),
        save_logprobs=True,
        debug=True,
        out_dir=Path(full_out_dir),
        **kwargs
    )
    elicit.execute()

    return out_dir


def eval(
    model: str,
    from_out_dir: str,
    to_ds_names: list[str],
    ccs: bool,
    template: str,
    max_examples: tuple[int, int],
    fsdp: bool,
    **kwargs
):
    """Evaluates a probe trained on `from_out_dir` on each of `to_ds_names`.
    Saves results
    """

    maybe_fsdp = {} if ccs else {"fsdp": fsdp}
    out_dirs = []
    for to_ds_name in to_ds_names:
        eval = Eval(
            data=Extract(
                model=model,
                datasets=(to_ds_name,),
                max_examples=max_examples,
                template_path=template,
                balance=False,
                **maybe_fsdp
            ),
            source=Path(from_out_dir),
            save_logprobs=True,
            debug=True,
            **kwargs
        )
        eval.execute()

        out_dirs.append(
            os.path.join(reporter_dir(ccs), from_out_dir, "transfer", to_ds_name)
        )

    return out_dirs


def reporter_dir(ccs: bool):
    """Returns the path to the ELK/CCS reporter directory."""
    # this copies the behavior from EleutherAI/elk and EleutherAI/ccs
    module_name = "ccs" if ccs else "elk"
    return os.environ.get(f"{module_name.upper()}_DIR", Path.home() / f"{module_name}-reporters")

def transfer(args, datasets, both_ways=False):
    out_dirs = defaultdict(dict)  # from_dataset -> {to_dataset -> out_dir}
    assert len(datasets) == 2
    keys = list(datasets.keys())

    kwargs = {
        "ccs": args.ccs,
        "num_gpus": args.num_gpus,
        "template": f"qm_{args.template}",
        "disable_cache": args.disable_cache,
        "max_examples": args.max_examples,
        "fsdp": args.fsdp,
        "min_gpu_mem": args.min_gpu_mem,
    }
    elk_train_args = {"alpha": args.alpha} if not args.ccs else {}
    if args.l1_ratio > 0:
        elk_train_args["l1_ratio"] = args.l1_ratio

    pairs = [(keys[0], keys[1]), (keys[1], keys[0])] if both_ways else [(keys[0], keys[1])]
    for fr, to in pairs:
        from_dataset = datasets[fr]
        to_dataset = datasets[to]

        # train probe on from_dataset
        from_out_dir = elicit(
            args.model, from_dataset, supervised=args.supervised, **elk_train_args, **kwargs
        )

        # eval probe on both datasets
        transfer_out_dirs = eval(
            args.model,
            from_out_dir,
            [from_dataset, to_dataset],
            **kwargs
        )

        out_dirs[fr][fr] = transfer_out_dirs[0]
        out_dirs[fr][to] = transfer_out_dirs[1]

    print(dict(out_dirs))
    return out_dirs


def get_templates_dir(module_name: str):
    """Returns the path to the ELK templates directory."""
    # Use importlib to get the module information
    module_origin = importlib.util.find_spec(module_name).origin  # type: ignore
    module_path = Path(module_origin).parent
    return str(module_path / "promptsource" / "templates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default="A->B",
        help="Indicate the source and target distributions of the form <source>-><target>." \
        "Choices: [A, AE, AH, B, BE, BH], for combinations of Alice, Bob, Easy and Hard." \
        "Use <-> to run both directions."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/ssd-1/alexm/elk-generalization/custom-models/Llama-2-7b-hf-v1692471371",
    )
    parser.add_argument("--template", choices=["mixture", "grader_first", "grader_last"], type=str, required=True)
    parser.add_argument("--ccs", action="store_true")
    parser.add_argument("--max-examples", type=int, nargs=2, default=[1000, 1000])
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--supervised", type=str, default="single")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--l1-ratio", type=float, default=0.0)
    parser.add_argument("--label-column", type=str, default=None)
    parser.add_argument("--min-gpu-mem", type=int, default=1000000000)  # 1 GB
    parser.add_argument("--p-err", type=float, default=1.0)

    args = parser.parse_args()

    if args.ccs:
        from ccs import Elicit, Extract, Eval  # type: ignore
        module_name = "ccs"
    else:
        from elk import Elicit, Extract, Eval  # type: ignore
        module_name = "elk"


    # make sure the appropriate templates are made
    make_jinja(args.template, args.label_column, args.ccs, get_templates_dir(module_name=module_name))
    
    distrs = {  
        "all": "",
        "A": "alice_",
        "AE": "alice_easy_2_",
        "AH": "alice_hard_4_",
        "B": "bob_",
        "BE": "bob_easy_2_",
        "BH": "bob_hard_4_",
    }

    try:
        if "<->" in args.experiment:
            both_ways = True
            args.experiment = args.experiment.replace("<->", "->")
        else:
            both_ways = False
        source, target = args.experiment.split("->")
        assert source in distrs and target in distrs
        source, target = distrs[source], distrs[target]
    except ValueError:
        raise ValueError("Experiment must be of the form <source>-><target>")
    
    datasets = {
        source: f"atmallen/qm_{source}{float(args.p_err)}e_eval",
        target: f"atmallen/qm_{target}{float(args.p_err)}e_eval",
    }
    transfer(args, datasets, both_ways=both_ways)
