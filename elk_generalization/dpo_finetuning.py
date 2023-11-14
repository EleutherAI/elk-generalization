from argparse import ArgumentParser
from datasets import load_dataset, DatasetDict
from peft import LoraConfig  # type: ignore
import torch
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import random
import numpy as np

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    ds: DatasetDict = load_dataset(  # type: ignore
        args.dataset,
    ).rename_column(
        'statement', 'prompt'
    ).map(
        lambda x: {
            'chosen': x['choices'][x['label']],
            'rejected': x['choices'][1 - x['label']],
        },
        remove_columns=['choices', 'label', 'true_label']
    ).shuffle(args.seed)

    trainer = DPOTrainer(
        model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto"),
        args=TrainingArguments(
            args.output_dir or f"checkpoints/{args.name}",
            fp16=args.fp16,
            gradient_accumulation_steps=args.grad_accumulation_steps,
            logging_steps=args.eval_every,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            remove_unused_columns=False,
            run_name=args.name,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            eval_steps=args.eval_every,
            save_steps=args.eval_every,
            max_grad_norm=args.grad_clip,
            learning_rate=args.lr,
            adam_beta2=0.95,
            save_total_limit=1,
            load_best_model_at_end=True,  # the best checkpoint should also be retained
            seed=args.seed,
        ),
        max_length=512,
        max_prompt_length=args.max_len,
        peft_config=(
            LoraConfig( # type: ignore
                r=args.lora_rank, target_modules=args.lora_modules
            )
            if args.lora_rank > 0 else None
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, help="Experiment name for wandb")
    parser.add_argument(
        "--dataset", type=str, default="atmallen/qm_mixture_1.0e_0.5p_finetuning",
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        nargs="+",
        default=["gate_proj", "down_proj", "up_proj", "q_proj", "k_proj", "v_proj"],
    )
    parser.add_argument(
        "--lora-rank", type=int, default=8, 
        help="Non-positive value disables LoRA and performs full finetuning",
    )
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--grad-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)  # steps
    parser.add_argument("--output-dir", type=str, default="../custom-models")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=633)
    args = parser.parse_args()

    # make reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # type: ignore

    args = parser.parse_args()
    main(args)
