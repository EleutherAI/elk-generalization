from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler  # type: ignore
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_  # type: ignore
import torch
import os
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from collections import defaultdict
import wandb
import uuid
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from dataloaders import get_dataloader, get_pile_dataloaders
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType, PeftType, PeftModel  # type: ignore
import json

# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        base_model: nn.Module | None,
        devices: tuple[int, int] | None,
        train_data: DataLoader,
        val_dataloaders: dict[str, DataLoader],
        train_pile: DataLoader | None,
        val_pile: DataLoader,
        optimizer,
        scheduler,
        scaler,
        eval_every: int,
        save_every: int,
        snapshot_path: str,
        best_checkpoint_path: str,
        grad_clip: float = 1.0,
        kl_weight: float = 0.3,
        verbose: bool = True,
    ) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.world_size > 1 and devices is not None:
            raise ValueError("devices must be None in distributed training")
        self.device = int(os.environ.get("LOCAL_RANK", 0)) if devices is None else devices[0]
        self.base_model_device = (
            self.device + self.world_size
        ) if devices is None else devices[1]
        self.model = model.to(self.device)
        self.base_model = (
            base_model.to(self.base_model_device) if kl_weight > 0 else None  # type: ignore
        )
        self.train_data = train_data
        self.val_dataloaders = val_dataloaders
        self.train_pile = train_pile or [None] * len(train_data)
        self.val_pile = val_pile
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.eval_every = eval_every
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.best_checkpoint_path = best_checkpoint_path
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.kl_weight = kl_weight

        # this line resumes training if it was interrupted
        if os.path.exists(self.snapshot_path):
            self._load_snapshot()

        if self.world_size > 1 and isinstance(self.model, PeftModel):
            raise ValueError("PEFT is not supported in distributed training")

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device])
        else:
            self.model = self.model.to(self.device)

        self.best_val = float("-inf")
        self.epoch = 0

    def _load_snapshot(self):
        if self.verbose and self.device == 0 or self.world_size == 1:
            print(f"Loading snapshot from {self.snapshot_path}")
        snapshot = torch.load(self.snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epoch = snapshot["EPOCHS_RUN"]
        self.best_val = snapshot["BEST_VAL"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.scaler.load_state_dict(snapshot["SCALER_STATE"])

    def _save_snapshot(self):
        snapshot = dict()
        snapshot["MODEL_STATE"] = (
            self.model.module.state_dict()  # type: ignore
            if self.world_size > 1
            else self.model.state_dict()
        )
        snapshot["EPOCHS_RUN"] = self.epoch
        snapshot["BEST_VAL"] = self.best_val
        snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        snapshot["SCHEDULER_STATE"] = self.scheduler.state_dict()
        snapshot["SCALER_STATE"] = self.scaler.state_dict()
        torch.save(snapshot, self.snapshot_path)
    
    def _save_checkpoint(self):
        print(f"Saving checkpoint to {self.best_checkpoint_path}")
        model = self.model.module if self.world_size > 1 else self.model
        model.save_pretrained(self.best_checkpoint_path)  # type: ignore

    def _train_epoch(self):
        self.model.train()
        for step, (batch, pile_batch) in tqdm(
            enumerate(zip(self.train_data, self.train_pile)),
            total=len(self.train_data),
            position=self.device,
            disable=not self.verbose,
        ):
            if (step + 1) % self.eval_every == 0 or step == len(self.train_data) - 1:
                val_results = self.eval()
                val_score_summary = sum(
                    val_results[(name + "/")]["auroc"] for name in self.val_dataloaders
                )

                # save the model if it's the best one so far
                if val_score_summary >= self.best_val and (self.device == 0 or self.world_size == 1):
                    self.best_val = val_score_summary
                    self._save_checkpoint()

                if self.device == 0 or self.world_size == 1:
                    # log the results to wandb
                    logs = {
                        **val_results,
                        "step": step + len(self.train_data) * self.epoch,
                        "epoch": self.epoch + step / len(self.train_data),
                    }
                    wandb.log(logs)
            self.optimizer.zero_grad()

            model_inputs = {k: batch[k].to(self.device) for k in ["input_ids", "attention_mask", "labels"]}
            loss = self.model(**model_inputs).loss

            # we do this before KL loss to avoid modifying model in distributed training
            self.scaler.scale(loss).backward()
            if self.kl_weight > 0:
                with torch.no_grad():
                    pile_batch_base = {
                        k: v.to(self.base_model_device) for k, v in pile_batch.items()
                    }
                    base_model_output = self.base_model(**pile_batch_base)  # type: ignore
                    base_logprobs = (
                        base_model_output.logits.log_softmax(dim=-1)
                        .to(self.device)
                        .detach()
                    )
                pile_batch_main = {k: v.to(self.device) for k, v in pile_batch.items()}
                model_output = self.model(**pile_batch_main)
                logprobs = model_output.logits.log_softmax(dim=-1)
                # KL(model || base_model)
                kl_loss = self.kl_weight * (
                    ((logprobs - base_logprobs) * logprobs.exp()).sum(dim=-1).mean()
                )
                self.scaler.scale(kl_loss).backward()
                if self.device == 0 or self.world_size == 1:
                    wandb.log({"train/kl_loss": kl_loss.item()})

            clip_grad_norm_(self.model.module.parameters(), self.grad_clip) if self.world_size > 1 else clip_grad_norm_(self.model.parameters(), self.grad_clip)  # type: ignore
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if self.device == 0 or self.world_size == 1:
                wandb.log({"train/loss": loss.item()})

            if (step + 1) % self.save_every == 0 and (self.device == 0 or self.world_size == 1):
                self._save_snapshot()

    def train(self, max_epochs: int):
        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            self._train_epoch()

    @torch.no_grad()
    def eval(self):
        self.model.eval()

        results = defaultdict(dict)
        for dataset_name, val_data in self.val_dataloaders.items():
            scores = []
            labels = []
            loss = 0
            for batch_num, batch in enumerate(val_data):
                model_inputs = {k: batch[k].to(self.device) for k in ["input_ids", "attention_mask", "labels"]}
                output = self.model(**model_inputs)

                batch_size = model_inputs["input_ids"].shape[0]

                # get the only element that's not -100 in batch["labels"] per row
                target_mask = batch["labels"][..., 1:] != -100
                labs = batch["labels"][..., 1:][target_mask]  # [batch_size,]
                assert len(labs) == batch_size

                neg_ids, pos_ids = batch["choice_ids"][:, 0], batch["choice_ids"][:, 1]
                logprobs = output.logits.log_softmax(dim=-1)[..., :-1, :]
                # this indexing is tricky, since we want a different sequence position and token id for each row in the batch
                logp_pos = torch.stack([logprobs[row, :, pos_id] for row, pos_id in enumerate(pos_ids)])[target_mask]  # [batch_size,]
                logp_neg = torch.stack([logprobs[row, :, neg_id] for row, neg_id in enumerate(neg_ids)])[target_mask]  # [batch_size,]
                assert len(logp_pos) == len(labs)

                scores.append(logp_pos - logp_neg)
                labels.append(torch.tensor([l == pos_id for l, pos_id in zip(labs, pos_ids)], dtype=bool).to(self.device))  # type: ignore

                loss += output.loss

            loss /= batch_num + 1  # type: ignore

            scores = torch.cat(scores)
            labels = torch.cat(labels)

            # gather results from processes
            if self.world_size > 1:
                global_scores = [
                    torch.zeros_like(scores) for _ in range(self.world_size)
                ]
                global_labels = [
                    torch.zeros_like(labels) for _ in range(self.world_size)
                ]
                dist.all_gather(global_scores, scores)
                dist.all_gather(global_labels, labels)
                scores = torch.cat(global_scores).cpu().numpy()
                labels = torch.cat(global_labels).cpu().numpy()
            else:
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
            preds = (scores > np.log(0.5)).astype(int)

            wandb_ds_name = dataset_name + "/"
            # this is a bit redundant to compute results on each process, but very cheap
            results[wandb_ds_name]["f1"] = f1_score(labels, preds, pos_label=1)
            results[wandb_ds_name]["precision"] = precision_score(
                labels, preds, pos_label=1
            )
            results[wandb_ds_name]["recall"] = recall_score(
                labels, preds, pos_label=1
            )
            results[wandb_ds_name]["accuracy"] = accuracy_score(labels, preds)
            results[wandb_ds_name]["auroc"] = roc_auc_score(labels, scores)
            if self.world_size > 1:
                dist.all_reduce(loss)
            results[wandb_ds_name]["loss"] = (loss / self.world_size).item()  # type: ignore

        # get CE loss on pile
        loss = 0
        for batch in self.val_pile:
            model_inputs = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**model_inputs)
            loss += output.loss / len(self.val_pile)
        if self.world_size > 1:
            dist.all_reduce(loss)
        results["pile/"]["loss"] = (loss / self.world_size).item()  # type: ignore

        if self.verbose and (self.device == 0 or self.world_size == 1):
            print(f"Epoch {self.epoch} validation results:")
            for dataset_name, dataset_results in results.items():
                print(f"  {dataset_name}:")
                for metric_name, metric_value in dataset_results.items():
                    print(f"    {metric_name}: {metric_value:.3f}")

        return results


def main(args):
    # check if this is a distributed training run
    is_distributed = "WORLD_SIZE" in os.environ
    if is_distributed:
        init_process_group(backend="nccl")

    cfg = vars(args)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # hash the args to get a unique id for this run
    id = str(hash(str(cfg)))[-8:]
    model_last = args.model.split("/")[-1]
    save_dir = os.path.join(args.save_dir, f"{model_last}-{id}")
    snapshot_path = os.path.join(save_dir, "snapshot.pt")
    best_checkpoint_path = os.path.join(save_dir, "best")
    if local_rank == 0:
        print(f"Using save dir {save_dir}")
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        os.makedirs(best_checkpoint_path, exist_ok=True)
        # write the config to a file
        with open(os.path.join(best_checkpoint_path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        wandb.login()
        wandb.init(
            project="sloppy-addition",
            name=f"{model_last}-{id}",
            resume=False,
            config=cfg,
        )

    # setup models
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, add_prefix_space=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    if args.lora_rank != -1:
        if args.lora_modules is None:
            model_cls = model.__class__.__name__
            if "llama" in model_cls.lower():
                args.lora_modules = [
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                ]
            elif "gptneox" in model_cls.lower():
                args.lora_modules = [
                    "dense_h_to_4h",
                    "dense_4h_to_h",
                    "query_key_value",
                ]
            else:
                raise ValueError(
                    f"Target modules not specified for model class `{model_cls}`"
                )

        model = model.half()
        peft_config = LoraConfig(
            peft_type=PeftType.LORA,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=args.lora_modules,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        if args.verbose:
            model.print_trainable_parameters()

    if args.kl_weight > 0:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32
        )
        base_model.eval()
    else:
        base_model = None

    # get train_data, val_datasets, train_pile, val_pile
    train_dl = get_dataloader(
        tokenizer,
        args.n_train,
        args.max_len,
        args.batch_size,
        ds_name=args.train_ds_name,
        split="train",
        is_distributed=is_distributed,
    )
    val_dls = {
        "val/"
        + ds_name.split("/")[-1]: get_dataloader(
            tokenizer,
            args.n_val,
            args.max_len,
            args.batch_size,
            ds_name=ds_name,
            split="validation",
            is_distributed=is_distributed,
        )
        for ds_name in args.val_ds_names
    }
    val_dls = {
        f"train/{args.train_ds_name.split('/')[-1]}": get_dataloader(
            tokenizer,
            args.n_val,
            args.max_len,
            args.batch_size,
            ds_name=args.train_ds_name,
            split="train",
            is_distributed=is_distributed,
        ),
        **val_dls,
    }

    train_pile, val_pile = get_pile_dataloaders(
        tokenizer,
        args.n_train,
        args.n_val,
        args.max_pretrain_len,
        args.batch_size,
        jsonl_path=args.pile_path,
        is_distributed=is_distributed,
    )

    # setup optimizer, scheduler, scale
    learnable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        learnable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_dl),
    )
    scaler = GradScaler()

    # setup trainer
    trainer = Trainer(
        model=model,
        base_model=base_model,
        devices=args.devices,
        train_data=train_dl,
        val_dataloaders=val_dls,
        train_pile=train_pile,
        val_pile=val_pile,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        eval_every=args.eval_every,
        save_every=args.save_every,
        snapshot_path=snapshot_path,
        best_checkpoint_path=best_checkpoint_path,
        grad_clip=args.grad_clip,
        kl_weight=args.kl_weight,
        verbose=args.verbose,
    )
    trainer.train(args.epochs)

    if local_rank == 0:
        wandb.finish()

    if is_distributed:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train-ds-name", type=str, required=True)
    parser.add_argument("--val-ds-names", type=str, nargs="+", required=True)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--max-pretrain-len", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=200)  # steps
    parser.add_argument("--save-every", type=int, default=200)  # steps
    parser.add_argument("--save-dir", type=str, default="../custom-models")
    parser.add_argument("--pile-path", type=str, required=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.3)
    parser.add_argument("--lora-rank", type=int, default=-1)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-modules", type=str, nargs="+")
    parser.add_argument("--devices", type=int, nargs="+")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=633)

    import random
    import numpy as np

    random.seed(633)
    np.random.seed(633)
    torch.manual_seed(633)
    torch.cuda.manual_seed(633)
    torch.cuda.manual_seed_all(633)
    torch.backends.cudnn.deterministic = True  # type: ignore

    args = parser.parse_args()
    main(args)
