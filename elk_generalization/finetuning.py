from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import torch.nn as nn
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
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from dataloaders import get_dataloader, get_pile_dataloaders
import numpy as np

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        base_model: nn.Module | None,
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
        label_choice_ids: list[int],
        grad_clip: float = 1.0,
        kl_weight: float = 0.3,
        verbose: bool = True,
    ) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = int(os.environ.get("LOCAL_RANK", 0))
        self.base_model_device = (
            self.device + self.world_size
        )  # This assumes world_size <= device_count / 2
        self.model = model.to(self.device)
        self.base_model = (
            base_model.to(self.base_model_device) if kl_weight > 0 else None
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
        self.label_choice_ids = label_choice_ids
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.kl_weight = kl_weight

        # this line resumes training if it was interrupted
        if os.path.exists(self.snapshot_path):
            self._load_snapshot()
        self.model = (
            DDP(self.model, device_ids=[self.device])
            if self.world_size > 1
            else self.model
        )

        self.best_val = float("inf")
        self.epoch = 0

    def _load_snapshot(self):
        if self.verbose and self.device == 0:
            print(f"Loading snapshot from {self.snapshot_path}")
        snapshot = torch.load(self.snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epoch = snapshot["EPOCHS_RUN"]
        self.best_val = snapshot["BEST_VAL"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.scaler.load_state_dict(snapshot["SCALER_STATE"])

    def _save_snapshot(self, to_best_path=False):
        snapshot = dict()
        snapshot["MODEL_STATE"] = (
            self.model.module.state_dict()
            if self.world_size > 1
            else self.model.state_dict()
        )
        snapshot["EPOCHS_RUN"] = self.epoch
        snapshot["BEST_VAL"] = self.best_val
        snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        snapshot["SCHEDULER_STATE"] = self.scheduler.state_dict()
        snapshot["SCALER_STATE"] = self.scaler.state_dict()
        save_path = self.best_checkpoint_path if to_best_path else self.snapshot_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(snapshot, save_path)

    def _train_epoch(self):
        self.model.train()
        for step, (batch, pile_batch) in tqdm(
            enumerate(zip(self.train_data, self.train_pile)),
            total=len(self.train_data),
            position=self.device,
            disable=not self.verbose,
        ):
            self.optimizer.zero_grad()

            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self.model(**batch).loss

            loss.backward()  # we do this before KL loss to avoid modifying model in distributed training
            if self.kl_weight > 0:
                with torch.no_grad():
                    pile_batch_base = {
                        k: v.to(self.base_model_device) for k, v in pile_batch.items()
                    }
                    base_model_output = self.base_model(**pile_batch_base)
                    base_logprobs = (
                        base_model_output.logits[..., -1, :]
                        .log_softmax(dim=-1)
                        .to(self.device)
                    )
                pile_batch_main = {k: v.to(self.device) for k, v in pile_batch.items()}
                model_output = self.model(**pile_batch_main)
                logprobs = model_output.logits[..., -1, :].log_softmax(dim=-1)
                # KL(model || base_model)
                kl_loss = self.kl_weight * (
                    ((logprobs - base_logprobs) * logprobs.exp()).sum(dim=-1).mean()
                )
                kl_loss.backward()
                if self.device == 0:
                    wandb.log({"train/kl_loss": kl_loss.item()})

            if self.device == 0:
                wandb.log({"train/loss": loss.item()})

            if (step + 1) % self.eval_every == 0:
                val_results = self.eval()
                val_score_summary = sum(
                    val_results[(name + "/")]["f1"] for name in self.val_dataloaders
                )

                # save the model if it's the best one so far
                if val_score_summary < self.best_val:
                    self.best_val = val_score_summary
                    self._save_snapshot(to_best_path=True)

                if self.device == 0:
                    # log the results to wandb
                    logs = {
                        "val": val_results,
                        "step": step + len(self.train_data) * self.epoch,
                        "epoch": self.epoch + step / len(self.train_data),
                    }
                    wandb.log(logs)
            if self.verbose:
                print(f"Epoch {self.epoch} step {step} train loss: {loss.item():.3f}")

            if (step + 1) % self.save_every == 0:
                self._save_snapshot()

            # if self.mixed_precision:
            #     self.scaler.scale(loss).backward()
            #     self.scaler.unscale_(self.optimizer)
            #     torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.grad_clip) if self.world_size > 1 else torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            self.optimizer.step()

    def train(self, max_epochs: int):
        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            self._train_epoch()
            self.scheduler.step()

    @torch.no_grad()
    def eval(self):
        self.model.eval()

        results = defaultdict(dict)
        for dataset_name, val_data in self.val_dataloaders.items():
            scores = []
            labels = []
            loss = 0
            for batch in val_data:
                model_inputs = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**model_inputs)
                
                # get the only element that's not -100 in batch["labels"] per row
                target_mask = batch["labels"] != -100
                labs = batch["labels"][target_mask]  # [batch_size,]
                assert len(labs) == model_inputs["input_ids"].shape[0]

                neg_id, pos_id  = self.label_choice_ids
                logprobs = output.logits.log_softmax(dim=-1)
                logp_pos = logprobs[..., pos_id][target_mask]  # [batch_size,]
                logp_neg = logprobs[..., neg_id][target_mask]
                assert len(logp_pos) == len(labs)
                
                scores.append(logp_pos - logp_neg)
                labels.append(labs)

                loss += output.loss / len(val_data)

            scores = torch.cat(scores)
            labels = torch.cat(labels).to(self.device)

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
            preds = (np.exp(scores) > 0.5).astype(int)
            bool_labels = (labels == self.label_choice_ids[1]).astype(int)

            wandb_ds_name = dataset_name + "/"
            # this is a bit redundant to compute results on each process, but very cheap
            results[wandb_ds_name]["f1"] = f1_score(bool_labels, preds, pos_label=1)
            results[wandb_ds_name]["precision"] = precision_score(
                bool_labels, preds, pos_label=1
            )
            results[wandb_ds_name]["recall"] = recall_score(
                bool_labels, preds, pos_label=1
            )
            results[wandb_ds_name]["accuracy"] = accuracy_score(bool_labels, preds)
            results[wandb_ds_name]["auroc"] = roc_auc_score(bool_labels, scores)
            if self.world_size > 1:
                dist.all_reduce(loss)
            results[wandb_ds_name]["loss"] = (loss / self.world_size).item()

        # get CE loss on pile
        loss = 0
        for batch in self.val_pile:
            model_inputs = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**model_inputs)
            loss += output.loss / len(self.val_pile)
        if self.world_size > 1:
            dist.all_reduce(loss)
        results["pile/"]["loss"] = (loss / self.world_size).item()

        if self.verbose and self.device == 0:
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
    if local_rank == 0:
        id = str(uuid.uuid4())
        model_last = args.model.split("/")[-1]
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
    if args.kl_weight > 0:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32
        )
        base_model.eval()
    else:
        base_model = None

    # get train_data, val_datasets, train_pile, val_pile
    train_dl, label_choice_ids = get_dataloader(
        tokenizer,
        args.n_train,
        args.max_len,
        args.batch_size,
        ds_name="atmallen/sloppy_addition_AB_1.0_balanced",
        split="train",
        is_distributed=is_distributed,
    )
    val_ds_names = [
        "atmallen/sloppy_addition_alice_1.0_balanced",
        "atmallen/sloppy_addition_bob_1.0_balanced",
    ]
    val_dls = {
        ds_name.split("/")[-1]: get_dataloader(
            tokenizer,
            args.n_val,
            args.max_len,
            args.batch_size,
            ds_name=ds_name,
            split="validation",
            is_distributed=is_distributed,
        )[0]
        for ds_name in val_ds_names
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

    # setup optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_dl),
    )
    scaler = GradScaler()

    # setup trainer
    trainer = Trainer(
        model=model,
        base_model=base_model,
        train_data=train_dl,
        val_dataloaders=val_dls,
        train_pile=train_pile,
        val_pile=val_pile,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        eval_every=args.eval_every,
        save_every=args.save_every,
        snapshot_path=args.snapshot_path,
        best_checkpoint_path=args.best_checkpoint_path,
        label_choice_ids=label_choice_ids,
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
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--max-pretrain-len", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=200)  # steps
    parser.add_argument("--save-every", type=int, default=200)  # steps
    parser.add_argument("--snapshot-path", type=str, required=True)
    parser.add_argument("--best-checkpoint-path", type=str, required=True)
    parser.add_argument("--pile-path", type=str, required=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.3)
    parser.add_argument("--verbose", action="store_true")
    # e.g. python finetuning.py --model EleutherAI/pythia-410m --snapshot-path ../custom-models/pythia-410m/snapshot.pt \
    #  --best-checkpoint-path ../custom-models/pythia-410m/best.pt --pile-path ../data/pile.jsonl --verbose --max-len 50 --max-pretrain-len 128
    args = parser.parse_args()
    main(args)
