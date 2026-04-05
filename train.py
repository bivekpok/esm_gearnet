"""
train.py - One fold training with proper train/val/test split

Data roles:
  train_manifest.csv  -> model sees gradients from this
  valid_manifest.csv  -> early stopping, scheduler, checkpointing (NO gradients)
  test_manifest.csv   -> final evaluation only AFTER training (run evaluate.py)

Launch:
  Single GPU : python -u train.py --outer 1 --inner 1
  Multi-GPU  : torchrun --nproc_per_node=2 train.py --outer 1 --inner 1
"""

import json
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
from collections import Counter

from config import config
from utils import calculate_metrics, save_checkpoint
from dataset import create_dataloaders
from model import get_model


def _debug_log(location, message, data, run_id, hypothesis_id):
    try:
        payload = {
            "sessionId": "4adfef",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("/work/hdd/bdja/bpokhrel/esm_new2/.cursor/debug-4adfef.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank       = 0
        world_size = 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return local_rank, rank, world_size, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(t, world_size):
    """Sum a scalar tensor across all ranks."""
    if world_size == 1:
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(rank, world_size, device, outer_fold=1, inner_fold=1):
    is_main = (rank == 0)

    train_loader, val_loader, class_names, class_weights = create_dataloaders(
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        rank=rank,
        world_size=world_size,
    )
    num_classes        = len(class_names)
    config.num_classes = num_classes

    if is_main:
        print(f"Outer-{outer_fold} / Inner-{inner_fold} | "
              f"Classes: {num_classes} | "
              f"Train batches: {len(train_loader)} | "
              f"Val batches: {len(val_loader)}", flush=True)
              
        print("\n--- Class Distribution & Weights ---")
        _train_csv = os.path.join(config.cv_splits_dir, f"Outer_Fold_{outer_fold}", f"Inner_Fold_{inner_fold}", "train_manifest.csv")
        _val_csv   = os.path.join(config.cv_splits_dir, f"Outer_Fold_{outer_fold}", f"Inner_Fold_{inner_fold}", "valid_manifest.csv")
        _train_df = pd.read_csv(_train_csv)
        _val_df   = pd.read_csv(_val_csv)
        _train_counts = Counter(_train_df["label"].tolist())
        _val_counts   = Counter(_val_df["label"].tolist())
        for i, name in enumerate(class_names):
            print(f"  Class {i} ({name}): Train={_train_counts.get(name, 0)}, Val={_val_counts.get(name, 0)}, Weight={class_weights[i]:.4f}")
        print("------------------------------------\n", flush=True)

    model = get_model(
        num_classes,
        classify_dropout=getattr(config, "classify_dropout", 0.3),
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    raw_model = model.module if world_size > 1 else model

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=getattr(config, "label_smoothing", 0.1),
        reduction="mean",
    )

    optimizer = optim.AdamW(
        [
            {"params": raw_model.esmc.parameters(),       "lr": config.lr_esmc},
            {"params": raw_model.classifier.parameters(), "lr": config.lr_classifier},
        ],
        weight_decay=config.weight_decay,
    )

    grad_acc_steps = getattr(config, "gradient_accumulation_steps", 1)
    total_steps = (len(train_loader) // grad_acc_steps) * config.num_epochs
    warmup_steps = getattr(config, "warmup_steps", 100)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_macro_f1 = -float("inf")
    epochs_no_improve = 0
    ckpt_path = config.model_save_path.replace(
        ".pt", f"_outer{outer_fold}_inner{inner_fold}.pt"
    )

    for epoch in range(config.num_epochs):

        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        # ---- Train -------------------------------------------------------
        model.train()
        train_loss_sum   = 0.0
        train_correct    = 0
        train_samples    = 0
        
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train", disable=not is_main)):
            # region agent log H4
            if epoch == 0 and train_samples == 0:
                lengths = batch["lengths"]
                _debug_log(
                    "train.py:149",
                    "first train batch snapshot",
                    {
                        "rank": rank,
                        "world_size": world_size,
                        "device": str(device),
                        "config_batch_size": config.batch_size,
                        "actual_batch_size": int(lengths.numel()),
                        "min_seq_len": int(lengths.min().item()),
                        "max_seq_len": int(lengths.max().item()),
                        "mean_seq_len": float(lengths.float().mean().item()),
                        "labels_dtype": str(batch["labels"].dtype),
                    },
                    run_id="pre-fix",
                    hypothesis_id="H4",
                )
            # endregion
            labels  = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                loss    = criterion(outputs.float(), labels)
            
            # With reduction="mean", accumulate without manual rescaling
            loss.backward()
            
            if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs   # accumulate as sum for per-sample avg later
            train_correct  += (outputs.argmax(dim=1) == labels).sum().item()
            train_samples  += bs

        # Sample-weighted averages (no batch-size bias)
        train_loss = train_loss_sum / max(train_samples, 1)
        train_acc  = train_correct  / max(train_samples, 1)

        # ---- Validate ----------------------------------------------------
        model.eval()
        val_loss_sum   = 0.0
        val_correct    = 0
        val_samples    = 0
        
        val_preds_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val",
                              disable=not is_main):
                labels  = batch["labels"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(batch)
                    loss    = criterion(outputs.float(), labels)

                bs = labels.size(0)
                val_loss_sum += loss.item() * bs  # keep as sum, divide by total samples later
                
                preds = outputs.argmax(dim=1)
                val_correct  += (preds == labels).sum().item()
                val_samples  += bs
                
                val_preds_list.append(preds)
                val_labels_list.append(labels)

        # ---- Global reduce across GPUs -----------------------------------
        stats = torch.tensor(
            [val_loss_sum, val_correct, val_samples,
             train_loss_sum, train_correct, train_samples],
            dtype=torch.float64, device=device,
        )
        stats = reduce_tensor(stats, world_size)

        g_val_loss   = (stats[0] / stats[2].clamp(min=1)).item()
        g_val_acc    = (stats[1] / stats[2].clamp(min=1)).item()
        g_train_loss = (stats[3] / stats[5].clamp(min=1)).item()
        g_train_acc  = (stats[4] / stats[5].clamp(min=1)).item()
        
        # Gather preds/labels across ranks — all_gather_object handles uneven splits safely
        val_preds_tensor  = torch.cat(val_preds_list)
        val_labels_tensor = torch.cat(val_labels_list)
        
        if world_size > 1:
            gathered_preds  = [None] * world_size
            gathered_labels = [None] * world_size
            dist.all_gather_object(gathered_preds,  val_preds_tensor.cpu().numpy())
            dist.all_gather_object(gathered_labels, val_labels_tensor.cpu().numpy())
            val_preds_cpu  = np.concatenate(gathered_preds)
            val_labels_cpu = np.concatenate(gathered_labels)
        else:
            val_preds_cpu  = val_preds_tensor.cpu().numpy()
            val_labels_cpu = val_labels_tensor.cpu().numpy()
        
        g_val_macro_f1 = f1_score(val_labels_cpu, val_preds_cpu, average='macro', zero_division=0)
        g_val_mcc = matthews_corrcoef(val_labels_cpu, val_preds_cpu)

        if is_main:
            print(f"Epoch {epoch+1} | "
                  f"Train Loss: {g_train_loss:.4f} | "
                  f"Val Loss: {g_val_loss:.4f} | Val Acc: {g_val_acc:.4f} | "
                  f"Val Macro F1: {g_val_macro_f1:.4f} | Val MCC: {g_val_mcc:.4f}", flush=True)
            
            print("\n--- Classification Report ---")
            print(classification_report(val_labels_cpu, val_preds_cpu, labels=list(range(len(class_names))),target_names=class_names, zero_division=0))
            print("-----------------------------\n")
            
            wandb.log({
                "epoch":      epoch + 1,
                "train/loss": g_train_loss, "train/acc": g_train_acc,
                "val/loss":   g_val_loss,   "val/acc":   g_val_acc,
                "val/macro_f1": g_val_macro_f1, "val/mcc": g_val_mcc,
                "val/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_labels_cpu.tolist(),
                    preds=val_preds_cpu.tolist(),
                    class_names=class_names
                )
            })

            # Checkpoint based on Macro F1 instead of Val Loss
            if g_val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = g_val_macro_f1
                epochs_no_improve = 0
                save_checkpoint(raw_model, ckpt_path, epoch, optimizer, scheduler)
                print(f"  Best val Macro F1 {g_val_macro_f1:.4f} -- checkpoint saved", flush=True)
            else:
                epochs_no_improve += 1
                print(f"  No improvement ({epochs_no_improve}/{config.patience})", flush=True)

        # Sync early-stopping counter so all ranks stop together
        if world_size > 1:
            t = torch.tensor(epochs_no_improve, dtype=torch.int, device=device)
            dist.broadcast(t, src=0)
            epochs_no_improve = t.item()

        if epochs_no_improve >= config.patience:
            if is_main:
                print(f"Early stopping triggered after {config.patience} "
                      "epochs without val improvement.", flush=True)
            break

    if is_main:
        print(f"\nBest val Macro F1 : {best_val_macro_f1:.4f}", flush=True)
        print(f"Checkpoint    : {ckpt_path}", flush=True)
        print(f"Next step     : python evaluate.py "
              f"--outer {outer_fold} --checkpoint {ckpt_path}", flush=True)

    return raw_model, class_names, ckpt_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--inner", type=int, default=1)
    args = parser.parse_args()

    local_rank, rank, world_size, device = setup_ddp()
    is_main = (rank == 0)

    if is_main:
        print(f"[train] rank={rank} device={device} outer={args.outer} inner={args.inner}", flush=True)
        wandb.init(
            project="OPMesm-protein-localization2",
            name=f"train-outer{args.outer}-inner{args.inner}",
            config={
                "outer_fold":    args.outer,
                "inner_fold":    args.inner,
                "batch_size":    config.batch_size,
                "lr_esmc":       config.lr_esmc,
                "lr_classifier": config.lr_classifier,
                "epochs":        config.num_epochs,
                "weight_decay":  config.weight_decay,
                "patience":      config.patience,
                "world_size":    world_size,
            },
        )

    try:
        train(
            rank, world_size, device,
            outer_fold=args.outer,
            inner_fold=args.inner,
        )
        if is_main:
            print("Training complete.", flush=True)

    except KeyboardInterrupt:
        if is_main:
            print("\nInterrupted.", flush=True)

    finally:
        if is_main:
            wandb.finish()
        cleanup_ddp()


if __name__ == "__main__":
    print(f"[train.py] __file__={os.path.abspath(__file__)} "
          f"nbytes={os.path.getsize(__file__)}", flush=True)
    try:
        main()
    except Exception:
        print("[train.py] fatal error:", flush=True)
        raise
