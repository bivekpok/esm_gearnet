"""
evaluate.py - Final evaluation on the held-out test set

Run ONLY after training is complete. Never call this during training.

Usage:
  python evaluate.py --outer 1 --checkpoint /path/to/best_model_outer1_inner1.pt

What this does:
  1. Loads test_manifest.csv for the given outer fold
  2. Loads the best checkpoint saved by train.py
  3. Runs one forward pass over the entire test set (no gradients)
  4. Prints and saves: accuracy, F1, per-class report, confusion matrix
"""

import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from config import config
from dataset import ProteinDataset, collate_fn
from model import get_model


# ---------------------------------------------------------------------------
# Test loader — loads test_manifest.csv only
# ---------------------------------------------------------------------------

def create_test_loader(outer_fold, class_to_idx):
    """
    Loads ONLY the test_manifest.csv for the given outer fold.
    class_to_idx must come from the training set so label indices match.
    """
    test_csv = os.path.join(
        config.cv_splits_dir,
        f"Outer_Fold_{outer_fold}",
        "test_manifest.csv",
    )
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"Test manifest not found: {test_csv}\n"
            "Run generate_hybrid_splits() first."
        )

    test_dataset = ProteinDataset(
        test_csv,
        class_to_idx=class_to_idx,  # reuse train mapping — critical
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,              # never shuffle test
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    return test_loader, test_dataset.classes


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(outer_fold, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint -------------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # save_checkpoint should have stored class info — fall back if not
    # If your save_checkpoint doesn't save class_to_idx, load it from the
    # train manifest to reconstruct the mapping.
    if "class_to_idx" in ckpt:
        class_to_idx = ckpt["class_to_idx"]
        class_names  = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    else:
        # Reconstruct from train manifest (safe fallback)
        train_csv = os.path.join(
            config.cv_splits_dir,
            f"Outer_Fold_{outer_fold}",
            "Inner_Fold_1",
            "train_manifest.csv",
        )
        train_ds     = ProteinDataset(train_csv)
        class_to_idx = train_ds.class_to_idx
        class_names  = train_ds.classes

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # ---- Test loader -----------------------------------------------------
    test_loader, _ = create_test_loader(outer_fold, class_to_idx)
    print(f"Test batches: {len(test_loader)}")

    # ---- Model -----------------------------------------------------------
    model = get_model(
        num_classes,
        classify_dropout=0.0,   # disable dropout at test time
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Inference -------------------------------------------------------
    all_preds  = []
    all_labels = []
    all_ids    = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch)
            preds   = torch.argmax(outputs, dim=1).cpu().tolist()
            labels  = batch["labels"].tolist()
            ids     = list(batch["ids"])

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_ids.extend(ids)

    # ---- Metrics ---------------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS — Outer Fold {outer_fold}")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # ---- Save results to CSV --------------------------------------------
    results_df = pd.DataFrame({
        "pdbid":      all_ids,
        "true_label": [class_names[l] for l in all_labels],
        "pred_label": [class_names[p] for p in all_preds],
        "correct":    [l == p for l, p in zip(all_labels, all_preds)],
    })

    out_csv = os.path.join(
        config.cv_splits_dir,
        f"Outer_Fold_{outer_fold}",
        "test_results.csv",
    )
    results_df.to_csv(out_csv, index=False)
    print(f"\nPer-protein results saved to: {out_csv}")

    return acc, f1, results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer",      type=int, required=True,
                        help="Outer fold number (1-6)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint from train.py")
    args = parser.parse_args()

    evaluate(args.outer, args.checkpoint)