"""
dataset.py - Protein Localization Dataset with DDP support

Key changes from original:
  1. create_dataloaders() now accepts (outer_fold, inner_fold, rank, world_size)
     and loads directly from the pre-generated manifest CSVs instead of doing
     a random split — so every run uses the exact same fold boundaries.
  2. DistributedSampler is used when world_size > 1 (DDP), otherwise falls
     back to the original shuffle=True behaviour (single GPU / CPU).
  3. Everything else (ProteinDataset, collate_fn, generate_hybrid_splits) is
     unchanged so the rest of your codebase keeps working.
"""

import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from collections import Counter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from config import config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    """
    Loads sequences + labels from a manifest CSV.
    Manifest CSVs have columns: pdbid, label, sequence
    (same format as the original full CSV — generate_hybrid_splits writes them
    in exactly this format).
    """

    def __init__(self, manifest_csv, class_to_idx=None, test_mode=False):
        df = pd.read_csv(manifest_csv)

        if test_mode:
            df = df.head(10)
            print(f"TEST MODE: Using only 10 samples from {manifest_csv}")

        self.ids       = df['pdbid'].tolist()
        self.sequences = df['sequence'].tolist()
        raw_labels     = df['label'].tolist()

        # If class_to_idx is provided (e.g. from the train set), reuse it so
        # val/test indices are consistent with training.
        if class_to_idx is None:
            self.classes    = sorted(set(raw_labels))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes      = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

        self.labels = [self.class_to_idx[lab] for lab in raw_labels]

        # --- IMPROVED WEIGHT CALCULATION ---
        count_dict = Counter(self.labels)
        counts = torch.tensor([count_dict.get(i, 0) for i in range(len(self.classes))], dtype=torch.float)
        
        # 1. Balanced weighting formula: n_samples / (n_classes * n_samples_j)
        # This is more stable than a pure 1/x reciprocal
        total_samples = counts.sum()
        n_classes = len(self.classes)
        
        weights = total_samples / (n_classes * counts + 1e-6)
        
        # 2. Normalize so the mean weight is exactly 1.0 
        # This keeps your learning rate feeling the same as unweighted training
        self.class_weights = weights / weights.mean()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.ids[idx],
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    sequences, labels, ids = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return {
        "sequences": sequences,
        "labels":    torch.stack(labels),
        "ids":       ids,
        "lengths":   lengths,
    }


# ---------------------------------------------------------------------------
# create_dataloaders  ← main change
# ---------------------------------------------------------------------------

def create_dataloaders(
    outer_fold=1,
    inner_fold=1,
    rank=0,
    world_size=1,
):
    """
    Load train / val manifests for a specific (outer_fold, inner_fold) pair
    and return DataLoaders ready for DDP or single-GPU training.

    Parameters
    ----------
    outer_fold  : int   Outer fold number (1-indexed, matches directory names)
    inner_fold  : int   Inner fold number (1-indexed, matches directory names)
    rank        : int   Current process rank  (0 for single-GPU)
    world_size  : int   Total number of processes (1 for single-GPU)

    Returns
    -------
    train_loader, val_loader, class_names (list), class_weights (Tensor)
    """

    # ---- Resolve manifest paths ------------------------------------------
    fold_dir = os.path.join(
        config.cv_splits_dir,
        f"Outer_Fold_{outer_fold}",
        f"Inner_Fold_{inner_fold}",
    )
    train_csv = os.path.join(fold_dir, "train_manifest.csv")
    val_csv   = os.path.join(fold_dir, "valid_manifest.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Train manifest not found: {train_csv}\n"
            "Run generate_hybrid_splits() first (python dataset.py)."
        )
    if not os.path.exists(val_csv):
        raise FileNotFoundError(
            f"Val manifest not found: {val_csv}\n"
            "Run generate_hybrid_splits() first (python dataset.py)."
        )

    # ---- Build datasets --------------------------------------------------
    # Train dataset defines the canonical class→index mapping
    train_dataset = ProteinDataset(
        train_csv,
        test_mode=config.test_mode,
    )
    # Val dataset reuses the same mapping so indices are consistent
    val_dataset = ProteinDataset(
        val_csv,
        class_to_idx=train_dataset.class_to_idx,
        test_mode=config.test_mode,
    )

    # ---- Samplers --------------------------------------------------------
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        train_shuffle = False   # sampler owns shuffling
    else:
        train_sampler = None
        val_sampler   = None
        train_shuffle = True

    # ---- DataLoaders -----------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes, train_dataset.class_weights


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------

def print_split_stats(df, split_name):
    total  = len(df)
    counts = df['label'].value_counts().sort_index()
    ratios = df['label'].value_counts(normalize=True).sort_index() * 100

    stats_str   = f"{split_name:<30} | Total: {total:<5} | "
    class_stats = [
        f"Class {cls}: {counts[cls]} ({ratios[cls]:.1f}%)"
        for cls in counts.index
    ]
    print(stats_str + " | ".join(class_stats))


def generate_hybrid_splits(
    csv_path=None,
    output_root="cv_splits",
    n_outer=6,
    n_inner_for_tune=3,
):
    """Unchanged from original — generates the manifest CSVs on disk."""
    # --- NEW LOGIC: Check if splits are already generated ---
    check_file = os.path.join(output_root, "Outer_Fold_1", "test_manifest.csv")
    if os.path.exists(check_file):
        print(f"✅ Cross-validation splits already detected in '{output_root}'.")
        print("   Skipping generation to preserve existing folds.")
        return  # Exits the function immediately without doing anything
    if csv_path is None:
        csv_path = config.label_csv

    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)

    print("=" * 85)
    print_split_stats(df, "FULL ORIGINAL DATASET")
    print("=" * 85)

    outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=65)

    for out_idx, (remainder_idx, test_idx) in enumerate(outer_skf.split(df, df['label'])):
        outer_num  = out_idx + 1
        outer_path = os.path.join(output_root, f"Outer_Fold_{outer_num}")
        os.makedirs(outer_path, exist_ok=True)

        test_df      = df.iloc[test_idx]
        remainder_df = df.iloc[remainder_idx].reset_index(drop=True)

        test_df.to_csv(os.path.join(outer_path, "test_manifest.csv"), index=False)

        print(f"\n--- OUTER FOLD {outer_num} ---")
        print_split_stats(test_df, f"Outer {outer_num} TEST SET")

        if outer_num == 1:
            inner_sss = StratifiedShuffleSplit(
                n_splits=n_inner_for_tune, test_size=0.08, random_state=65
            )
            for in_idx, (train_idx, val_idx) in enumerate(
                inner_sss.split(remainder_df, remainder_df['label'])
            ):
                inner_path = os.path.join(outer_path, f"Inner_Fold_{in_idx + 1}")
                os.makedirs(inner_path, exist_ok=True)

                train_df = remainder_df.iloc[train_idx]
                val_df   = remainder_df.iloc[val_idx]

                train_df.to_csv(os.path.join(inner_path, "train_manifest.csv"), index=False)
                val_df.to_csv(os.path.join(inner_path,   "valid_manifest.csv"), index=False)

                print_split_stats(train_df, f"  Inner {in_idx + 1} TRAIN SET")
                print_split_stats(val_df,   f"  Inner {in_idx + 1} VALID SET")
        else:
            train_df, val_df = train_test_split(
                remainder_df,
                test_size=0.08,
                random_state=65,
                stratify=remainder_df['label'],
            )
            inner_path = os.path.join(outer_path, "Inner_Fold_1")
            os.makedirs(inner_path, exist_ok=True)

            train_df.to_csv(os.path.join(inner_path, "train_manifest.csv"), index=False)
            val_df.to_csv(os.path.join(inner_path,   "valid_manifest.csv"), index=False)

            print_split_stats(train_df, f"  Inner 1 TRAIN SET")
            print_split_stats(val_df,   f"  Inner 1 VALID SET")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Generate splits (only needs to run once)
    print(f"Generating hybrid splits in {config.cv_splits_dir}...")
    generate_hybrid_splits(
        csv_path=config.label_csv,
        output_root=config.cv_splits_dir,
    )
    print("\n" + "=" * 85)

    # 2. Test the new manifest-based dataloader (Outer 1, Inner 1, single GPU)
    train_loader, val_loader, classes, class_weights = create_dataloaders(
        outer_fold=1, inner_fold=1
    )

    print(f"Classes       : {classes}")
    print(f"Class weights : {class_weights}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    print("\n--- One training batch ---")
    for batch in train_loader:
        print(f"Keys            : {list(batch.keys())}")
        print(f"Sequences       : {len(batch['sequences'])}")
        print(f"Labels shape    : {batch['labels'].shape}")
        print(f"First seq (50c) : {batch['sequences'][0][:50]}...")
        print(f"First label     : {batch['labels'][0].item()}")
        print(f"First ID        : {batch['ids'][0]}")
        break
