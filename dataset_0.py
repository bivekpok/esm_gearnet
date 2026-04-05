import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from collections import Counter

class Config:
    # Path to your CSV file containing 'pdbid', 'label', 'sequence' columns
    label_csv = '/work/hdd/bdja/bpokhrel/esm_new2/opmesm_035data.csv'  

    # Path where validation indices will be saved
    val_indices_path = "/work/hdd/bdja/bpokhrel/esm_new2/val_indices.pt"
    
    # Path where the generated nested CV folds will be saved
    cv_splits_dir = "/work/hdd/bdja/bpokhrel/esm_new2/cv_splits"

    # Training hyperparameters
    batch_size = 32
    seed = 42
    test_mode = False          

config = Config()

class ProteinDataset(Dataset):
    def __init__(self, label_csv, test_mode=False):
        df = pd.read_csv(label_csv)
        
        if test_mode:
            df = df.head(10)
            print("TEST MODE: Using only 10 samples")
        
        # 1. Parse Data
        self.ids = df['pdbid'].tolist()
        self.sequences = df['sequence'].tolist()
        raw_labels = df['label'].tolist()

        # 2. Process Labels (Map string/int labels to 0..N indices)
        self.classes = sorted(set(raw_labels))
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [class_to_idx[lab] for lab in raw_labels]
        
        # 3. Calculate Class Weights (Inverse Square Root)
        count_dict = Counter(self.labels)
        # Ensure we count every class index, even if count is 0 (though set(raw_labels) prevents 0)
        counts = [count_dict[i] for i in range(len(self.classes))]
        
        # Convert to tensor
        counts_tensor = torch.tensor(counts, dtype=torch.float)
        weights = 1.0 / (counts_tensor + 1e-6)
        self.class_weights = weights / weights.mean()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.ids[idx]
        )

def collate_fn(batch):
    """
    Custom collate function to handle batching of variable-length sequences (strings)
    and their corresponding labels/IDs/lengths.
    """
    sequences, labels, ids = zip(*batch)
    
    # NEW: Calculate the length of every sequence in this batch
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    return {
        "sequences": sequences,          # Tuple of strings
        "labels": torch.stack(labels),   # Tensor of shape (batch_size)
        "ids": ids,                      # Tuple of strings
        "lengths": lengths               # NEW: Tensor of sequence lengths
    }
    
def create_dataloaders():
    # 1. Instantiate Dataset
    dataset = ProteinDataset(config.label_csv, test_mode=config.test_mode)
    
    # 2. Split Indices
    # Stratify ensures train/val have same proportion of classes
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        stratify=dataset.labels,
        test_size=0.3,
        random_state=config.seed
    )
    
    # 3. Save Validation Indices for reproducibility/inference later
    torch.save({'val_idx': val_idx, 'train_idx': train_idx}, config.val_indices_path)
    
    # 4. Create Subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # 5. Create Loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=config.batch_size, 
        collate_fn=collate_fn, 
        shuffle=False, 
        num_workers=0
    )
    
    # Return loaders AND metadata
    return train_loader, val_loader, dataset.classes, dataset.class_weights

def print_split_stats(df, split_name):
    """Helper function to print the population and class ratios of a dataframe."""
    total = len(df)
    # Sort by index to ensure Class 0 and Class 1 are always printed in the same order
    counts = df['label'].value_counts().sort_index()
    ratios = df['label'].value_counts(normalize=True).sort_index() * 100
    
    # Format a neat, aligned string
    stats_str = f"{split_name:<30} | Total: {total:<5} | "
    class_stats = []
    for cls in counts.index:
        class_stats.append(f"Class {cls}: {counts[cls]} ({ratios[cls]:.1f}%)")
    
    stats_str += " | ".join(class_stats)
    print(stats_str)

def generate_hybrid_splits(csv_path=None, output_root="cv_splits", n_outer=6, n_inner_for_tune=3):
    if csv_path is None:
        csv_path = config.label_csv
        
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)

    print("="*85)
    print_split_stats(df, "FULL ORIGINAL DATASET")
    print("="*85)

    # 1. OUTER LOOP: Create the 6 Test Sets
    outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=65)

    for out_idx, (remainder_idx, test_idx) in enumerate(outer_skf.split(df, df['label'])):
        outer_num = out_idx + 1
        outer_path = os.path.join(output_root, f"Outer_Fold_{outer_num}")
        os.makedirs(outer_path, exist_ok=True)
        
        # Save the Test Set
        test_df = df.iloc[test_idx]
        test_df.to_csv(os.path.join(outer_path, "test_manifest.csv"), index=False)
        remainder_df = df.iloc[remainder_idx].reset_index(drop=True)

        print(f"\n--- OUTER FOLD {outer_num} ---")
        print_split_stats(test_df, f"Outer {outer_num} TEST SET")

        # 2. INNER LOOP LOGIC
        if outer_num == 1:
            # --- FOLD 1: Create 3 Inner Folds for Hyperparameter Sweeping ---
            # Swapped to StratifiedShuffleSplit so we can force a 10% val size while keeping 3 splits
            inner_sss = StratifiedShuffleSplit(n_splits=n_inner_for_tune, test_size=0.08, random_state=65)
            
            for in_idx, (train_idx, val_idx) in enumerate(inner_sss.split(remainder_df, remainder_df['label'])):
                inner_path = os.path.join(outer_path, f"Inner_Fold_{in_idx + 1}")
                os.makedirs(inner_path, exist_ok=True)
                
                train_df = remainder_df.iloc[train_idx]
                val_df = remainder_df.iloc[val_idx]
                
                train_df.to_csv(os.path.join(inner_path, "train_manifest.csv"), index=False)
                val_df.to_csv(os.path.join(inner_path, "valid_manifest.csv"), index=False)
                
                print_split_stats(train_df, f"  Inner {in_idx + 1} TRAIN SET")
                print_split_stats(val_df, f"  Inner {in_idx + 1} VALID SET")
            
        else:
            # --- FOLDS 2 to 6: Create exactly 1 Inner Fold for Final Training ---
            # Shrunk test_size from 0.20 down to 0.10
            train_df, val_df = train_test_split(
                remainder_df, 
                test_size=0.08, 
                random_state=65, 
                stratify=remainder_df['label']
            )
            
            inner_path = os.path.join(outer_path, "Inner_Fold_1")
            os.makedirs(inner_path, exist_ok=True)
            
            train_df.to_csv(os.path.join(inner_path, "train_manifest.csv"), index=False)
            val_df.to_csv(os.path.join(inner_path, "valid_manifest.csv"), index=False)
            
            print_split_stats(train_df, f"  Inner 1 TRAIN SET")
            print_split_stats(val_df, f"  Inner 1 VALID SET")

if __name__ == "__main__":
    
    # 1. Generate the Nested Cross-Validation Splits
    print(f"Generating hybrid splits in {config.cv_splits_dir}...")
    generate_hybrid_splits(
        csv_path=config.label_csv,
        output_root=config.cv_splits_dir
    )
    print("\n" + "="*85)
    
    # 2. Test the old dataloaders functionality
    # FIXED: Unpack into local variables, not into 'dataset.classes'
    train_loader, val_loader, classes, class_weights = create_dataloaders()
    
    print(f"Classes: {classes}")
    print(f"Class weights: {class_weights}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Test one batch from training loader
    print("\n--- Testing one training batch ---")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Number of sequences: {len(batch['sequences'])}")
        print(f"Labels tensor shape: {batch['labels'].shape}")
        # Careful with slicing strings vs tensors
        print(f"First sequence (truncated): {batch['sequences'][0][:50]}...")
        print(f"First label: {batch['labels'][0].item()}")
        print(f"First ID: {batch['ids'][0]}")
        break 
    
    # Check validation indices
    if os.path.exists(config.val_indices_path):
        print(f"\nValidation indices saved to {config.val_indices_path}")
        val_data = torch.load(config.val_indices_path)
        print(f"Number of validation indices: {len(val_data['val_idx'])}")
    else:
        print(f"\nWarning: {config.val_indices_path} not found.")