import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter

class Config:
    # Path to your CSV file containing 'pdbid', 'label', 'sequence' columns
    label_csv = '/work/hdd/bdja/bpokhrel/esm_new2/opmesm_035data.csv'  

    # Path where validation indices will be saved
    val_indices_path = "/work/hdd/bdja/bpokhrel/esm_new2/val_indices.pt"

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

if __name__ == "__main__":
    
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