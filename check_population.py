import pandas as pd
from pathlib import Path

def print_split_stats(csv_path, split_name):
    """Reads a CSV and prints the count and percentage for each label."""
    if not csv_path.exists():
        return
    
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    # Calculate counts and percentages (ratios)
    counts = df['label'].value_counts().sort_index()
    percentages = df['label'].value_counts(normalize=True).sort_index() * 100
    
    print(f"  {split_name:<15} | Total Samples: {total_samples}")
    
    # Print the stats for each class
    for label in counts.index:
        count = counts[label]
        pct = percentages[label]
        print(f"    🔹 {label:<20}: {count:>5} sequences ({pct:>5.1f}%)")
    print("  " + "-" * 50)

def main():
    base_dir = Path("/work/hdd/bdja/bpokhrel/esm_new2/cv_splits_cleaned")
    
    if not base_dir.exists():
        print(f" Error: Could not find the '{base_dir}' directory.")
        return
        
    print(f"🔍 Analyzing splits in: {base_dir.absolute()}\n")
    
    # Find all Outer_Fold folders and sort them numerically (1, 2, 3...)
    outer_folds = sorted(
        [d for d in base_dir.glob("Outer_Fold_*") if d.is_dir()],
        key=lambda x: int(x.name.split('_')[-1])
    )
    
    for outer_dir in outer_folds:
        print(f"==================================================")
        print(f" {outer_dir.name.upper()}")
        print(f"==================================================")
        
        # 1. Outer Loop (Test Set)
        test_csv = outer_dir / "test_manifest.csv"
        print_split_stats(test_csv, "[Outer Test]")
        
        # 2. Inner Loops (Train & Valid Sets)
        inner_folds = sorted(
            [d for d in outer_dir.glob("Inner_Fold_*") if d.is_dir()],
            key=lambda x: int(x.name.split('_')[-1])
        )
        
        for inner_dir in inner_folds:
            print(f"{inner_dir.name}:")
            
            train_csv = inner_dir / "train_manifest.csv"
            valid_csv = inner_dir / "valid_manifest.csv"
            
            print_split_stats(train_csv, "[Inner Train]")
            print_split_stats(valid_csv, "[Inner Valid]")
            
        print() # Blank line for readability

if __name__ == "__main__":
    main()