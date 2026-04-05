import pandas as pd
from pathlib import Path

# The classes you want to surgically remove
CLASSES_TO_REMOVE = [
    'Lysosome', 
    'Golgi', 
    'Mitochon. outer', 
    'Vacuole'
]

def clean_and_copy_splits(source_dir="/work/hdd/bdja/bpokhrel/esm_new2/cv_splits", dest_dir="/work/hdd/bdja/bpokhrel/esm_new2/cv_splits_cleaned"):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"❌ Error: Could not find '{source_dir}'.")
        return

    print(f"🧹 Reading from '{source_dir}' and saving to '{dest_dir}'...\n")
    
    total_removed = 0
    files_processed = 0

    # rglob("*.csv") recursively finds every CSV in all outer and inner folders
    for csv_path in source_path.rglob("*.csv"):
        df = pd.read_csv(csv_path)
        original_len = len(df)
        
        # Keep only rows where the label is NOT in our removal list
        df_clean = df[~df['label'].isin(CLASSES_TO_REMOVE)]
        clean_len = len(df_clean)
        
        # Figure out where to save the new file
        relative_path = csv_path.relative_to(source_path)
        new_csv_path = dest_path / relative_path
        
        # Create the necessary subdirectories in the new folder
        new_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the cleaned CSV to the new location
        df_clean.to_csv(new_csv_path, index=False)
        
        rows_dropped = original_len - clean_len
        print(f"✅ Saved: {relative_path}")
        if rows_dropped > 0:
            print(f"   Dropped {rows_dropped:2} sequences -> New total: {clean_len}")
            
        total_removed += rows_dropped
        files_processed += 1

    print("\n" + "="*50)
    print(f"🎉 DONE! Processed {files_processed} files and saved to '{dest_dir}'.")
    print(f"🗑️  Total sequences removed across all splits: {total_removed}")
    print("="*50)

if __name__ == "__main__":
    clean_and_copy_splits()