import torch
import wandb
import os
from torch.utils.data import DataLoader, Subset
from dataset import ProteinDataset, collate_fn
from model import get_model, ESMCClassifier, DeepProteinClassifier
from utils import set_seed, plot_confusion_matrix, calculate_metrics
from config import config
from train import train
import numpy as np
from esm.models.esmc import ESMC

def main():
    set_seed(config.seed)
    wandb.init(project=config.project_name, name=config.run_name, config=config.__dict__)
    
    # Run Training
    trained_model, class_names = train()
    
    # --- EVALUATION PHASE ---
    print("\nStarting Evaluation Phase...")
    
    # Reload Best Model
    # Note: We need to reconstruct the architecture first
    client = ESMC.from_pretrained(config.model_name_or_path).to(config.device)
    # (LoRA wrapper happens inside get_model usually, but here we reload weights)
    # Simplest way: call get_model again to get structure, then load weights
    model = get_model(len(class_names))
    
    try:
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {config.model_save_path}")
    except Exception as e:
        print(f"Could not load saved model ({e}). Using last training state.")
        model = trained_model

    # Prepare Validation Set (using saved indices)
    val_indices = torch.load(config.val_indices_path)['val_idx']
    full_dataset = ProteinDataset(config.label_csv, test_mode=False)
    val_dataset = Subset(full_dataset, val_indices)
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, 
        collate_fn=collate_fn, shuffle=False
    )
    
    # Inference
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            
    # Metrics & Plotting
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    
    cm_path = os.path.join(config.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)
    wandb.log({"confusion_matrix": wandb.Image(cm_path)})
    
    wandb.finish()

if __name__ == "__main__":
    main()
