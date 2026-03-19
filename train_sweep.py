import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from config import config
from utils import calculate_metrics, save_checkpoint
from dataset import create_dataloaders
from model import get_model

def train():
    # 1. Initialize W&B run FIRST. 
    # If launched by a sweep agent, it automatically grabs the sweep parameters.
    # If run normally (python train.py), it just runs a standard logged session.
    wandb.init(project="OPMesm-protein-localization")
    
    # 2. OVERRIDE static config with W&B's suggested hyperparameters for this run.
    # We use .get() so the script doesn't crash if you run it manually without a sweep.
    config.lr_esmc = wandb.config.get("lr_esmc", config.lr_esmc)
    config.lr_classifier = wandb.config.get("lr_classifier", config.lr_classifier)
    config.lora_r = wandb.config.get("lora_r", getattr(config, 'lora_r', 8))
    config.lora_dropout = wandb.config.get("lora_dropout", getattr(config, 'lora_dropout', 0.1))
    config.classify_dropout = wandb.config.get("classify_dropout", getattr(config, 'classify_dropout', 0.3))    
    # Automatically scale LoRA alpha if LoRA rank is being swept
    if "lora_r" in wandb.config:
        config.lora_alpha = config.lora_r * 4

    # Setup Data
    train_loader, val_loader, class_names, class_weights = create_dataloaders()
    num_classes = len(class_names)
    config.num_classes = num_classes # Update config
    
    # Setup Model
    model = get_model(num_classes, config.classify_dropout)
    
    # Setup Training Components (Using class weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))
    
    optimizer = optim.AdamW(
        [
            {'params': model.esmc.parameters(), 'lr': config.lr_esmc},
            {'params': model.classifier.parameters(), 'lr': config.lr_classifier}
        ],
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training Loop State
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
        
        # --- TRAINING PHASE ---
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["labels"].to(config.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            acc, f1 = calculate_metrics(outputs, batch["labels"].to(config.device))
            train_loss += loss.item()
            train_acc += acc
            train_f1 += f1
            
        # Average train metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_f1 /= len(train_loader)
        
        # Base wandb log with training metrics
        wandb_log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss, 
            "train/acc": train_acc, 
            "train/f1": train_f1
        }
        
        # --- VALIDATION PHASE (EVERY 1 EPOCH) ---
        if (epoch + 1) % 1 == 0:
            model.eval()
            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    outputs = model(batch)
                    loss = criterion(outputs, batch["labels"].to(config.device))
                    acc, f1 = calculate_metrics(outputs, batch["labels"].to(config.device))
                    val_loss += loss.item()
                    val_acc += acc
                    val_f1 += f1
            
            # Average validation metrics
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_f1 /= len(val_loader)
            
            # Add validation metrics to wandb log dictionary
            # IMPORTANT: "val_loss" must exist exactly like this for the sweep to optimize it
            wandb_log_dict.update({
                "val_loss": val_loss, 
                "val/acc": val_acc, 
                "val/f1": val_f1
            })
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
            # Scheduler and saving rely on val_loss, so they live here
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(model, config.model_save_path, epoch, optimizer, scheduler)
                print("Saved Best Model!")
            else:
                epochs_no_improve += 1
            
            # Early Stopping Check
            if epochs_no_improve >= config.patience:
                print(f"Early stopping triggered after {epochs_no_improve} validation cycles without improvement.")
                wandb.log(wandb_log_dict) # Log the final metrics before breaking
                break
                
        else:
            # Print just training stats for the "in-between" epochs
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | (Skipping Validation)")
        
        # Log all gathered metrics to wandb for this epoch
        wandb.log(wandb_log_dict)
            
    return model, class_names


if __name__ == "__main__":
    print("Starting training pipeline...")
    try:
        # The W&B sweep agent will execute train() repeatedly.
        final_model, classes = train()
        print("Training complete! Best model saved.")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user. Shutting down gracefully...")
        
    finally:
        # Always close wandb to ensure logs sync properly and the sweep agent knows the run finished
        if wandb.run is not None:
            wandb.finish()