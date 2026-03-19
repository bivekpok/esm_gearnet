# import torch
# import torch.nn as nn
# import torch.optim as optim
# import wandb
# from tqdm import tqdm
# from config import config
# from utils import calculate_metrics, save_checkpoint
# from dataset import create_dataloaders
# from model import get_model

# def train():
#     # Setup Data
#     train_loader, val_loader, class_names, class_weights = create_dataloaders()
#     num_classes = len(class_names)
#     config.num_classes = num_classes # Update config
    
#     # Setup Model
#     model = get_model(num_classes)
    
#     # Setup Training Components (Now using class weights!)
#     criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))
    
#     optimizer = optim.AdamW(
#         [
#             {'params': model.esmc.parameters(), 'lr': config.lr_esmc},
#             {'params': model.classifier.parameters(), 'lr': config.lr_classifier}
#         ],
#         weight_decay=config.weight_decay
#     )
    
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, verbose=True
#     )
    
#     # Training Loop
#     best_val_loss = float('inf')
#     epochs_no_improve = 0
    
#     for epoch in range(config.num_epochs):
#         if epochs_no_improve >= config.patience:
#             print("Early stopping triggered")
#             break
            
#         model.train()
#         train_loss, train_acc, train_f1 = 0.0, 0.0, 0.0
        
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
#             optimizer.zero_grad()
#             outputs = model(batch)
#             loss = criterion(outputs, batch["labels"].to(config.device))
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             acc, f1 = calculate_metrics(outputs, batch["labels"].to(config.device))
#             train_loss += loss.item()
#             train_acc += acc
#             train_f1 += f1
            
#         # Validation
#         model.eval()
#         val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
#                 outputs = model(batch)
#                 loss = criterion(outputs, batch["labels"].to(config.device))
#                 acc, f1 = calculate_metrics(outputs, batch["labels"].to(config.device))
#                 val_loss += loss.item()
#                 val_acc += acc
#                 val_f1 += f1
        
#         # Averages
#         train_loss /= len(train_loader)
#         train_acc /= len(train_loader)
#         train_f1 /= len(train_loader)
#         val_loss /= len(val_loader)
#         val_acc /= len(val_loader)
#         val_f1 /= len(val_loader)
        
#         scheduler.step(val_loss)
        
#         wandb.log({
#             "epoch": epoch + 1,
#             "train/loss": train_loss, "train/acc": train_acc, "train/f1": train_f1,
#             "val/loss": val_loss, "val/acc": val_acc, "val/f1": val_f1
#         })
        
#         print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#             save_checkpoint(model, config.model_save_path, epoch, optimizer, scheduler)
#             print("Saved Best Model!")
#         else:
#             epochs_no_improve += 1
            
#     return model, class_names


# if __name__ == "__main__":
#     # 1. Initialize Weights & Biases (wandb) before training starts
#     # You can change the project name to whatever you like
#     wandb.init(
#         project="OPMesm-protein-localization", 
#         name="esm-lora-run-1",
#         # If your config is an object, you might need to pass specific dictionary values here
#         config={
#             "batch_size": config.batch_size,
#             "lr_esmc": config.lr_esmc,
#             "lr_classifier": config.lr_classifier,
#             "epochs": config.num_epochs
#         }
#     )
    
#     print("Starting full training pipeline...")
    
#     try:
#         # 2. Call your massive train function
#         final_model, classes = train()
#         print("Training complete! Best model saved.")
        
#     except KeyboardInterrupt:
#         print("\n Training interrupted by user. Shutting down gracefully...")
        
#     finally:
#         # 3. Always close wandb, even if the run crashes or you press Ctrl+C
#         wandb.finish()



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
    # Setup Data
    train_loader, val_loader, class_names, class_weights = create_dataloaders()
    num_classes = len(class_names)
    config.num_classes = num_classes # Update config
    
    # Setup Model
    model = get_model(num_classes)
    
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
        
        # --- VALIDATION PHASE (EVERY 5 EPOCHS) ---
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
            wandb_log_dict.update({
                "val/loss": val_loss, "val/acc": val_acc, "val/f1": val_f1
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
    # 1. Initialize Weights & Biases (wandb)
    wandb.init(
        project="OPMesm-protein-localization", 
        name="esm-lora-run-1",
        config={
            "batch_size": getattr(config, 'batch_size', 32),
            "lr_esmc": getattr(config, 'lr_esmc', 1e-4),
            "lr_classifier": getattr(config, 'lr_classifier', 1e-3),
            "epochs": getattr(config, 'num_epochs', 50),
            "weight_decay": getattr(config, 'weight_decay', 0.01),
            "patience": getattr(config, 'patience', 5)
        }
    )
    
    print("Starting full training pipeline...")
    
    try:
        final_model, classes = train()
        print("Training complete! Best model saved.")
        
    except KeyboardInterrupt:
        print("\n Training interrupted by user. Shutting down gracefully...")
        
    finally:
        # 3. Always close wandb to ensure logs sync properly
        wandb.finish()