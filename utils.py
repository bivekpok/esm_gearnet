import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import wandb
from config import config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
    return accuracy, f1

def save_checkpoint(model, base_save_path, epoch, optimizer, scheduler):
    """
    Saves the LoRA adapters and the classification head efficiently.
    Prevents saving the massive frozen ESM backbone.
    """
    # Create a unique folder for this specific W&B sweep run
    run_id = wandb.run.id if wandb.run is not None else "local_run"
    checkpoint_dir = os.path.join(base_save_path, f"run_{run_id}_best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Save ONLY the Peft/LoRA weights (~10MB)
    model.esmc.save_pretrained(os.path.join(checkpoint_dir, "lora_adapter"))
    
    # 2. Save the custom classification head (~2MB)
    classifier_path = os.path.join(checkpoint_dir, "classifier.pt")
    torch.save(model.classifier.state_dict(), classifier_path)
    
    # 3. Save optimizer/scheduler state (Optional, but good for resuming)
    train_state_path = os.path.join(checkpoint_dir, "train_state.pt")
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config.__dict__,
    }, train_state_path)

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path