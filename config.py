import torch
import os

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model settings
    model_name_or_path = "esmc_300m"
    max_len = 2000
    
    # Paths
    label_csv = '/work/hdd/bdja/bpokhrel/esm_new2/opmesm_035data.csv'
    output_dir = '/work/hdd/bdja/bpokhrel/esm_new2/lora_attn/'
    # model_save_path = os.path.join(output_dir, 'best_model_hidden.pth')
    model_save_path = os.path.join(output_dir, 'sweep_checkpoints')
    val_indices_path = os.path.join(output_dir, 'val_indices.pt')
    
    # Training Hyperparameters
    batch_size = 30
    num_epochs = 250
    patience = 20
    classify_dropout = 0.3
    lr_esmc = 1e-5
    lr_classifier = 1e-4
    weight_decay = 1e-5
    
    # LoRA Config
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    
    # Misc
    test_mode = False
    seed = 42
    project_name = "lora"
    run_name = "hidden"

    def __init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

config = Config()
