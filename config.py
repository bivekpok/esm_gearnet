import os

class Config:
    # -----------------------------------------------------------------------
    # 1. Model Settings
    # -----------------------------------------------------------------------
    model_name_or_path = "esmc_300m"
    classify_dropout   = 0.3
    
    # LoRA Config
    lora_r       = 8
    lora_alpha   = 32
    lora_dropout = 0.1

    # -----------------------------------------------------------------------
    # 2. Data & Paths
    # -----------------------------------------------------------------------
    label_csv     = '/work/hdd/bdja/bpokhrel/esm_new2/opmesm_035data.csv'
    cv_splits_dir = "/work/hdd/bdja/bpokhrel/esm_new2/cv_splits"
    
    output_dir      = '/work/hdd/bdja/bpokhrel/esm_new2/lora_attn/'
    model_save_path = os.path.join(output_dir, 'sweep_checkpoints')

    # -----------------------------------------------------------------------
    # 3. Training Hyperparameters
    # -----------------------------------------------------------------------
    batch_size    = 4
    num_epochs    = 250
    patience      = 20
    lr_esmc       = 1e-5
    lr_classifier = 1e-4
    weight_decay  = 1e-5
    
    # Debugging
    test_mode = False

    def __init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

config = Config()
