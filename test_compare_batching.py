import torch
from esm.models.esmc import ESMProtein, LogitsConfig

# 1. Import your factory function directly from your model.py
from model import get_model 

def compare_esm_methods(esmc_model, device="cuda"):
    print("=" * 60)
    print("🔍 COMPARING SDK vs DIRECT PYTORCH FORWARD PASS")
    print("=" * 60)

    sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHG"
    print(f"Test Sequence Length: {len(sequence)}")
    
    esmc_model.eval() 
    esmc_model.to(device)

    with torch.no_grad():
        # --- METHOD A: SDK ---
        protein = ESMProtein(sequence=sequence)
        protein_tensor = esmc_model.encode(protein).to(device)
        outputs_sdk = esmc_model.logits(
            protein_tensor,
            LogitsConfig(return_embeddings=False, return_hidden_states=True)
        )
        layer_2_sdk = outputs_sdk.hidden_states[2].squeeze(1) 
        
        print("\n--- Method A: SDK ---")
        print(f"Layer 2 Shape: {layer_2_sdk.shape}")

        # --- METHOD B: PyTorch Direct ---
        tokens_pt = torch.tensor(esmc_model.tokenizer.encode(sequence)).unsqueeze(0).to(device)
        B, L = tokens_pt.shape
        sequence_id_pt = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        
        # Bypass the Peft wrapper for the forward pass
        base_model = esmc_model.base_model.model if hasattr(esmc_model, "base_model") else esmc_model
        
        outputs_pt = base_model.forward(
            sequence_tokens=tokens_pt,
            sequence_id=sequence_id_pt
        )
        layer_2_pt = outputs_pt.hidden_states[2].squeeze(1) 

        print("\n--- Method B: PyTorch Direct ---")
        print(f"Layer 2 Shape: {layer_2_pt.shape}")

        # --- COMPARISON ---
        print("\n--- Comparison Results ---")
        if layer_2_sdk.shape != layer_2_pt.shape:
            print(f"❌ Shape Mismatch! SDK: {layer_2_sdk.shape} vs PT: {layer_2_pt.shape}")
            return
            
        max_diff = (layer_2_sdk - layer_2_pt).abs().max().item()
        print(f"Max Absolute Difference : {max_diff:.8f}")
        
        if torch.allclose(layer_2_sdk, layer_2_pt, atol=1e-5):
            print("✅ SUCCESS: Both methods produce mathematically identical representations!")
        else:
            print("⚠️ WARNING: Differences detected.")

if __name__ == "__main__":
    print("Loading model from model.py...")
    
    # 2. Instantiate your model (8 classes because we dropped 4!)
    full_model = get_model(num_classes=8)
    
    # 3. Extract just the ESM backbone from your ESMCClassifier
    esmc_backbone = full_model.esmc
    
    # 4. Run the comparison
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compare_esm_methods(esmc_backbone, device=device)
    