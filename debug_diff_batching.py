import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def ensure_batch(x):
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x

def monkey_patch_interception(model_name="esmc_300m", device="cuda"):
    print("=" * 80)
    print("🏴‍☠️ HIJACKING THE SDK'S FORWARD METHOD")
    print("=" * 80)

    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()

    sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHG"
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein).to(device)

    # These are YOUR direct inputs
    my_tokens = ensure_batch(protein_tensor.sequence).to(device)
    B, L = my_tokens.shape
    my_seq_id = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, L)

    # ---------------------------------------------------------
    # THE HEIST: Monkey-patch the forward method
    # ---------------------------------------------------------
    original_forward = model.forward
    captured_args = []
    captured_kwargs = {}

    def hijacked_forward(*args, **kwargs):
        captured_args.extend(args)
        captured_kwargs.update(kwargs)
        # Pass execution back to the real forward method so it doesn't crash
        return original_forward(*args, **kwargs)

    # Override the method
    model.forward = hijacked_forward

    with torch.no_grad():
        # Trigger the SDK path (this will call our hijacked_forward)
        model.logits(
            protein_tensor,
            LogitsConfig(return_embeddings=True, return_hidden_states=True)
        )
    
    # Put the original method back just to be clean
    model.forward = original_forward

    # ---------------------------------------------------------
    # FORENSICS
    # ---------------------------------------------------------
    print("\n--- YOUR DIRECT INPUTS ---")
    print(f"Tokens Shape: {my_tokens.shape}")
    print(f"Tokens[:10] : {my_tokens[0, :10].tolist()}")
    print(f"Seq ID[:10] : {my_seq_id[0, :10].tolist()}")

    print("\n--- SDK'S HIDDEN INPUTS (POSITIONAL ARGS) ---")
    if not captured_args:
        print("(None)")
    for i, arg in enumerate(captured_args):
        if isinstance(arg, torch.Tensor):
            print(f"Arg {i}: Tensor of shape {arg.shape}")
            if arg.dim() >= 2:
                print(f"         Values[:10]: {arg[0, :10].tolist()}")
        else:
            print(f"Arg {i}: {type(arg)}")

    print("\n--- SDK'S HIDDEN INPUTS (KWARGS) ---")
    if not captured_kwargs:
        print("(None)")
    for k, v in captured_kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"-> {k}: Tensor of shape {v.shape}")
            if v.numel() > 0 and v.dim() >= 1:
                # Print first few values if it's a 1D or 2D tensor
                flat_v = v.flatten()
                print(f"         Values[:10]: {flat_v[:10].tolist()}")
        else:
            print(f"-> {k}: {v}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    monkey_patch_interception(model_name="esmc_300m", device=device)