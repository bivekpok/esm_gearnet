import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def ensure_batch(x):
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def get_sdk_sequence_logits(out_sdk):
    # Based on your prior inspection: out_sdk.logits is ForwardTrackData
    # and the comparable tensor is out_sdk.logits.sequence
    return out_sdk.logits.sequence


def summarize_diff(name, a, b):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    print(f"{name}:")
    print(f"  shape      : {tuple(a.shape)}")
    print(f"  max diff   : {diff.max().item():.8f}")
    print(f"  mean diff  : {diff.mean().item():.8f}")
    print(f"  allclose   : {torch.allclose(a, b, atol=1e-5)}")


def compare_hidden_states(label, sdk_hidden, test_hidden):
    n = min(len(sdk_hidden), len(test_hidden))
    print(f"\nHidden-state comparison for {label} ({n} layers)")
    for i in range(n):
        h1 = sdk_hidden[i].float()
        h2 = test_hidden[i].float()
        diff = (h1 - h2).abs()
        print(
            f"  Layer {i:02d}: "
            f"max={diff.max().item():.8f}, "
            f"mean={diff.mean().item():.8f}"
        )


def run_verification(model_name="esmc_300m", device="cuda"):
    print("=" * 90)
    print("VERIFY WHETHER sequence_id=None IS THE REAL FIX")
    print("=" * 90)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model = ESMC.from_pretrained(model_name)
    model = model.float()
    model.eval()
    model.to(device)

    print("Model dtype :", next(model.parameters()).dtype)
    print("Device      :", next(model.parameters()).device)

    sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHG"

    with torch.no_grad():
        protein = ESMProtein(sequence=sequence)
        protein_tensor = model.encode(protein).to(device)

        tokens = ensure_batch(protein_tensor.sequence).to(device)
        B, L = tokens.shape
        manual_sequence_id = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, L)

        print("\nInput tokens shape:", tokens.shape)
        print("First 10 tokens    :", tokens[0, :10].tolist())

        # -------------------------
        # A. SDK path
        # -------------------------
        out_sdk = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)
        )

        sdk_embeddings = out_sdk.embeddings.float()
        sdk_seq_logits = get_sdk_sequence_logits(out_sdk).float()
        sdk_hidden = [x.float() for x in out_sdk.hidden_states]

        print("\nSDK outputs:")
        print("  embeddings shape      :", tuple(sdk_embeddings.shape))
        print("  sequence logits shape :", tuple(sdk_seq_logits.shape))
        print("  sequence logits dtype :", out_sdk.logits.sequence.dtype)

        # -------------------------
        # B. Direct forward with manual sequence_id
        # -------------------------
        out_manual = model.forward(
            sequence_tokens=tokens,
            sequence_id=manual_sequence_id
        )

        manual_embeddings = out_manual.embeddings.float()
        manual_seq_logits = out_manual.sequence_logits.float()
        manual_hidden = [x.float() for x in out_manual.hidden_states]

        print("\nDirect forward with manual sequence_id:")
        print("  sequence_id shape     :", tuple(manual_sequence_id.shape))
        print("  first 10 ids          :", manual_sequence_id[0, :10].tolist())

        # -------------------------
        # C. Direct forward with sequence_id=None
        # -------------------------
        out_none = model.forward(
            sequence_tokens=tokens,
            sequence_id=None
        )

        none_embeddings = out_none.embeddings.float()
        none_seq_logits = out_none.sequence_logits.float()
        none_hidden = [x.float() for x in out_none.hidden_states]

        print("\nDirect forward with sequence_id=None:")
        print("  sequence_id passed    : None")

        # -------------------------
        # Compare to SDK
        # -------------------------
        print("\n" + "=" * 90)
        print("COMPARE AGAINST SDK")
        print("=" * 90)

        summarize_diff("Embeddings: SDK vs MANUAL", sdk_embeddings, manual_embeddings)
        summarize_diff("Embeddings: SDK vs NONE", sdk_embeddings, none_embeddings)

        summarize_diff("Seq logits: SDK vs MANUAL", sdk_seq_logits, manual_seq_logits)
        summarize_diff("Seq logits: SDK vs NONE", sdk_seq_logits, none_seq_logits)

        print("\nLayer 2 quick check")
        summarize_diff("Hidden[2]: SDK vs MANUAL", sdk_hidden[2], manual_hidden[2])
        summarize_diff("Hidden[2]: SDK vs NONE", sdk_hidden[2], none_hidden[2])

        print("\nLast layer quick check")
        summarize_diff("Hidden[-1]: SDK vs MANUAL", sdk_hidden[-1], manual_hidden[-1])
        summarize_diff("Hidden[-1]: SDK vs NONE", sdk_hidden[-1], none_hidden[-1])

        # Optional: full layer listing
        compare_hidden_states("MANUAL", sdk_hidden, manual_hidden)
        compare_hidden_states("NONE", sdk_hidden, none_hidden)

        # -------------------------
        # Direct MANUAL vs NONE
        # -------------------------
        print("\n" + "=" * 90)
        print("DIRECT PATH ONLY: MANUAL vs NONE")
        print("=" * 90)

        summarize_diff("Embeddings: MANUAL vs NONE", manual_embeddings, none_embeddings)
        summarize_diff("Seq logits: MANUAL vs NONE", manual_seq_logits, none_seq_logits)
        summarize_diff("Hidden[2]: MANUAL vs NONE", manual_hidden[2], none_hidden[2])
        summarize_diff("Hidden[-1]: MANUAL vs NONE", manual_hidden[-1], none_hidden[-1])

        # -------------------------
        # Verdict
        # -------------------------
        sdk_manual_l2 = (sdk_hidden[2] - manual_hidden[2]).abs().mean().item()
        sdk_none_l2 = (sdk_hidden[2] - none_hidden[2]).abs().mean().item()

        sdk_manual_last = (sdk_hidden[-1] - manual_hidden[-1]).abs().mean().item()
        sdk_none_last = (sdk_hidden[-1] - none_hidden[-1]).abs().mean().item()

        print("\n" + "=" * 90)
        print("VERDICT")
        print("=" * 90)
        print(f"Layer2 mean diff   | SDK vs MANUAL : {sdk_manual_l2:.8f}")
        print(f"Layer2 mean diff   | SDK vs NONE   : {sdk_none_l2:.8f}")
        print(f"Last layer mean diff | SDK vs MANUAL : {sdk_manual_last:.8f}")
        print(f"Last layer mean diff | SDK vs NONE   : {sdk_none_last:.8f}")

        if sdk_none_l2 < sdk_manual_l2 and sdk_none_last < sdk_manual_last:
            print("\nRESULT: sequence_id=None is much closer to SDK.")
            print("This strongly supports that manual sequence_id was the issue.")
        else:
            print("\nRESULT: sequence_id=None did not clearly fix it.")
            print("So manual sequence_id was not the main issue, or not the only one.")


if __name__ == "__main__":
    print("Starting sequence_id verification...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_verification(model_name="esmc_300m", device=device)