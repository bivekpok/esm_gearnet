"""
model.py - ESM-based Protein Localization Classifier with LoRA and Attention

Parallelization strategy:
- Uses self.esmc.forward() directly (bypasses broken SDK encode/logits path)
- Single batched GPU forward pass instead of sequential per-sequence loop
- Compatible with DistributedDataParallel (DDP) wrapping in train.py
"""

import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from esm.models.esmc import ESMC
from peft import get_peft_model, LoraConfig, TaskType
from config import config


def _debug_log(location, message, data, run_id, hypothesis_id):
    try:
        payload = {
            "sessionId": "4adfef",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("/work/hdd/bdja/bpokhrel/esm_new2/.cursor/debug-4adfef.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(x.shape[-1])
        if mask is not None:
            # mask: (B, L) → (B, 1, L)  — zero-out padding positions
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return self.layer_norm(context + x)


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class DeepProteinClassifier(nn.Module):
    def __init__(self, num_classes, classify_dropout):
        super().__init__()
        self.attention = Attention(embed_dim=960)
        self.fc_layers = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(),
            nn.Dropout(classify_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(classify_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(classify_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, mask=None):
        context = self.attention(x, mask)

        if mask is not None:
            mask_float = mask.unsqueeze(-1).to(context.dtype) # (B, L, 1) — stay in bfloat16
            pooled = (context * mask_float).sum(dim=1) \
                     / mask_float.sum(dim=1).clamp(min=1e-9)  # (B, 960)
        else:
            pooled = context.mean(dim=1)

        # region agent log H1
        if not hasattr(self, "_dbg_classifier_logged"):
            self._dbg_classifier_logged = True
            _debug_log(
                "model.py:81",
                "classifier dtype snapshot",
                {
                    "rank": os.environ.get("RANK", "0"),
                    "x_dtype": str(x.dtype),
                    "context_dtype": str(context.dtype),
                    "mask_dtype": None if mask is None else str(mask.dtype),
                    "pooled_dtype": str(pooled.dtype),
                    "fc0_weight_dtype": str(self.fc_layers[0].weight.dtype),
                },
                run_id="post-fix",
                hypothesis_id="H1",
            )
        # endregion

        return self.fc_layers(pooled)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ESMCClassifier(nn.Module):
    """
    Wraps the LoRA-adapted ESMC backbone + classification head.

    Forward pass design:
      1. Tokenize all sequences on CPU with the ESM tokenizer.
      2. Pad into a single (B, L) tensor and move to GPU.
      3. Call self.esmc.forward() ONCE — one batched GPU pass.
         This completely bypasses the SDK's encode/logits path which
         deadlocks on variable-length batches during training.
      4. Pass (B, L, 960) embeddings to the classifier.
    """

    def __init__(self, esmc_model, classifier):
        super().__init__()
        self.esmc       = esmc_model
        self.classifier = classifier

    def forward(self, batch):
        device = next(self.parameters()).device  # works correctly under DDP

        # ------------------------------------------------------------------
        # 1. Tokenize on CPU
        # ------------------------------------------------------------------
        tokenized = [
            torch.tensor(self.esmc.tokenizer.encode(seq))
            for seq in batch["sequences"]
        ]

        # ------------------------------------------------------------------
        # 2. Pad → (B, L) and move to the correct GPU
        #    ESM-C pad token id = 1
        # ------------------------------------------------------------------
        tokens_t = torch.nn.utils.rnn.pad_sequence(
            tokenized,
            batch_first=True,
            padding_value=1,
        ).to(device)

        # ------------------------------------------------------------------
        # 3. Position ids for RoPE — Let the model handle it internally
        # ------------------------------------------------------------------
        B, L = tokens_t.shape
        sequence_id = None

        # region agent log H2
        if not hasattr(self, "_dbg_backbone_input_logged"):
            self._dbg_backbone_input_logged = True
            seq_lengths = batch.get("lengths")
            _debug_log(
                "model.py:154",
                "backbone input snapshot",
                {
                    "rank": os.environ.get("RANK", "0"),
                    "device": str(device),
                    "batch_size": int(B),
                    "token_shape": list(tokens_t.shape),
                    "token_dtype": str(tokens_t.dtype),
                    "max_seq_len": None if seq_lengths is None else int(seq_lengths.max().item()),
                    "min_seq_len": None if seq_lengths is None else int(seq_lengths.min().item()),
                    "mean_seq_len": None if seq_lengths is None else float(seq_lengths.float().mean().item()),
                    "cuda_allocated_mb": None if not torch.cuda.is_available() else round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
                    "cuda_reserved_mb": None if not torch.cuda.is_available() else round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
                },
                run_id="pre-fix",
                hypothesis_id="H2",
            )
        # endregion

        # ------------------------------------------------------------------
        # 4. Single batched forward pass (bypasses SDK, gradient-friendly)
        #    Call through base_model.model to skip PeftModelForFeatureExtraction
        #    .forward() which injects an `input_ids` kwarg that ESMC doesn't accept.
        #    LoRA adapters live on the individual Linear layers and remain active.
        # ------------------------------------------------------------------
        try:
            output = self.esmc.base_model.model.forward(
                sequence_tokens=tokens_t,
                sequence_id=sequence_id,
            )
        # region agent log H3
        except Exception as e:
            _debug_log(
                "model.py:180",
                "backbone forward exception",
                {
                    "rank": os.environ.get("RANK", "0"),
                    "exc_type": type(e).__name__,
                    "exc": str(e),
                    "cuda_allocated_mb": None if not torch.cuda.is_available() else round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
                    "cuda_reserved_mb": None if not torch.cuda.is_available() else round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
                },
                run_id="pre-fix",
                hypothesis_id="H3",
            )
            raise
        # endregion
        
        # Expected Shape: (Num_Layers=30, Batch_Size, Seq_Len, Hidden_Dim=960)
        stacked_hiddens = output.hidden_states
        
        # Mean Pooling Across the LAYERS (excluding the first/embedding layer)
        # This collapses the layers down, leaving you with [B, L, 960]
        layer_averaged_hiddens = stacked_hiddens[1:].mean(dim=0) # all  layers except the first one
        # layer_averaged_hiddens = stacked_hiddens[-4:].mean(dim=0) # last 4 layers

        embeddings = layer_averaged_hiddens  # (B, L, 960)
        # 

        # ------------------------------------------------------------------
        # 5. Attention mask: 1 = real token, 0 = padding
        # ------------------------------------------------------------------
        attention_mask = (tokens_t != 1).long()

        # ------------------------------------------------------------------
        # 6. Classify
        # ------------------------------------------------------------------
        return self.classifier(embeddings, attention_mask)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(num_classes, classify_dropout=0.3):
    """
    Instantiates the full model on CPU.
    train.py is responsible for moving it to the correct device / wrapping
    with DDP — do NOT call .to(device) here so DDP can manage placement.
    """
    # 1. Load backbone
    backbone = ESMC.from_pretrained(config.model_name_or_path)

    # 2. Attach LoRA adapters
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["attn.layernorm_qkv.1", "attn.out_proj", "ffn.1", "ffn.3"],
    )
    backbone = get_peft_model(backbone, peft_config)

    # 3. Build full model (stays on CPU until train.py moves it)
    # The ESM-C backbone uses bfloat16 by default, so we must cast the 
    # classifier to match to avoid dtype mismatch errors (bfloat16 vs float32)
    classifier = DeepProteinClassifier(num_classes, classify_dropout).to(torch.bfloat16)
    model = ESMCClassifier(backbone, classifier)

    return model
