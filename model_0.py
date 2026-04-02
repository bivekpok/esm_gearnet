
"""
model.py - ESM-based Protein Localization Classifier with LoRA and Attention

This module defines the neural network architecture for predicting protein subcellular 
localization. It leverages a pre-trained ESM (Evolutionary Scale Modeling) backbone 
efficiently tuned via LoRA (Low-Rank Adaptation) and a custom attention mechanism.

Key Components:
1. ESMCClassifier: 
   - The main wrapper class.
   - Extracts token-level embeddings from the ESM-300M backbone.
   - Handles variable-length sequences via padding and masking.
   
2. DeepProteinClassifier:
   - A dedicated classification head.
   - Uses a learnable Attention mechanism to pool sequence embeddings into a single
     fixed-size vector (weighted average of amino acids).
   - Passes the pooled representation through a 3-layer MLP (Multi-Layer Perceptron)
     with Dropout and ReLU activation for final class prediction.

3. Attention (Self-Attention):
   - A standard Query-Key-Value (QKV) attention layer with LayerNorm.
   - Allows the model to weigh specific amino acids (motifs/signals) more heavily 
     than others when determining localization.

Usage:
   call `get_model(num_classes)` to instantiate the full architecture with 
   LoRA adapters already attached and configured.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from peft import get_peft_model, LoraConfig, TaskType
from config import config
from esm.sdk.api import ESMProteinTensor 
class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(x.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return self.layer_norm(context + x)

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
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, mask=None):
        context = self.attention(x, mask)
        
        if mask is not None:
            # unsqueeze(-1) puts the new dimension at the end -> (B, L, 1) to multiply
            mask_float = mask.unsqueeze(-1).float() 

            pooled = (context * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = context.mean(dim=1)   # fallback
            
        return self.fc_layers(pooled)

class ESMCClassifier(nn.Module):
    def __init__(self, esmc_model, classifier):
        super().__init__()
        self.esmc = esmc_model
        self.classifier = classifier
        
    

    def forward(self, batch):
        embeddings = []
        # Inference loop for ESM
        for seq in batch["sequences"]:
            protein = ESMProtein(sequence=seq)
            protein_tensor = self.esmc.encode(protein).to(config.device)
            outputs = self.esmc.logits(
                protein_tensor,
                LogitsConfig(return_embeddings=False, return_hidden_states=True)
            )
            hidden_s = outputs.hidden_states[1:]
            hidden_s = hidden_s.squeeze(1)
            embed = torch.mean(hidden_s, dim=0)
            embeddings.append(embed)
        
        # Padding logic
        lengths = batch["lengths"].tolist()
        max_len = max(lengths)
        embed_dim = embeddings[0].shape[-1]
        batch_size = len(embeddings)
        
        padded_embeddings = torch.zeros((batch_size, max_len, embed_dim), device=config.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=config.device)
        
        for i, (embed, seq_len) in enumerate(zip(embeddings, lengths)):
            padded_embeddings[i, :seq_len] = embed[:seq_len]
            attention_mask[i, :seq_len] = 1
        
        return self.classifier(padded_embeddings, attention_mask)

def get_model(num_classes, classify_dropout):
    # 1. Load Base Model
    client = ESMC.from_pretrained(config.model_name_or_path).to(config.device)
    
    # 2. Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["attn.layernorm_qkv.1", "attn.out_proj", "ffn.1", "ffn.3"],
    )
    client = get_peft_model(client, peft_config)
    
    # 3. Attach Classifier head
    classifier = DeepProteinClassifier(num_classes, classify_dropout).to(config.device)
    model = ESMCClassifier(client, classifier).to(config.device)
    
    return model

###Paralilzation

# def forward(self, batch):
    #    # 1. Tokenization on CPU (Very Fast)
    #    tokenized = []
    #    for seq in batch["sequences"]:
    #        # Get raw tokens: [BOS, tokens..., EOS]
    #        t = torch.tensor(self.esmc.tokenizer.encode(seq))
    #        tokenized.append(t)
       
    #    # 2. Pad to create a (Batch, Length) tensor
    #    # ESM-C uses 1 as the padding token
    #    tokens_t = torch.nn.utils.rnn.pad_sequence(
    #        tokenized, 
    #        batch_first=True, 
    #        padding_value=1 
    #    ).to(config.device)

    #    # 3. Create explicit sequence IDs (Positions)
    #    # This tells the RoPE layer EXACTLY which position each token is at.
    #    # Shape: (Batch, Length) -> [0, 1, 2, ..., L-1] for every row
    #    B, L = tokens_t.shape
    #    sequence_id = torch.arange(L, device=config.device).unsqueeze(0).expand(B, L)

    #    # 4. Direct GPU Forward (The Speedup)
    #    # We call self.esmc.forward directly to bypass the faulty SDK auto-batcher
    #    # This returns an ESMOutput object containing .embeddings
    #    output = self.esmc.forward(
    #        sequence_tokens=tokens_t, 
    #        sequence_id=sequence_id
    #    )
       
    #    embeddings = output.embeddings # Shape: (B, L, 960)
      
    #    # 5. Create Attention Mask for your classifier
    #    # 1 for real tokens, 0 for padding
    #    attention_mask = (tokens_t != 1).to(config.device)
      
    #    # 6. Final Classification
    #    classifier_output = self.classifier(embeddings, attention_mask)
      
    #    # We return a dictionary to match your training loop expectations
    #    return classifier_output


# """
# model.py - ESM-based Protein Localization Classifier with LoRA and Attention

# This module defines the neural network architecture for predicting protein subcellular 
# localization. It leverages a pre-trained ESM (Evolutionary Scale Modeling) backbone 
# efficiently tuned via LoRA (Low-Rank Adaptation).
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
# from peft import get_peft_model, LoraConfig, TaskType
# from config import config

# class Attention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.layer_norm = nn.LayerNorm(embed_dim)

#     def forward(self, x, mask=None):
#         Q, K, V = self.query(x), self.key(x), self.value(x)
#         # Scaled Dot-Product Attention
#         scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(x.shape[-1])
#         if mask is not None:
#             # mask shape (B, L) -> (B, 1, L)
#             scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
#         attn_weights = F.softmax(scores, dim=-1)
#         context = torch.matmul(attn_weights, V)
#         return self.layer_norm(context + x)

# class DeepProteinClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.attention = Attention(embed_dim=960)
#         self.fc_layers = nn.Sequential(
#             nn.Linear(960, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x, mask=None):
#         # x is (Batch, SeqLen, 960)
#         context = self.attention(x, mask)
        
#         # We perform masked mean pooling to ignore the padding tokens in the average
#         if mask is not None:
#             mask_float = mask.unsqueeze(-1).float() # (B, L, 1)
#             pooled = (context * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
#         else:
#             pooled = context.mean(dim=1)
            
#         return self.fc_layers(pooled)

# class ESMCClassifier(nn.Module):
#     def __init__(self, esmc_model, classifier):
#         super().__init__()
#         self.esmc = esmc_model
#         self.classifier = classifier
#     def forward(self, batch):
#     # 1. Build ESMProtein objects (CPU-side)
#         proteins = tuple(
#             ESMProtein(sequence=seq) for seq in batch["sequences"]
#         )

#         # 2. Encode ONCE (this version requires a tuple, not a list)
#         encoded = self.esmc.encode(proteins)

#         # 3. Logits ONCE
#         outputs = self.esmc.logits(
#             encoded,
#             LogitsConfig(sequence=True, return_embeddings=True)
#         )

#         embeddings = outputs.embeddings  # (B, L, 960)

#         # 4. Attention mask (ESM pad token = 1)
#         attention_mask = (encoded.sequence != 1).long()

#         # 5. Classifier
#         return self.classifier(embeddings, attention_mask)


# def get_model(num_classes):
#     # Load model
#     client = ESMC.from_pretrained(config.model_name_or_path).to(config.device)
    
#     # LoRA config
#     peft_config = LoraConfig(
#         task_type=TaskType.FEATURE_EXTRACTION,
#         inference_mode=False,
#         r=config.lora_r,
#         lora_alpha=config.lora_alpha,
#         lora_dropout=config.lora_dropout,
#         target_modules=["attn.layernorm_qkv.1", "attn.out_proj", "ffn.1", "ffn.3"],
#     )
#     client = get_peft_model(client, peft_config)
    
#     classifier = DeepProteinClassifier(num_classes).to(config.device)
#     model = ESMCClassifier(client, classifier).to(config.device)
    
#     return model