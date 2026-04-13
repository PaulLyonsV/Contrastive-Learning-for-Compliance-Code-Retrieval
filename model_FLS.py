import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(
            cfg['model_id'], 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    def last_token_pooling(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False 
        )

        last_hidden = outputs.last_hidden_state
        
        # finds index of the last valid token ignoring padding
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sequence_lengths = sequence_lengths.clamp(min=0)
        
        #input_ids has shape [batch_size, sequence_length]. 
        # .shape[0] gives batch size, 
        # torch.arange gives array counting up to batch_size
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
    
        #last_hidden has shape [batch_size, sequence_length, hidden dim].
        #we return 2d tensor with shape [batch size, hidden dim]
        last_token_emb = last_hidden[batch_indices, sequence_lengths]
        
        return last_token_emb
        
    def get_embedding(self, input_ids, attention_mask):
        raw_embeddings = self.last_token_pooling(input_ids, attention_mask)
        normalized_embeddings = F.normalize(raw_embeddings.to(torch.float32), p=2, dim=1)
        
        return normalized_embeddings