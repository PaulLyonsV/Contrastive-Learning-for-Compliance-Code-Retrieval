import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_embed, pos_embed):
        # anchor_embed, pos_embed shape [batch_size, hidden_dim]
        # sim_matrix[i][j] = similarity between anchor i and positive j
        sim_matrix = (anchor_embed @ pos_embed.T) / self.temperature  
        
        # correct answer for anchor i is positive i, so labels are just 0,1,2,...,B-1
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        # track diagonal (correct pairs) vs off-diagonal mean (negatives)
        sim_pos = sim_matrix.diagonal().mean() * self.temperature
        sim_neg = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / \
                  (sim_matrix.numel() - sim_matrix.shape[0])
        sim_neg = sim_neg * self.temperature
        
        return loss, sim_pos, sim_neg