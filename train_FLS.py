import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model, TaskType


from data_parser_FLS import BuffaloComplianceDataset
from model_FLS import Model
from loss_FLS import InfoNCELoss

config = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./checkpoints/compliance_retrieval_experiment",
    
    # Buffalo dataset
    "data_path": "Code_Violations.csv",
    "max_seq_len": 512, 
    
    "model_id": "Qwen/Qwen1.5-7B",
    
    "batch_size": 32, 
    "grad_accum": 1, 
    "num_epochs": 4, 
    "learning_rate": 2e-5,

    "lora_rank": 32, 
    "lora_alpha": 64, 
    "lora_dropout": 0.05,
    
    "temperature": 0.1 
}

# Recall at K evaluation. 
def recall_at_k(anchors, positives, k_values=[1, 5, 10]):
    # N is number of anchors
    # anchors shape: [N, 4096], positives shape: [N, 4096]
    # For each anchor (description), retrieve top-k from all
    # positives, check if correct one is in there
    sim_matrix = anchors @ positives.T  # [N, N]
    results = {}
    for k in k_values:
        top_k = sim_matrix.topk(k, dim=1).indices  # [N, k]
        correct = torch.arange(anchors.shape[0]).unsqueeze(1)  # [N, 1]
        hits = (top_k == correct).any(dim=1).float()
        results[f"Recall@{k}"] = hits.mean().item()
    return results

def train(cfg):
    set_seed(cfg['seed'])
    os.makedirs(cfg['output_dir'], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = BuffaloComplianceDataset(
        csv_path=cfg['data_path'],
        tokenizer=tokenizer,
        seq_len=cfg['max_seq_len']
    )

    train_size = 12000
    val_size = 2400
    remainder = len(dataset) - train_size - val_size

    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, remainder], generator=torch.Generator().manual_seed(cfg['seed']))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
    )
    
    model = Model(cfg)
    model.to(cfg['device'])

    model.backbone.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=cfg['lora_rank'], 
        lora_alpha=cfg['lora_alpha'], 
        lora_dropout=cfg['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model.backbone = get_peft_model(model.backbone, peft_config)
    model.backbone.print_trainable_parameters()

    optimizer = AdamW(
        model.parameters(), 
        lr=cfg['learning_rate'], 
        weight_decay=0.01
    )

    loss_fn = InfoNCELoss(temperature=cfg['temperature'])

    epoch_sim_pos = []
    epoch_sim_neg = []
    
    for epoch in range(cfg['num_epochs']):
        model.train() 
        
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1} out of {cfg['num_epochs']}")
        for step, batch in enumerate(pbar):
            #shape [32, 512]
            a_ids = batch['anchor_input_ids'].to(cfg['device'])
            a_mask = batch['anchor_attention_mask'].to(cfg['device'])
            
            p_ids = batch['positive_input_ids'].to(cfg['device'])
            p_mask = batch['positive_attention_mask'].to(cfg['device'])
    
            # Generate the 2 embeddings
            #shape [32, 512] -> [32, 4096] (batch size, Qwen hidden dim of last token)
            embed_anchor = model.get_embedding(a_ids, a_mask)
            embed_pos = model.get_embedding(p_ids, p_mask)
        
            # Calculate loss and track similarities
            loss, sim_p, sim_n = loss_fn(embed_anchor, embed_pos)
            
            loss = loss / cfg['grad_accum']
            loss.backward()
            
            #.item to extract raw floats. 
            epoch_sim_pos.append(sim_p.item())
            epoch_sim_neg.append(sim_n.item())

            if (step + 1) % cfg['grad_accum'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                #20 step moving average
                curr_sim_p = np.mean(epoch_sim_pos[-20:]) if epoch_sim_pos else 0
                curr_sim_n = np.mean(epoch_sim_neg[-20:]) if epoch_sim_neg else 0
                
                pbar.set_postfix({
                    "Loss": f"{loss.item() * cfg['grad_accum']:.4f}",
                    "Sim(A,P)": f"{curr_sim_p:.3f}", 
                    "Sim(A,N)": f"{curr_sim_n:.3f}"
                })
        
        # Save checkpoints after each epoch. 
        ckpt_path = os.path.join(cfg['output_dir'], f"epoch_{epoch+1}")
        model.backbone.save_pretrained(ckpt_path) 
        tokenizer.save_pretrained(ckpt_path)

        model.eval() 
        val_sim_pos, val_sim_neg = [], []

        val_anchors = []
        val_positives = [] 

    #Evaluate validation set and save their embeddings.
        with torch.no_grad(): 
            for val_batch in val_loader: 
                v_a_ids = val_batch['anchor_input_ids'].to(cfg['device'])
                v_a_mask = val_batch['anchor_attention_mask'].to(cfg['device'])
                v_p_ids = val_batch['positive_input_ids'].to(cfg['device'])
                v_p_mask = val_batch['positive_attention_mask'].to(cfg['device'])
                    
                v_embed_a = model.get_embedding(v_a_ids, v_a_mask)
                v_embed_p = model.get_embedding(v_p_ids, v_p_mask)
                    
                _, v_sim_p, v_sim_n = loss_fn(v_embed_a, v_embed_p)
                    
                val_sim_pos.append(v_sim_p.item())
                val_sim_neg.append(v_sim_n.item())
                val_anchors.append(v_embed_a.cpu())
                val_positives.append(v_embed_p.cpu())

        print(f"Sim (anchor, positive): {np.mean(val_sim_pos):.3f}, sim(anchor,negative): {np.mean(val_sim_neg):.3f}")
        final_anchors = torch.cat(val_anchors, dim=0)
        final_positives = torch.cat(val_positives, dim=0)

        metrics = recall_at_k(final_anchors, final_positives)
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")

        torch.save(final_anchors, os.path.join(ckpt_path, "val_anchors.pt"))
        torch.save(final_positives, os.path.join(ckpt_path, "val_postives.pt"))
        epoch_sim_pos, epoch_sim_neg = [], []

if __name__ == "__main__":
    train(config)