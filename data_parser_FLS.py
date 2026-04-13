import pandas as pd
import torch
import re
from torch.utils.data import Dataset

class BuffaloComplianceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        #load and clean
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Drop rows w/ na
        df = df.dropna(subset=['Code', 'Description', 'Comments'])
        
        # Strip HTML tags and &nbsp (non-breaking space) from comments.
        # (comments are the contractor descriptions that are matched
        # to compliance codes in the dataset).
        df['Comments'] = df['Comments'].apply(self._clean_html)
        
        # filter out rows that became empty after cleaning
        df = df[df['Comments'].str.strip() != '']
        
        # build the anchor (compliance code + description of code) 
        df['Anchor_Text'] = df['Code'].astype(str) + " - " + df['Description'].astype(str)
        
        # reset index
        self.df = df.reset_index(drop=True)
        
    def _clean_html(self, text):
        # Remove HTML tags (<tk>), &nbsp, remove empty space
        clean = re.sub(r'<[^>]+>', ' ', str(text))
        clean = re.sub(r'&[a-z]+;', ' ', clean) 
        return re.sub(r'\s+', ' ', clean).strip()

    def __len__(self): 
        return len(self.df)
    
    def __getitem__(self, idx):
        current_row = self.df.iloc[idx]
        anchor_text = current_row['Anchor_Text']
        positive_text = current_row['Comments']

        #match tokenizer in train files. 
        anchor_enc = self.tokenizer(anchor_text, truncation=True, 
                                    max_length=self.seq_len,
                                    padding="max_length", return_tensors="pt")
        
        pos_enc = self.tokenizer(positive_text, truncation=True, 
                                max_length=self.seq_len,
                                padding="max_length", return_tensors="pt")

        return {
            "anchor_input_ids": anchor_enc['input_ids'].squeeze(0),
            "anchor_attention_mask": anchor_enc['attention_mask'].squeeze(0),
            "positive_input_ids": pos_enc['input_ids'].squeeze(0),
            "positive_attention_mask": pos_enc['attention_mask'].squeeze(0),
        }