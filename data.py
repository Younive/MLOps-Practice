import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, load_dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, model_name='google/bert_uncased_L-2_H-128_A-2', batch_size=32):
        super().__init__()
        
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def prepare_data(self):
        cola_data = load_dataset('glue', 'cola')
        self.train_data = cola_data['train']
        self.val_data = cola_data['validation']

    def tokenize_data(self, data):
        return self.tokenizer(
            data['sentence'], 
            padding='max_length', 
            truncation=True, 
            max_length=128
        )
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            self.train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            self.val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)