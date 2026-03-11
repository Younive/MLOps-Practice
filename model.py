import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import wandb
import torchmetrics

class ColaModel(pl.LightningModule):
    def __init__(self, model_name='google/bert_uncased_L-2_H-128_A-2', lr=1e-2):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.W = torch.nn.Linear(self.bert.config.hidden_size,2)
        self.num_classes = 2
        self.train_acc_metric = torchmetrics.Accuracy()
        self.val_acc_metric = torchmetrics.Accuracy()
        self.precision_macro_metric = torchmetrics.Precision(num_classes=self.num_classes, average='macro')
        self.recall_macro_metric = torchmetrics.Recall(num_classes=self.num_classes, average='macro')
        self.f1_macro_metric = torchmetrics.F1(num_classes=self.num_classes, average='macro')
        self.precision_micro_metric = torchmetrics.Precision(num_classes=self.num_classes, average='micro')
        self.recall_micro_metric = torchmetrics.Recall(num_classes=self.num_classes, average='micro')
    
    def forward(self, input_ids, attention_mask, label=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=label)
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        # loss = F.cross_entropy(logits, batch['labels'])
        preds = torch.argmax(outputs.logits, dim=1)
        train_acc = self.train_acc_metric(preds, batch['labels'])
        self.log('train/acc', train_acc, prog_bar=True) 
        self.log('train/loss', outputs.loss, prog_bar=True)
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        label = batch['labels']
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], batch['labels'])
        # loss = F.cross_entropy(logits, batch['labels'])
        pred = torch.argmax(outputs.logits, dim=1)

        val_acc = self.val_acc_metric(pred, label)
        precision_macro = self.precision_macro_metric(pred, label)
        recall_macro = self.recall_macro_metric(pred, label)
        f1_macro = self.f1_macro_metric(pred, label)
        precision_micro = self.precision_micro_metric(pred, label)
        recall_micro = self.recall_micro_metric(pred, label)
        self.log('val_loss', outputs.loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_epoch=True)
        self.log('val_precision_macro', precision_macro, prog_bar=True, on_epoch=True)
        self.log('val_recall_macro', recall_macro, prog_bar=True, on_epoch=True)
        self.log('val_f1_macro', f1_macro, prog_bar=True, on_epoch=True)
        self.log('val_precision_micro', precision_micro, prog_bar=True, on_epoch=True)
        self.log('val_recall_micro', recall_micro, prog_bar=True, on_epoch=True)
        return {'labels': label, 'logits': outputs.logits}
    
    def validation_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs], dim=0)
        logits = torch.cat([x['logits'] for x in outputs], dim=0)
        preds = torch.argmax(logits, dim=1)
        self.logger.experiment.log({
            'confusion_matrix': wandb.plot.confusion_matrix(
                probs = logits.numpy(),
                y_true = labels.numpy(),
                class_names = ['not_acceptable', 'acceptable']
            )
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)