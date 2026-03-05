import torch
import pytorch_lightning as pl
from model import ColaModel
from data import DataModule

class ColaPredictor(pl.LightningModule):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(self.model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ['Unacceptable', 'Acceptable']

    def predict(self, text):
        inference_sample = {'sentence': text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor(processed['input_ids']).unsqueeze(0),
            torch.tensor(processed['attention_mask']).unsqueeze(0)
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for label, score in zip(scores, self.labels):
            predictions.append({'label': label, 'score': score})
        return predictions
    
if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor('./models/cola_model/version_1/checkpoints/epoch=0-step=0.ckpt')
    print(predictor.predict(sentence))