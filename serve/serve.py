import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AutoModel, AutoTokenizer, BertConfig,
                          BertModel, BertTokenizer)

LABEL_COLUMNS = ['ai', 'algorithm', 'algorithms', 'application', 'applications',
       'bandits', 'bayesian', 'best practice', 'best practices', 'big data',
       'book', 'books', 'business', 'career', 'cloud', 'clustering',
       'computer science', 'computer vision', 'course', 'courses', 'dashboard',
       'data', 'data engineering', 'data mining', 'data science', 'data shift',
       'data visualization', 'dataset', 'decision making', 'deep learning',
       'design patterns', 'designed patterns', 'engineering',
       'experimentation', 'forecasting', 'gis', 'graph', 'guide', 'hardware',
       'healthcare', 'hiring', 'how-to', 'improvement', 'industry',
       'infrastructure', 'interview', 'interviews', 'iot', 'libraries',
       'library', 'machine learning', 'mathematics', 'mlops',
       'mobile development', 'neural network', 'nlp', 'notes', 'paper',
       'papers', 'predictive analytics', 'probability', 'product',
       'programming', 'python', 'question', 'rant', 'real-time',
       'recommendation system', 'reinforcement learning', 'research',
       'resource', 'review', 'robotics', 'salary', 'simulation', 'software',
       'sql', 'statistics', 'system design', 'theory', 'tools', 'trading',
       'tutorial', 'tutorials', 'use case', 'use cases', 'ux', 'visualization',
       'web development']
BERT_MODEL_NAME = "bert-base-cased"

import torch
from torchmetrics.functional import accuracy, auroc, f1_score
from transformers import AdamW, BertModel
from transformers import BertTokenizerFast as BertTokenizer
from transformers import get_linear_schedule_with_warmup


class TitleTagger(pl.LightningModule):

  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)    
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def training_epoch_end(self, outputs):
    
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)

    for i, name in enumerate(LABEL_COLUMNS):
      class_roc_auc = auroc(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)


  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

import pytorch_lightning as pl
import torch
from transformers import (AdamW, AutoModel, AutoTokenizer, BertConfig,
                          BertModel, BertTokenizer)


class PythonPredictor:
    def __init__(self):
        self.device = "cpu"
        self.BERT_MODEL_NAME = "bert-base-cased"
        self.LABEL_COLUMNS = ['ai', 'algorithm', 'algorithms', 'application', 'applications',
       'bandits', 'bayesian', 'best practice', 'best practices', 'big data',
       'book', 'books', 'business', 'career', 'cloud', 'clustering',
       'computer science', 'computer vision', 'course', 'courses', 'dashboard',
       'data', 'data engineering', 'data mining', 'data science', 'data shift',
       'data visualization', 'dataset', 'decision making', 'deep learning',
       'design patterns', 'designed patterns', 'engineering',
       'experimentation', 'forecasting', 'gis', 'graph', 'guide', 'hardware',
       'healthcare', 'hiring', 'how-to', 'improvement', 'industry',
       'infrastructure', 'interview', 'interviews', 'iot', 'libraries',
       'library', 'machine learning', 'mathematics', 'mlops',
       'mobile development', 'neural network', 'nlp', 'notes', 'paper',
       'papers', 'predictive analytics', 'probability', 'product',
       'programming', 'python', 'question', 'rant', 'real-time',
       'recommendation system', 'reinforcement learning', 'research',
       'resource', 'review', 'robotics', 'salary', 'simulation', 'software',
       'sql', 'statistics', 'system design', 'theory', 'tools', 'trading',
       'tutorial', 'tutorials', 'use case', 'use cases', 'ux', 'visualization',
       'web development']
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL_NAME)
        self.model = TitleTagger(
          n_classes=len(self.LABEL_COLUMNS),
          n_warmup_steps=10,
          n_training_steps=50 
        )
        # from checkpoint
        # self.checkpoint = torch.load("checkpoints/best-checkpoint.ckpt")
        # self.model.load_state_dict(self.checkpoint['state_dict'])

        # from model
        self.checkpoint = torch.load("model/model_1.pt")
        self.model.load_state_dict(self.checkpoint)
        self.model.eval()
        self.THRESHOLD = 0.5

    def predict(self, payload):
        encoding = self.tokenizer.encode_plus(
          payload,
          add_special_tokens=True,
          max_length=512,
          return_token_type_ids=False,
          padding="max_length",
          return_attention_mask=True,
          return_tensors='pt',
        )
        _, test_prediction = self.model(encoding["input_ids"], encoding["attention_mask"])
        test_prediction = test_prediction.flatten().detach().numpy()
        
        results = []
        for label, prediction in zip(self.LABEL_COLUMNS, test_prediction):
          if prediction < self.THRESHOLD:
            continue
          print(f"{label}: {prediction}")
          results.append({label: prediction})

        return results

from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong!"}


@app.get("/predict/{sentence}")
def predict(sentence: str):
    print(sentence)
    pred = PythonPredictor().predict(sentence)
    return {"message": str(pred)}

