
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertModel
from .config import CONFIG

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

class Model(pl.LightningModule):

  def __init__(self, n_classes: int=len(LABEL_COLUMNS)):
    super().__init__()
    self.bert = BertModel.from_pretrained(CONFIG['BERT_PRETRAINED_PATH'], return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.criterion = nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)    
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output