from .config import CONFIG
from .model import LABEL_COLUMNS
import numpy as np
import torch

# def preprocess(package: dict, input: list) -> list:
#     """
#     Preprocess data before running with model, for example scaling and doing one hot encoding
#     :param package: dict from fastapi state including model and preocessing objects
#     :param package: list of input to be proprocessed
#     :return: list of proprocessed input
#     """

#     # scale the data based with scaler fit during training 
#     tokenizer = package['tokenizer']
#     input = tokenizer.encode_plus(
#         input,
#         add_special_tokens=True,
#         max_length=512,
#         return_token_type_ids=False,
#         padding="max_length",
#         return_attention_mask=True,
#         return_tensors='pt',
#       )

#     return input

# def predict(package: dict, input: list) -> list:
#     """
#     Run model and get result
#     :param package: dict from fastapi state including model and preocessing objects
#     :param package: list of input values
#     :return: list of model output
#     """

#     # process data
#     X = preprocess(package, input)

#     # run model
#     model = package['model']
#     _, test_prediction = model(X["input_ids"], X["attention_mask"])
#     test_prediction = test_prediction.flatten().detach().numpy()
    
#     y_pred = []
#     thresold = 0.5
#     for label, prediction in zip(LABEL_COLUMNS, test_prediction):
#       if prediction < thresold:
#         continue
#       # convert numpy float to python float
#       y_pred.append({label: prediction.item()})

#     return y_pred



from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def predict(package, text):
  tokenizer = get_tokenizer('basic_english')
  vocab = package['tokenizer']

  text_pipeline = lambda x: vocab(tokenizer(x))

  model = package['model']

  with torch.no_grad():
      text = torch.tensor(text_pipeline(text))
      output = model(text, torch.tensor([0]))
      return LABEL_COLUMNS[output.argmax(1).item() + 1]