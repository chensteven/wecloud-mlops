from .config import CONFIG
from .model import LABEL_COLUMNS
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def predict(package, text):
  tokenizer = get_tokenizer('basic_english')
  vocab = package['vocab']

  text_pipeline = lambda x: vocab(tokenizer(x))

  model = package['model']

  with torch.no_grad():
      text = torch.tensor(text_pipeline(text))
      output = model(text, torch.tensor([0]))
      return LABEL_COLUMNS[output.argmax(1).item() + 1]