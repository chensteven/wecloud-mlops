
import pytorch_lightning as pl
import torch.nn as nn

LABEL_COLUMNS = {1: "World",
                2: "Sports",
                3: "Business",
                4: "Sci/Tec"}
              
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size=95811, embed_dim=64, num_class=4):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)