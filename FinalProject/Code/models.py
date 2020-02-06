from torch import nn


class DummyEncoder(nn.Module):

    def __init__(self, config):
        super(DummyEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)

    def forward(self, x):

        embedding = self.word_embeddings(x)
        enc = self.lstm(embedding)

        return enc
