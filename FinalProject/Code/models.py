from torch import nn


class DummyEncoder(nn.Module):

    def __init__(self, config):
        super(DummyEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(in_features=config.hidden_size * 2, out_features=config.hidden_size)

    def forward(self, x):

        embedding = self.word_embeddings(x)
        enc = self.lstm(embedding)
        enc = self.linear(enc[0])

        return enc
