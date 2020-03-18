from torch import nn
import torchvision .models as models


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


class IdentityEmbedding(nn.Module):
    def __init__(self, input_argument_name):
        super(IdentityEmbedding, self).__init__()
        self.input_arg = input_argument_name

    def forward(self, **kwargs):
        return kwargs[self.input_arg]


########## Video  #########

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG_LSTM(nn.Module):
    def __init__(self, hidden_size, n_layers, dropt, bi):
        super(VGG_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = n_layers

        dim_feats = 4096

        self.cnn = models.vgg19(pretrained=True)
        self.cnn.classifier[-1] = Identity()
        self.rnn = nn.LSTM(
            input_size=dim_feats,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropt,
            bidirectional=bi)
        self.bidirectional = bi
        if self.bidirectional:
            self.fc = nn.Linear(2, 1)

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        c_out = self.cnn(c_in)

        out = self.rnn(c_out.view(-1, batch_size, 4096))

        hidden = out[1][0]
        last_hidden = hidden[-2:, ...].view(batch_size, self.hidden_size, 2)
        if self.bidirectional:
            last_hidden = self.fc(last_hidden).view(batch_size, 1, self.hidden_size)

        return out, last_hidden

