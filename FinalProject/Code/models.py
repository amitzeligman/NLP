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


    def forward(self, x):

        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        c_out = self.cnn(c_in)

        r_out = c_out.view(-1, batch_size, 4096)

        #r_out, (h_n, h_c) = self.rnn(c_out.view(-1, batch_size, 4096))


        return r_out

