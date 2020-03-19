import torchvision .models as models
import torch.nn as nn
import transformers
import torch


class IdentityEmbedding(nn.Module):
    def __init__(self, input_argument_name):
        super(IdentityEmbedding, self).__init__()
        self.input_arg = input_argument_name

    def forward(self, **kwargs):
        return kwargs[self.input_arg]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGGLSTM(nn.Module):
    def __init__(self, hidden_size, n_layers, dropt, bi):
        super(VGGLSTM, self).__init__()

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

        # Freeze the VGG model
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        c_out = self.cnn(c_in)

        out = self.rnn(c_out.view(-1, batch_size, 4096))

        hidden = out[1][0]
        last_hidden = hidden[-2:, ...].view(batch_size, self.hidden_size, 2)
        # TODO - check indeed this indices return the desired hidden state.
        if self.bidirectional:
            last_hidden = self.fc(out[0].view(timesteps, self.hidden_size, 2)).view(1, timesteps, self.hidden_size)
            #last_hidden = self.fc(last_hidden).view(batch_size, 1, self.hidden_size) # TODO

        return out, last_hidden


class NLPModel(nn.Module):

    def __init__(self):
        super().__init__()
        config = transformers.BertConfig.from_pretrained('bert-base-uncased')
        config.is_decoder = True
        decoder = transformers.BertForMaskedLM(config)
        model = transformers.Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder)
        model.encoder.embeddings = IdentityEmbedding('input_ids')
        self.model = model

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_attention_mask = torch.ones(encoder_inputs.shape[:-1], device=encoder_inputs.device)
        outputs = self.model(encoder_input_ids=encoder_inputs,
                             decoder_input_ids=decoder_inputs,
                             encoder_attention_mask=encoder_attention_mask)

        return outputs


class FullModel(nn.Module):

    def __init__(self, hidden_size=768, n_layers=2, drop_out=0.25, bidirectional=True):
        super().__init__()
        self.vid_model = VGGLSTM(hidden_size, n_layers, drop_out, bidirectional)
        self.nlp_model = NLPModel()

    def forward(self, video_inputs, decoder_inputs):

        video_embeddings = self.vid_model(video_inputs)[1]  # Take only the last hidden state (position 0 is the seq outputs).
        text = self.nlp_model(video_embeddings, decoder_inputs)

        return text


class MultiGpuModel(nn.DataParallel):
    """
    Wrapper to model class for using multi-gpu model.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)







