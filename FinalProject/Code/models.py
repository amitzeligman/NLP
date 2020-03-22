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
    """
    Video model - from raw video to video embedding.
    """
    def __init__(self, hidden_size, n_layers, dropt, bi):
        super(VGGLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = n_layers

        self.features_dim = 4096

        self.cnn = models.vgg19(pretrained=True)
        self.cnn.classifier[-1] = Identity()
        self.rnn = nn.LSTM(
            input_size=self.features_dim ,
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

        self.frames_per_token = 10

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        assert batch_size == 1, 'Currently only batch_size=1 is supported'
        c_in = x.view(batch_size * time_steps, C, H, W)

        # Extract features per each frame independently.
        c_out = self.cnn(c_in)
        out = self.rnn(c_out.view(-1, batch_size, self.features_dim))[0]
        out = out.view(time_steps, batch_size, self.hidden_size, 2)
        out = self.fc(out).view(batch_size, time_steps, self.hidden_size)


        # ADD DESCRIPTION
        #final_output = None
        #for i in range(time_steps // self.frames_per_token + 1 * (time_steps % self.frames_per_token is True)):
        #    current_time_steps = self.frames_per_token if i <= time_steps // self.frames_per_token else time_steps % self.frames_per_token
        #    c = c_out[i * self.frames_per_token: (i + 1) * current_time_steps]
        #    out = self.rnn(c.view(-1, batch_size, self.features_dim))
        #    hidden = out[1][0]
        #    last_hidden = hidden[-2:, ...].view(batch_size, self.hidden_size, 2)
        #    if self.bidirectional:
        #        last_hidden = self.fc(last_hidden).view(batch_size, 1, self.hidden_size)

        #    if not i:
        #        final_output = last_hidden
        #    else:
        #        final_output = torch.cat([final_output, last_hidden], dim=1)

        return out


class NLPModel(nn.Module):
    """
    NLP model - from video embedding (encoder inputs) to text.
    """

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
    """
    Full lip reading model from raw video to text.
    """

    def __init__(self, hidden_size=768, n_layers=8, drop_out=0.25, bidirectional=True):
        super().__init__()
        self.vid_model = VGGLSTM(hidden_size, n_layers, drop_out, bidirectional)
        self.nlp_model = NLPModel()

    def forward(self, video_inputs, decoder_inputs):

        video_embeddings = self.vid_model(video_inputs)
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







