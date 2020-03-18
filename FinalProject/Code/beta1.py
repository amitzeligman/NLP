from FinalProject.Code.vid_models import *
import cv2
import torch
import transformers
import math
from torch.nn import CrossEntropyLoss
import logging
#logging.basicConfig(level=logging.INFO)


# TODO - algorithmic:
#  1. Teacher forcing True/False in training.
#  2. Beam search in prediction.
#  3. Embedding loss for video embeddings.
#  4. Design video embeddings - Last hidden state or outputs of LSTM?, frames per id rate?.
#  5. Check cross-attention mechanism.


device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Nlp model definitions
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
decoder = transformers.BertForMaskedLM(config)
nlp_model = transformers.Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder)
nlp_model.encoder.embeddings = IdentityEmbedding('input_ids')
nlp_model.to(device)
nlp_model.train()

# Video model definitions
vid_model = VGG_LSTM(hidden_size=768, n_layers=8, dropt=0.25, bi=True)
vid_model.train()
vid_model = vid_model.to(device)
frames_per_inference = 5

# Over fit nlp sentence to train
decoder_inputs = "SO THE TECHNOLOGY IS THERE"
decoder_input_ids = tokenizer.encode(decoder_inputs)
decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze_(0)

# Optimizers and loss function
optimizer_nlp = torch.optim.Adam(params=nlp_model.parameters(), lr=1e-4)
optimizer_vid = torch.optim.Adam(params=vid_model.parameters(), lr=1e-4)
loss_fct = CrossEntropyLoss()

# load models
vid_model.load_state_dict(torch.load('/media/cs-dl/HD_6TB/Data/Trained_models_nlp/vid_model.py'))
nlp_model.load_state_dict(torch.load('/media/cs-dl/HD_6TB/Data/Trained_models_nlp/nlp_model.py'))

epochs = 5

# Training

for _ in range(epochs):

    optimizer_nlp.zero_grad()
    optimizer_vid.zero_grad()
    loss = 0

    # Video embedding

    cap = cv2.VideoCapture('/media/cs-dl/HD_6TB/Data/Amit/trainval/0af00UcTOSc/50002.mp4')

    idx = 0
    out = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        frame = torch.tensor(frame / 255, dtype=torch.float32)
        frame = frame.permute(-1, 0, 1).unsqueeze(0).unsqueeze(0)
        if idx == 0:
            vid = frame
        else:
            vid = torch.cat([vid, frame], dim=1)

        if not idx % (frames_per_inference - 1) and idx:
            vid = vid.to('cuda')
            output, last_hidden = vid_model(vid)
            if out is None:
                out = last_hidden
            else:
                pass
                out = torch.cat([out, last_hidden], dim=1)
                #print(out.shape)
            idx = 0
        else:
            idx += 1

    cap.release()

# Nlp prediction form video embeddings
    encoder_attention_mask = torch.ones(out.shape[:-1], device=device)
    decoder_input_ids = decoder_input_ids.to(device)

    for l in range(decoder_input_ids.shape[-1] - 1):

        outputs = nlp_model(encoder_input_ids=out, decoder_input_ids=decoder_input_ids[:, :l+1].view(1, -1),
                            encoder_attention_mask=encoder_attention_mask)
        loss += loss_fct(outputs[0].view(l+1, -1), decoder_input_ids[:, 1:l+2].view(-1)) / (l + 1)

    print(loss.item())
    loss.backward()
    optimizer_nlp.step()
    optimizer_vid.step()

    torch.save(vid_model.state_dict(), '/media/cs-dl/HD_6TB/Data/Trained_models_nlp/vid_model.py')
    torch.save(nlp_model.state_dict(), '/media/cs-dl/HD_6TB/Data/Trained_models_nlp/nlp_model.py')


# Testing

vid_model.eval()
nlp_model.eval()


for _ in range(1):
    cap = cv2.VideoCapture('/media/cs-dl/HD_6TB/Data/Amit/trainval/0af00UcTOSc/50002.mp4')

    idx = 0
    out = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        frame = torch.tensor(frame / 255, dtype=torch.float32)
        frame = frame.permute(-1, 0, 1).unsqueeze(0).unsqueeze(0)
        if idx == 0:
            vid = frame
        else:
            vid = torch.cat([vid, frame], dim=1)

        if not idx % (frames_per_inference - 1) and idx:
            vid = vid.to('cuda')
            output, last_hidden = vid_model(vid)
            if out is None:
                out = last_hidden
            else:
                pass
                out = torch.cat([out, last_hidden], dim=1)
                #print(out.shape)
            idx = 0
        else:
            idx += 1

    cap.release()

    encoder_attention_mask = torch.ones(out.shape[:-1], device=device)
    decoder_input_ids = decoder_input_ids.to(device)

for l in range(decoder_input_ids.shape[-1]):
    outputs = nlp_model(encoder_input_ids=out, decoder_input_ids=decoder_input_ids[:, :l + 1].view(1, -1),
                        encoder_attention_mask=encoder_attention_mask)
    pred = torch.argmax(outputs[0], -1).view(-1, 1)

    if tokenizer.decode(pred[-1]) == '[SEP]':
        break
    predicted_tokens = tokenizer.decode(pred)

    print(predicted_tokens)






