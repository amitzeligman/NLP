import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from models import DummyEncoder
from text_utils import *
from torch.optim import Adam
import torch.nn as nn


# Configurations
class Configuration:

    def __init__(self):
        self.hidden_size = 768
        self.vocab_size = 30522
        self.epochs = 100
        self.lr = 1e-4


config = Configuration()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Input/Labels pre processing
sent = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokens_tensor, segments_tensors, labels = sentences_to_bert_inputs(sent)

# Define models
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
enc = model.bert
enc_dummy = DummyEncoder(config)
classifier = model.cls

# Upload to device
tokens_tensor = tokens_tensor.to(device)
segments_tensors = segments_tensors.to(device)
labels = labels.to(device)
classifier = classifier.to(device)
enc = enc.to(device)
enc_dummy = enc_dummy.to(device)


# Training
loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
classifier.requires_grad_(False)  # Freeze classifier
optimizer = Adam(params=enc_dummy.parameters(), lr=config.lr)

classifier.train()
enc_dummy.train()

for epoch in range(config.epochs):
    enc_dummy.zero_grad()
    features = enc_dummy(tokens_tensor)[0]
    predictions_scores = classifier(features)
    masked_lm_loss = loss_fct(predictions_scores.view(-1, config.vocab_size), labels.view(-1))
    print('{}/{}'.format(epoch, config.epochs), 'Loss:', masked_lm_loss.item())
    masked_lm_loss.backward()
    optimizer.step()


# Predict
enc_dummy.eval()
classifier.eval()

with torch.no_grad():
    features = enc_dummy(tokens_tensor)
    predictions = classifier(features[0])

    predicted_tokens = list(map(lambda p: tokenizer.convert_ids_to_tokens(p), [torch.argmax(predictions, -1)[0].numpy()]))
    print(predicted_tokens)








