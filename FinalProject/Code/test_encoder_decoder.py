import torch
import transformers
import math
from torch.nn import CrossEntropyLoss
import logging
logging.basicConfig(level=logging.INFO)


def beam_search_decoder(data, k):

    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -math.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[2])
        # select k best
        sequences = ordered[:k]
    return sequences


#tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
#model = transformers.Model2Model.from_pretrained('bert-base-uncased')

#tokenizer = transformers.XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
#tokenizer = transformers.XLMTokenizer.from_pretrained('bert-base-uncased')
tokenizer_bert = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = tokenizer_bert
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
decoder = transformers.BertForMaskedLM(config)
#model = transformers.PreTrainedEncoderDecoder.from_pretrained('xlm-mlm-en-2048', 'bert-base-uncased')
model = transformers.Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder)


encoder_inputs = "I like cats and dogs"
decoder_inputs = "I like cats and dogs"
model_kwargs = {}
encoder_input_ids = tokenizer.encode(encoder_inputs)
encoder_input_ids = torch.tensor(encoder_input_ids).unsqueeze_(0)
decoder_input_ids = tokenizer_bert.encode(decoder_inputs)
decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze_(0)


optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
loss_fct = CrossEntropyLoss()
model.train()

#model.load_state_dict(torch.load('/Users/amitzeligman/NLP_models/test.pt'))

#encoder_attention_mask = torch.ones_like(encoder_input_ids)
#decoder_attention_mask = 1#torch.ones_like(encoder_input_ids)

for i in range(100):
    optimizer.zero_grad()
    loss = 0

    for l in range(decoder_input_ids.shape[-1] - 1):

        outputs = model(encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids[:, :l+1].view(1, -1))
        loss += loss_fct(outputs[0].view(l+1, -1), decoder_input_ids[:, 1:l+2].view(-1)) / (l + 1)

    print(loss.item())
    loss.backward()
    optimizer.step()

#torch.save(model.state_dict(), '/Users/amitzeligman/NLP_models/test.pt')

# Predicting

model.eval()
#decoder_input_ids = tokenizer_bert.encode("[CLS]")
#decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze_(0)


for l in range(decoder_input_ids.shape[-1]):
    outputs = model(encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids[:, :l+1].view(1, -1))
    pred = torch.argmax(outputs[0], -1).view(-1, 1)

    if tokenizer.decode(pred[-1]) == '[SEP]':
        break

    #decoder_input_ids = torch.cat([decoder_input_ids, pred[..., -1].unsqueeze(-1)], dim=-1)

    #logits = torch.nn.Softmax(-1)(outputs[0].squeeze(0))
    #pred = beam_search_decoder(logits.detach().numpy(), 1)
    #predicted_tokens = list(map(lambda p: tokenizer_bert.decode(p), pred[0][0]))
    predicted_tokens = tokenizer_bert.decode(pred)

    print(predicted_tokens)



#predicted_tokens = list(map(lambda p: tokenizer.decode(p), [torch.argmax(out[0], -1)[0].numpy()]))



