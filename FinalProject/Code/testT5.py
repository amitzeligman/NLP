import torch
import transformers
import math


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
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

transformers.T5Model
tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small')

model = transformers.T5WithLMHeadModel.from_pretrained('t5-small')
#model.eval()
# model = transformers.T5Model.from_pretrained('t5-base')

inputs = 'translate English to German: "Luigi often said to me that he neverwanted the brothers to end up in court," she wrote.'
outputs = 'target: "Luigi sagte oft zu mir, dass er nie wollte, dass die Brüder vor Gericht landen", schrieb sie.'
model_kwargs = {}
encoder_input_ids = tokenizer.encode(inputs)
encoder_input_ids = torch.tensor(encoder_input_ids).unsqueeze_(0)

decoder_attention_mask = torch.zeros_like(encoder_input_ids)
decoder_input_ids = tokenizer.encode(outputs)
decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze_(0)

out = model(input_ids=encoder_input_ids)#, decoder_input_ids=decoder_input_ids)

logits = torch.nn.Softmax(-1)(out[0].squeeze(0))


pred = beam_search_decoder(logits.detach().numpy(), 1)


predicted_tokens = list(map(lambda p: tokenizer.decode(p), pred[0][0]))
#predicted_tokens = list(map(lambda p: tokenizer.decode(p), [torch.argmax(out[0], -1)[0].numpy()]))
print(predicted_tokens)

print(out[0].shape)
print(out[1].shape)
