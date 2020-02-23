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


#tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
#model = transformers.Model2Model.from_pretrained('bert-base-uncased')

tokenizer = transformers.XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
tokenizer_bert = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.PreTrainedEncoderDecoder.from_pretrained('xlm-mlm-en-2048', 'bert-base-uncased')

# TODO try over fitting on single sentence LM

inputs = "I like cats and dogs"
outputs = ""
model_kwargs = {}
encoder_input_ids = tokenizer.encode(inputs)
encoder_input_ids = torch.tensor(encoder_input_ids).unsqueeze_(0)
decoder_input_ids = tokenizer_bert.encode(outputs)
decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze_(0)


out = model(encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)

logits = torch.nn.Softmax(-1)(out[0].squeeze(0))


pred = beam_search_decoder(logits.detach().numpy(), 1)


predicted_tokens = list(map(lambda p: tokenizer.decode(p), pred[0][0]))
#predicted_tokens = list(map(lambda p: tokenizer.decode(p), [torch.argmax(out[0], -1)[0].numpy()]))
print(predicted_tokens)




#generated_ids = model.decode(encoder_input_ids, length=6, temperature=1.3, k=9, p=0.9, repetition_penalty=1.4, **model_kwargs)
#generated_txt = tokenizer.decode(generated_ids)
#print(generated_txt)