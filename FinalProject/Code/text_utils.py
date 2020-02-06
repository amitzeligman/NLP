import torch
from pytorch_pretrained_bert import BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def sentences_to_bert_inputs(sentences):

    tokenized_text = tokenizer.tokenize(sentences)
    tokenized_labels = tokenized_text.copy()
    masked_index = np.random.choice([idx for idx, _ in enumerate(tokenized_text)
                                     if not tokenized_text[idx] in ['[CLS]', '[SEP]']])
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokenized_labels = tokenizer.convert_tokens_to_ids(tokenized_labels)
    segments_ids = []
    i = 0
    for token in tokenized_text:
        segments_ids.append(i)
        if token == '[SEP]':
            i += 1
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokenized_labels = torch.tensor([tokenized_labels])

    return tokens_tensor, segments_tensors, tokenized_labels





