import torch
from torch import nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Round-trip translations between English and German:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

#paraphrase = de2en.translate(en2de.translate('PyTorch Hub is an awesome interface!'))
#assert paraphrase == 'PyTorch Hub is a fantastic interface!'

# Compare the results with English-Russian round-trip translation:
#en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
#ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

#paraphrase = ru2en.translate(en2ru.translate('PyTorch Hub is an awesome interface!'))
#assert paraphrase == 'PyTorch is a great interface!'


class en2en(nn.Module):

    def __init__(self):
        super(en2en, self).__init__()

    def forward(self, x):

        x = en2de.translate(x)
        x = de2en.translate(x)

        return x

m = en2en()

optimizer = Adam(en2de.parameters(), lr=1e-4)
loss_function = CrossEntropyLoss()
m.train()
inputs = 'PyTorch Hub is an awesome interface!'
for i in range(100):

    outputs = m(inputs)
    loss = loss_function(inputs, outputs)
    loss.backward()
    optimizer.step()
    print(loss)



