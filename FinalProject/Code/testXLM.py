import transformers
import torch

model = transformers.XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')

#model = transformers.XLMModel.from_pretrained('xlm-mlm-en-2048')
tokenizer = transformers.XLMTokenizer.from_pretrained('xlm-mlm-en-2048')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
    0)  # Batch size 1
outputs = model(input_ids)
predicted_tokens = list(map(lambda p: tokenizer.decode(p), [torch.argmax(outputs[0], -1)[0].numpy()]))
print(predicted_tokens)

