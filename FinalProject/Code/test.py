import torch
from tqdm import tqdm
import nltk
import numpy as np
from torchtext.data.metrics import bleu_score

MAX_SEQUENCE_LENGTH = 100


def test(data_loader, model, device, tokenizer, logger):

    model.eval()
    scores = []
    n_samples = len(data_loader)

    with tqdm(total=n_samples) as progress:
        sample = 0
        for videos, sentences in data_loader:
            if videos.shape[0] > MAX_SEQUENCE_LENGTH:
                continue
            decoder_input_ids = tokenizer.encode(sentences)
            decoder_input_ids = torch.tensor(decoder_input_ids)

            if data_loader.batch_size == 1:
                decoder_input_ids.unsqueeze_(0)
                videos.unsqueeze_(0)
            videos = videos.to(device)
            decoder_input_ids = decoder_input_ids.to(device)

            outputs = model(videos, decoder_input_ids)

            # Calculate BLEU score
            output_ids = torch.argmax(outputs[0], -1)
            output_sentences = tokenizer.decode(output_ids.view(-1))
            # Using PyTorch function (for more info: https://pytorch.org/text/data_metrics.html)
            scores.append(bleu_score(output_sentences, [sentences]))
            progress.update()
            sample += 1

        logger.info('Average BLEU score: {}'.format(np.array(scores).mean()))

    return scores





