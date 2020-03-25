import torch
from tqdm import tqdm

MAX_SEQUENCE_LENGTH = 20


def train(data_loader, optimizer, model, epochs, device, tokenizer, logger, tensor_board_writer, weights_save_path):

    model.train()
    n_samples = len(data_loader)

    with tqdm(total=epochs) as epoch_progress:

        for epoch in range(epochs):

            with tqdm(total=n_samples) as progress:
                sample = 0
                tot_loss = 0
                for iteration, (videos, sentences) in enumerate(data_loader):
                    if videos.shape[0] > MAX_SEQUENCE_LENGTH:
                        continue
                    optimizer.zero_grad()

                    decoder_input_ids = tokenizer.encode(sentences)
                    decoder_input_ids = torch.tensor(decoder_input_ids)
                    if data_loader.batch_size == 1:
                        decoder_input_ids.unsqueeze_(0)
                        videos.unsqueeze_(0)
                    #l = decoder_input_ids.shape[-1]

                    videos = videos.to(device)
                    decoder_input_ids = decoder_input_ids.to(device)

                    outputs = model(videos, decoder_input_ids)
                    loss = outputs[0]
                    tot_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    progress.update()
                    sample += 1

                    if not sample % 100:
                        logger.info('total_loss for {} samples: {}'.format(sample, tot_loss / sample))
                        tensor_board_writer.add_scalar('Loss/train', tot_loss / sample, epoch * len(data_loader) + iteration)
                        sample = 0
                        tot_loss = 0
                        # Writing to T-Board

                        if len(videos.shape) == 4:
                            videos = torch.unsqueeze(videos, dim=0)
                        tensor_board_writer.add_video('Video (for check cropping)', videos, epoch * len(data_loader) + iteration)

            torch.save(model.state_dict(), weights_save_path + '_{}.pt'.format(epoch))

        epoch_progress.update()

    return model





