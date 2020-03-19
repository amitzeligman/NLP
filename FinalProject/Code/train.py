import torch
from FinalProject.Code.models import NLPModel
from FinalProject.Code.VideoDatasSet import VGGDataSet, collate_fn
from torch.utils.data import DataLoader
from torch.optim import Adam
from FinalProject.Code.models import FullModel, MultiGpuModel
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def train(data_loader, optimizer, model, loss_function, epochs, device, tokenizer):
    model.train()
    #if torch.cuda.device_count() > 1:
    #    model = MultiGpuModel(model)
    #    print('Using multi GPU mode with {} GPUS'.format(torch.cuda.device_count()))
    model.to(device)

    n_samples = len(data_loader)

    with tqdm(total=epochs) as epoch_progress:

        for epoch in range(epochs):
            epoch_loss = 0

            with tqdm(total=n_samples) as progress:

                for videos, sentences in data_loader:
                    optimizer.zero_grad()

                    decoder_input_ids = tokenizer.encode(sentences)
                    decoder_input_ids = torch.tensor(decoder_input_ids)
                    if data_loader.batch_size == 1:
                        decoder_input_ids.unsqueeze_(0)
                        videos.unsqueeze_(0)
                    l = decoder_input_ids.shape[-1]

                    videos = videos.to(device)
                    decoder_input_ids = decoder_input_ids.to(device)

                    outputs = model(videos, decoder_input_ids)
                    loss = loss_function(outputs[0].view(l, -1), decoder_input_ids.view(-1))
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    progress.update()
            print('total_loss: {}'.format(epoch_loss / n_samples))
            torch.save(model.state_dict(), '/media/cs-dl/HD_6TB/Data/Trained_models_nlp/{}.pt'.format(epoch))

        epoch_progress.update()


if __name__ == '__main__':

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/media/cs-dl/HD_6TB/Data/Amit/trainval'
    train_epochs = 40
    learning_rate = 1e-4

    model = FullModel()
    data_set = VGGDataSet(data_dir)
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    loss_function = CrossEntropyLoss()

    data_loader = DataLoader(dataset=data_set,
                             collate_fn=collate_fn,
                             batch_size=1,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True)

    train(data_loader, optimizer, model, loss_function, train_epochs, device, tokenizer)
