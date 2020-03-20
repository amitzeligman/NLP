import argparse
import transformers
import logging
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from FinalProject.Code.train import train
from FinalProject.Code.VideoDatasSet import VGGDataSet, collate_fn
from FinalProject.Code.models import FullModel
from collections import OrderedDict



def fix_state_dict(state_dict):

    assert isinstance(state_dict, OrderedDict)

    if list(state_dict.keys())[0].split('.')[0] == 'module':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    return new_state_dict


if __name__ == '__main__':

    # Device config
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", required=False, help='Whether to apply Training or not')
    parser.add_argument('--test', action="store_true", required=False, help='Whether to apply Testing or not')
    args = parser.parse_args()

    # Parameters
    data_dir = '/media/cs-dl/HD_6TB/Data/Amit/trainval'
    log_path = '/home/cs-dl/tmp/logs/nlp.log'
    pre_trained_weights = '/media/cs-dl/HD_6TB/Data/Trained_models_nlp/0.pt'

    train_epochs = 40
    learning_rate = 1e-4
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Logger setting
    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    if args.train:
        model = FullModel()
        if pre_trained_weights is not None:
            state_dict = fix_state_dict(torch.load(pre_trained_weights))
            model.load_state_dict(state_dict)
        data_set = VGGDataSet(data_dir)
        loss_function = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)

        data_loader = DataLoader(dataset=data_set,
                                 collate_fn=collate_fn,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)

        trained_model = train(data_loader, optimizer, model, loss_function, train_epochs, device, tokenizer, logger)

    if args.test:
        raise NotImplementedError

