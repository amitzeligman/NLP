import argparse
import transformers
import logging
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FinalProject.Code.train import train
from FinalProject.Code.test import test
from FinalProject.Code.VideoDatasSet import VGGDataSet, collate_fn
from FinalProject.Code.models import FullModel, MultiGpuModel
from FinalProject.Code.Utils import *
import datetime
import os


if __name__ == '__main__':

    # Device config
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", required=False, help='Whether to apply Training or not')
    parser.add_argument('--test', action="store_true", required=False, help='Whether to apply Testing or not')
    args = parser.parse_args()

    # Parameters
    train_data_dir = '/media/cs-dl/HD_6TB/Data/Amit/trainval'
    test_data_dir = '/media/cs-dl/HD_6TB/Data/Amit/test'

    log_path = '/home/cs-dl/tmp/logs/nlp.log'
    tboard_log_dir = './runs'
    pre_trained_weights = '/media/cs-dl/HD_6TB/Data/Trained_models_nlp/0.pt'

    train_epochs = 40
    learning_rate = 1e-4
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Instantiation of T-Board summary writer
    tboard_curr_dir = os.path.join(tboard_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not(tboard_curr_dir):
        os.makedirs(tboard_curr_dir)
    TBoard_writer = SummaryWriter(tboard_curr_dir)

    # Logger setting
    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    # Model configuration
    model = FullModel()
    if pre_trained_weights is not None:
        state_dict = fix_state_dict(torch.load(pre_trained_weights))
        model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        model = MultiGpuModel(model)
        logger.info('Using multi GPU mode with {} GPUS'.format(torch.cuda.device_count()))
    model.to(device)

    # Training
    if args.train:
        data_set = VGGDataSet(train_data_dir)
        loss_function = CrossEntropyLoss()
        optimizer = Adam(params=model.parameters(), lr=learning_rate)

        data_loader = DataLoader(dataset=data_set,
                                 collate_fn=collate_fn,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)

        trained_model = train(data_loader, optimizer, model, loss_function, train_epochs, device, tokenizer, logger, TBoard_writer)

    # Testing
    if args.test:
        data_set = VGGDataSet(test_data_dir)
        data_loader = DataLoader(dataset=data_set,
                                 collate_fn=collate_fn,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

        scores = test(data_loader, model, device, tokenizer, logger)



