import argparse
import transformers
import logging
import torch
from datetime import datetime
import os
from torch.optim import Adam
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from FinalProject.Code.Config import Parameters
from FinalProject.Code.train import train
from FinalProject.Code.test import test
from FinalProject.Code.VideoDatasSet import VGGDataSet, collate_fn
from FinalProject.Code.models import FullModel, MultiGpuModel
from FinalProject.Code.Utils import *


if __name__ == '__main__':

    # Device config
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", required=False, help='Whether to apply Training or not')
    parser.add_argument('--test', action="store_true", required=False, help='Whether to apply Testing or not')
    args = parser.parse_args()

    # Parameters
    params = Parameters().params
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Instantiation of T-Board summary writer
    tensor_board_curr_dir = os.path.join(params['tensor_board_log_dir'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tensor_board_curr_dir, exist_ok=True)
    TBoard_writer = SummaryWriter(tensor_board_curr_dir)

    # Logger setting
    logging.basicConfig(filename=params['log_path'], filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    # Model configuration
    model = FullModel(params['model_params'])
    if params['pre_trained_weights'] is not None:
        state_dict = fix_state_dict(torch.load(params['pre_trained_weights']))
        model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        model = MultiGpuModel(model)
        logger.info('Using multi GPU mode with {} GPUS'.format(torch.cuda.device_count()))
    model.to(device)

    # Training
    if args.train:
        data_set = VGGDataSet(params['train_data_dir'])
        optimizer = Adam(params=model.parameters(), lr=params['learning_rate'])

        data_loader = DataLoader(dataset=data_set,
                                 collate_fn=collate_fn,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True)

        trained_model = train(data_loader, optimizer, model,
                              params['epochs'], device, tokenizer, logger, TBoard_writer, params['weights_save_path'])

    # Testing
    if args.test:
        data_set = VGGDataSet(params['test_data_dir'])
        data_loader = DataLoader(dataset=data_set,
                                 collate_fn=collate_fn,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

        scores = test(data_loader, model, device, tokenizer, logger)

        # TODO - algorithmic:
        #  1. Beam search in prediction.
        #  2. Another train branch of LM sentences.
        #  3. Design video embeddings - Last hidden state or outputs of LSTM?, frames per id rate?.
        #

        # TODO - data:
        #  1. Add sub sequence option in data loader.



