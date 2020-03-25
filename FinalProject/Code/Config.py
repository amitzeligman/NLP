class Parameters(object):

    def __init__(self):
        self.params = dict()

        """ Paths """
        self.params['train_data_dir'] = '/media/cs-dl/HD_6TB/Data/Amit/trainval'
        self.params['test_data_dir'] = '/media/cs-dl/HD_6TB/Data/Amit/test'
        self.params['log_path'] = '/home/cs-dl/tmp/logs/nlp.log'
        self.params['weights_save_path'] = '/media/cs-dl/HD_6TB/Data/Trained_models_nlp/test_25_3'
        self.params['tensor_board_log_dir'] = './runs'
        self.params['pre_trained_weights'] = None
        self.params['test_weights'] = None

        """ Hyper Parameters """
        self.params['learning_rate'] = 1e-4
        self.params['epochs'] = 10

        """ Model Parameters """

        self.params['model_params'] = {'lstm_hidden_size': 768, 'lstm_layers': 8, 'bidirectional_lstm': True,
                                       'lstm_drop_out': 0.25,
                                       }

