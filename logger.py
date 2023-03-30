import os, socket, datetime, getpass
import wandb as wb


class Logger:

    def __init__(self, config):

        self.log_folder = config['save_dir']

        experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
        self.experiment_name = "{:s}__{:}".format(config['model_name'], experiment_date)

        machine_dict = {'cvc117': 'Local', 'cudahpc16': 'DAG', 'cudahpc25': 'DAG-A40'}
        machine = machine_dict.get(socket.gethostname(), socket.gethostname())

        dataset = config['dataset_name']
        page_retrieval = config.get('page_retrieval', '-').capitalize()
        visual_encoder = config.get('visual_module', {}).get('model', '-').upper()

        document_pages = config.get('max_pages', None)
        page_tokens = config.get('page_tokens', None)
        tags = [config['model_name'], dataset, machine]
        config = {'Model': config['model_name'], 'Weights': config['model_weights'], 'Dataset': dataset,
                  'Page retrieval': page_retrieval, 'Visual Encoder': visual_encoder,
                  'Batch size': config['batch_size'], 'Max. Seq. Length': config.get('max_sequence_length', '-'),
                  'lr': config['lr'], 'seed': config['seed']}

        if document_pages:
            config['Max Pages'] = document_pages

        if page_tokens:
            config['PAGE tokens'] = page_tokens

        self.logger = wb.init(project="MP-DocVQA", name=self.experiment_name, dir=self.log_folder, tags=tags, config=config)
        self._print_config(config)

        self.current_epoch = 0
        self.len_dataset = 0

    def _print_config(self, config):
        print("{:s}: {:s} \n{{".format(config['Model'], config['Weights']))
        for k, v in config.items():
            if k != 'Model' and k != 'Weights':
                print("\t{:}: {:}".format(k, v))
        print("}\n")

    def log_model_parameters(self, model):
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

        self.logger.config.update({
            'Model Params': int(total_params / 1e6),  # In millions
            'Model Trainable Params': int(trainable_params / 1e6)  # In millions
        })

        print("Model parameters: {:d} - Trainable: {:d} ({:2.2f}%)".format(
            total_params, trainable_params, trainable_params / total_params * 100))

    def log_val_metrics(self, accuracy, anls, ret_prec, update_best=False):

        str_msg = "Epoch {:d}: Accuracy {:2.2f}     ANLS {:2.4f}    Retrieval precision: {:2.2f}%".format(self.current_epoch, accuracy*100, anls, ret_prec*100)
        self.logger.log({
            'Val/Epoch Accuracy': accuracy,
            'Val/Epoch ANLS': anls,
            'Val/Epoch Ret. Prec': ret_prec,
        }, step=self.current_epoch*self.len_dataset + self.len_dataset)

        if update_best:
            str_msg += "\tBest Accuracy!"
            self.logger.config.update({
                "Best Accuracy": accuracy,
                "Best epoch": self.current_epoch
            }, allow_val_change=True)

        print(str_msg)

