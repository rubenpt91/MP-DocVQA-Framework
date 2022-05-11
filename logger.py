import os, socket, datetime, getpass
import wandb as wb


class Logger:

    def __init__(self, config):

        self.log_folder = config['save_dir']

        experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
        experiment_name = "{:s}__{:}".format(config['Model'], experiment_date)

        machine_dict = {'cvc117': 'Local', 'cudahpc16': 'DAG', 'cudahpc25': 'DAG-A40'}
        machine = machine_dict.get(socket.gethostname(), socket.gethostname())

        tags = [config['Model'], machine]
        config = {'Batch size': config['training_parameters']['batch_size'], 'Model': config['Model'], 'Weights': config['Model_weights']}
        self.logger = wb.init(project="DocCVQA Baselines", name=experiment_name, dir=self.log_folder, tags=tags, config=config)

        self.current_epoch = 0
        self.len_dataset = 0

    def log_model_parameters(self, model_parameters):
        total_params = sum(p.numel() for p in model_parameters)
        trainable_params = sum(p.numel() for p in model_parameters if p.requires_grad)

        self.logger.log({
            'Model Params': int(total_params / 1e6),  # In millions
            'Model Trainable Params': int(trainable_params / 1e6)  # In millions
        })

        print("Model parameters: {:d} - Trainable: {:d} ({:2.2f}%)".format(
            total_params, trainable_params, trainable_params / total_params * 100))

    def log_val_metrics(self, accuracy, anls, update_best=False):

        str_msg = "Epoch {:d}: Accuracy {:2.4f}  ANLS {:2.4f}".format(self.current_epoch, accuracy, anls)
        self.logger.log({
            'Val/Epoch Accuracy': accuracy,
            'Val/Epoch ANLS': anls,
        }, step=self.current_epoch*self.len_dataset + self.len_dataset)

        if update_best:
            str_msg += "\tBest Accuracy!"
            self.logger.config.update({
                "Best Accuracy": accuracy,
                "Best epoch": self.current_epoch
            })

        print(str_msg)

