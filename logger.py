import os, datetime
import wandb as wb


class Logger:

    def __init__(self, config):

        self.log_folder = config['save_dir']

        experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
        experiment_name = "{:s}__{:}".format(config['Model'], experiment_date)

        tags = [config['Model']]
        config = {'Batch size': config['batch_size']}
        self.logger = wb.init(project="DocCVQA Baselines", name=experiment_name, dir=self.log_folder, tags=tags, config=config)

        self.current_epoch = 0
        self.len_dataset = 0

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

