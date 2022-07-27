import os, datetime, logging, socket
from build_utils import build_model, build_dataset
from utils import parse_args, load_config
from logger import Logger
from trainer_based.trainer_utils import ModelArguments, DataTrainingArguments, data_collator
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = load_config(args)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    os.environ['WANDB_PROJECT'] = 'DocCVQA_Baselines'
    os.environ['WANDB_TAGS'] = [config['model_name'], config['dataset_name'], socket.gethostname(), 'Trainer']

    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    experiment_name = "{:s}__{:}".format(config['model_name'], experiment_date)
    model_args, data_args, training_args = parser.parse_json_file(json_file='trainer_based/args.json')
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    train_dataset = build_dataset(config, 'train')
    valid_dataset = build_dataset(config, 'val')
    model = build_model(config)

    # my_logger = Logger(config=config)
    # my_logger.log_model_parameters(model)
    # training_args.run_name = my_logger.experiment_name

    # Initialize our Trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        # prediction_loss_only=True,
    )

    trainer.train()
