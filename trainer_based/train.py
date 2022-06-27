import logging
from build_utils import build_model, build_dataset
from utils import parse_args, load_config
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

    model_args, data_args, training_args = parser.parse_json_file(json_file='trainer_based/args.json')
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    train_dataset = build_dataset(config, 'train')
    valid_dataset = build_dataset(config, 'val')
    model = build_model(config)

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
