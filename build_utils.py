
import transformers
from transformers import get_scheduler
from torch.optim import AdamW

from models.Longformer import Longformer
from models.BertQA import BertQA


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=2e-4)
    num_training_steps = config['training_parameters']['train_epochs'] * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config['training_parameters']['warmup_iterations'], num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def build_model(config):

    if config['Model'].lower() == 'bert' or config['Model'].lower() == 'bertqa':
        model = BertQA(config)

    elif config['Model'].lower() == 'longformer':
        model = Longformer(config)

    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose BertQA or Longformer".format(config['Model']))

    model.model.to(config['device'])
    return model
