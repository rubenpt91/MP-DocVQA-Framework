
import importlib
import transformers

from transformers import get_scheduler

from models.Longformer_SQuAD import Longformer
from models.BertQA_SQuAD import BertQA


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=float(config['training_parameters']['lr']))
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

"""
def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
"""

def build_dataset(config, split):

    dataset_config = config['dataset_parameters']
    if dataset_config['dataset_name'] == 'SQuAD':
        from datasets.SQuAD import SQuAD
        dataset = SQuAD(dataset_config['imdb_dir'], split)

    elif dataset_config['dataset_name'] == 'SingleDocVQA':
        from datasets.SingleDocVQA import SingleDocVQA
        dataset = SingleDocVQA(dataset_config['imdb_dir'], split)

    else:
        raise ValueError

    return dataset
