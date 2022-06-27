
import importlib
import transformers

from transformers import get_scheduler

from models.BertQA_SQuAD import BertQA
from models.Longformer import Longformer
# from models.Longformer_SQuAD import Longformer

from models.BigBird import BigBird


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=float(config['lr']))
    num_training_steps = config['train_epochs'] * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config['warmup_iterations'], num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def build_model(config):

    available_models = ['BertQA', 'Longformer', 'BigBird']
    if config['model_name'].lower() == 'bert' or config['model_name'].lower() == 'bertqa':
        model = BertQA(config)

    elif config['model_name'].lower() == 'longformer':
        model = Longformer(config)

    elif config['model_name'].lower() == 'bigbird':
        model = BigBird(config)

    else:
        raise ValueError("Value '{:s}' for model selection not expected. Please choose one of {:}".format(config['model_name'], ', '.join(available_models)))

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

    if config['dataset_name'] == 'SQuAD':
        from datasets.SQuAD import SQuAD
        dataset = SQuAD(config['imdb_dir'], split)

    elif config['dataset_name'] == 'SingleDocVQA':
        # from datasets.SingleDocVQA import SingleDocVQA
        # dataset = SingleDocVQA(config['imdb_dir'], split)
        from datasets.SingleDocVQA_trainer import SingleDocVQA
        dataset = SingleDocVQA(config['imdb_dir'], split, config['model_weights'])

    elif config['dataset_name'] == 'MP-DocVQA':
        from datasets.MP_DocVQA import MPDocVQA
        dataset = MPDocVQA(config['imdb_dir'], config['page_retrieval'], split)

    else:
        raise ValueError

    return dataset
