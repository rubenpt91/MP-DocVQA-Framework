
import torch
import transformers

from transformers import get_scheduler


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, 'AdamW')
    optimizer = optimizer_class(model.model.parameters(), lr=float(config.lr))
    num_training_steps = config.train_epochs * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=config.warmup_iterations, num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def build_model(config):

    model_name = config.model_name.lower()
    available_models = ['bertqa', 'longformer', 'bigbird', 'layoutlmv2', 'layoutlmv3', 't5', 'longt5', 'vt5', 'hi-vt5']
    if model_name == 'bert' or model_name == 'bertqa':
        from models.BertQA import BertQA
        model = BertQA(config)

    elif model_name == 'longformer':
        from models.Longformer import Longformer
        model = Longformer(config)

    elif model_name == 'bigbird':
        from models.BigBird import BigBird
        model = BigBird(config)

    elif model_name == 'layoutlmv2':
        from models.LayoutLMv2 import LayoutLMv2
        model = LayoutLMv2(config)

    elif model_name == 'layoutlmv3':
        from models.LayoutLMv3 import LayoutLMv3
        model = LayoutLMv3(config)

    elif model_name == 't5':
        from models.T5 import T5
        model = T5(config)

    elif model_name == 'longt5':
        from models.LongT5 import LongT5
        model = LongT5(config)

    elif model_name == 'vt5':
        from models.VT5 import VT5
        model = VT5(config)

    elif model_name in ['hivt5', 'hi-vt5']:
        from models.HiVT5 import Proxy_HiVT5 as HiVT5
        model = HiVT5(config)

    else:
        raise ValueError("Value '{:s}' model not expected. Please choose one of: {:}".format(config.model_name, ', '.join(available_models)))

    if config.distributed:
        torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[config.local_rank])

    model.model.to(config.device)
    return model


def build_dataset(config, split):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config.model_name.lower() in ['layoutlmv2', 'layoutlmv3', 'vt5', 'hivt5', 'hi-vt5']:
        dataset_kwargs['get_raw_ocr_data'] = True
        dataset_kwargs['use_images'] = True

    if config.model_name.lower() in ['hivt5', 'hi-vt5']:
        dataset_kwargs['max_pages'] = getattr(config, 'max_pages', 1)
        dataset_kwargs['hierarchical_method'] = True

    # Build dataset
    if config.dataset_name == 'SP-DocVQA':
        from datasets.SP_DocVQA import SPDocVQA
        dataset = SPDocVQA(config.imdb_dir, config.images_dir, split, dataset_kwargs)

    elif config.dataset_name == 'InfographicsVQA':
        from datasets.InfographicsVQA import InfographicsVQA
        dataset = InfographicsVQA(config.imdb_dir, config.images_dir, split, dataset_kwargs)

    elif config.dataset_name == 'MP-DocVQA':
        from datasets.MP_DocVQA import MPDocVQA
        dataset = MPDocVQA(config.imdb_dir, config.images_dir, config.page_retrieval, split, dataset_kwargs)

    elif config.dataset_name == 'DUDE':
        from datasets.DUDE import DUDE
        dataset = DUDE(config.imdb_dir, config.images_dir, config.page_retrieval, split, dataset_kwargs)

    else:
        raise ValueError

    return dataset
