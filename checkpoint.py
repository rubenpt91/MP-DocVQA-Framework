import os
from utils import save_yaml


def save_model(model, epoch, update_best, config):
    save_dir = os.path.join(config.save_dir, 'checkpoints', "{:s}_{:s}_{:s}".format(config.model_name.lower(), getattr(config, 'page_retrieval', '').lower(), config.dataset_name.lower()))
    model.model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.processor if hasattr(model, 'processor') else None
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

    if hasattr(model.model, 'visual_embeddings'):
        model.model.visual_embeddings.feature_extractor.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

    save_yaml(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch), "experiment_config.yml"), config)

    if update_best:
        model.model.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        save_yaml(os.path.join(save_dir, "best.ckpt", "experiment_config.yml"), config)


def load_model(base_model, ckpt_name, config):
    load_dir = config.save_dir
    base_model.model.from_pretrained(os.path.join(load_dir, ckpt_name))
