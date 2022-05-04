import os


def save_model(model, epoch, update_best=False, **kwargs):
    save_dir = os.path.join(kwargs['save_dir'], 'checkpoints')
    model.model.save_pretrained(os.path.join(save_dir, "{:s}_{:d}.ckpt".format('LongFormer', epoch)))

    if update_best:
        model.model.save_pretrained(os.path.join(save_dir, "{:s}_best.ckpt".format('LongFormer')))


def load_model(base_model, ckpt_name, **kwargs):
    load_dir = kwargs['save_dir']
    base_model.model.from_pretrained(os.path.join(load_dir, ckpt_name))
