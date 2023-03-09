
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.SP_DocVQA import SPDocVQA, singlepage_docvqa_collate_fn
from models.Longformer import Longformer
from eval import evaluate
from metrics import Evaluator
from build_utils import build_model, build_optimizer, build_dataset
from utils import parse_args, load_config, seed_everything
from logger import Logger
from checkpoint import save_model


def train_epoch(data_loader, model, optimizer, lr_scheduler, evaluator, logger, **kwargs):
    model.model.train()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        gt_answers = batch['answers']
        outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)
        loss = outputs.loss + outputs.ret_loss if hasattr(outputs, 'ret_loss') else outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

        metric = evaluator.get_metrics(gt_answers, pred_answers)

        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])

        log_dict = {
            'Train/Batch loss': outputs.loss.item(),
            'Train/Batch Accuracy': batch_acc,
            'Train/Batch ANLS': batch_anls,
            'lr': optimizer.param_groups[0]['lr']
        }

        if hasattr(outputs, 'ret_loss'):
            log_dict['Train/Batch retrieval loss'] = outputs.ret_loss.item()

        if 'answer_page_idx' in batch and None not in batch['answer_page_idx']:
            ret_metric = evaluator.get_retrieval_metric(batch.get('answer_page_idx', None), pred_answer_page)
            batch_ret_prec = np.mean(ret_metric)
            log_dict['Train/Batch Ret. Prec.'] = batch_ret_prec

        logger.logger.log(log_dict, step=logger.current_epoch * logger.len_dataset + batch_idx)

    # return total_accuracies, total_anls, answers


# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     np.seed(worker_seed)


def train(model, **kwargs):

    epochs = kwargs['train_epochs']
    # device = kwargs['device']
    batch_size = kwargs['batch_size']
    seed_everything(kwargs['seed'])

    evaluator = Evaluator(case_sensitive=False)
    logger = Logger(config=kwargs)
    logger.log_model_parameters(model)

    train_dataset = build_dataset(config, 'train')
    val_dataset   = build_dataset(config, 'val')

    # g = torch.Generator()
    # g.manual_seed(kwargs['seed'])

    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=singlepage_docvqa_collate_fn)
    val_data_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=singlepage_docvqa_collate_fn)
    # train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=singledocvqa_collate_fn, worker_init_fn=seed_worker, generator=g)
    # val_data_loader   = DataLoader(val_dataset, batch_size=config['batch_size'],  shuffle=False, collate_fn=singledocvqa_collate_fn, worker_init_fn=seed_worker, generator=g)

    logger.len_dataset = len(train_data_loader)
    optimizer, lr_scheduler = build_optimizer(model, length_train_loader=len(train_data_loader), config=kwargs)

    if kwargs.get('eval_start', False):
        logger.current_epoch = -1
        accuracy, anls, ret_prec, _, _ = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=False, return_pred_answers=False, **kwargs)
        is_updated = evaluator.update_global_metrics(accuracy, anls, -1)
        logger.log_val_metrics(accuracy, anls, ret_prec, update_best=is_updated)

    for epoch_ix in range(epochs):
        logger.current_epoch = epoch_ix
        train_epoch(train_data_loader, model, optimizer, lr_scheduler, evaluator, logger, **kwargs)
        accuracy, anls, ret_prec, _, _ = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=False, return_pred_answers=False, **kwargs)

        is_updated = evaluator.update_global_metrics(accuracy, anls, epoch_ix)
        logger.log_val_metrics(accuracy, anls, ret_prec, update_best=is_updated)
        save_model(model, epoch_ix, update_best=is_updated, **kwargs)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)

    model = build_model(config)

    train(model, **config)

