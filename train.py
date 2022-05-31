
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.SingleDocVQA import SingleDocVQA, singledocvqa_collate_fn
from models.Longformer import Longformer
from eval import evaluate
from metrics import Evaluator
from build_utils import build_model, build_optimizer, build_dataset
from utils import parse_args, load_config
from logger import Logger
from checkpoint import save_model


def train_epoch(data_loader, model, optimizer, lr_scheduler, evaluator, logger, **kwargs):
    model.model.train()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        questions = batch['questions']
        contexts = batch['contexts']
        gt_answers = batch['answers']
        start_indxs = batch['start_indxs']
        end_indxs = batch['end_indxs']

        outputs, pred_answers = model.forward(questions, contexts, gt_answers, start_indxs, end_indxs, return_pred_answer=True)

        # optimizer.zero_grad()
        outputs.loss.backward()
        optimizer.step()
        lr_scheduler.step()

        metric = evaluator.get_metrics(gt_answers, pred_answers)
        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])

        logger.logger.log({
            'Train/Batch loss': outputs.loss.item(),
            'Train/Batch Accuracy': batch_acc,
            'Train/Batch ANLS': batch_anls,
        }, step=logger.current_epoch * logger.len_dataset + batch_idx)

    # return total_accuracies, total_anls, answers


def train(model, **kwargs):

    epochs = kwargs['training_parameters']['train_epochs']
    # device = kwargs['device']
    batch_size = kwargs['training_parameters']['batch_size']

    evaluator = Evaluator(case_sensitive=False)
    logger = Logger(config=kwargs)
    logger.log_model_parameters(model)

    train_dataset = SingleDocVQA(kwargs['imdb_dir'], split='train')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=singledocvqa_collate_fn)
    val_dataset = SingleDocVQA(kwargs['imdb_dir'], split='val')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=singledocvqa_collate_fn)
    logger.len_dataset = len(train_data_loader)

    optimizer, lr_scheduler = build_optimizer(model, length_train_loader=len(train_data_loader), config=kwargs)

    for epoch_ix in range(epochs):
        logger.current_epoch = epoch_ix
        train_epoch(train_data_loader, model, optimizer, lr_scheduler, evaluator, logger, **kwargs)
        accuracy, anls, _ = evaluate(val_data_loader, model, evaluator, return_scores_by_sample=False, return_pred_answers=False, **kwargs)

        is_updated = evaluator.update_global_metrics(accuracy, anls, epoch_ix)
        logger.log_val_metrics(accuracy, anls, update_best=is_updated)
        save_model(model, epoch_ix, update_best=is_updated, **kwargs)


if __name__ == '__main__':

    args = parse_args()
    model = build_model(config)

    train(model, **config)

