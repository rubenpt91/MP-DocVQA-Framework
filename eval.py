import os, time, datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset_utils import docvqa_collate_fn
from logger import Logger
from metrics import Evaluator
from utils.commons import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset
from utils.parallel_utils import get_distributed_sampler


def evaluate(data_loader, model, evaluator, config):

    dataset_has_answers = data_loader.dataset.has_answers
    return_scores_by_sample = getattr(config, 'return_scores_by_sample', False)
    return_answers = getattr(config, 'return_answers', False)

    if return_scores_by_sample:
        scores_by_samples = {}
        total_accuracies = []
        total_anls = []
        total_ret_prec = []

    else:
        total_accuracies = 0
        total_anls = 0
        total_ret_prec = 0

    all_pred_answers = []
    model.model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch['question_id'])
        with torch.no_grad():
            outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)
            # print(pred_answers)

        if dataset_has_answers:
            metric = evaluator.get_metrics(batch['answers'], pred_answers, batch.get('answer_type', None))

            if 'answer_page_idx' in batch and pred_answer_page is not None:
                ret_metric = evaluator.get_retrieval_metric(batch['answer_page_idx'], pred_answer_page)
            else:
                ret_metric = [0 for _ in range(bs)]

            if return_scores_by_sample:
                for batch_idx in range(bs):
                    scores_by_samples[batch['question_id'][batch_idx]] = {
                        'accuracy': metric['accuracy'][batch_idx],
                        'anls': metric['anls'][batch_idx],
                        'ret_prec': ret_metric[batch_idx],
                        'pred_answer': pred_answers[batch_idx],
                        'pred_answer_conf': answer_conf[batch_idx],
                        'pred_answer_page': pred_answer_page[batch_idx] if pred_answer_page is not None else None
                    }

            if return_scores_by_sample:
                total_accuracies.extend(metric['accuracy'])
                total_anls.extend(metric['anls'])
                total_ret_prec.extend(ret_metric)

            else:
                total_accuracies += sum(metric['accuracy'])
                total_anls += sum(metric['anls'])
                total_ret_prec += sum(ret_metric)

        if return_answers:
            all_pred_answers.extend(pred_answers)

    if not return_scores_by_sample and dataset_has_answers:
        total_accuracies = total_accuracies/len(data_loader.dataset)
        total_anls = total_anls/len(data_loader.dataset)
        total_ret_prec = total_ret_prec/len(data_loader.dataset)
        scores_by_samples = []

    return total_accuracies, total_anls, total_ret_prec, all_pred_answers, scores_by_samples


def run_evaluation(local_rank, config):
# if __name__ == '__main__':

    # args = parse_args()
    # config = load_config(args)
    # config.return_answers = True
    # config.return_scores_by_sample = True

    if config.distributed:
        config.global_rank = config.node_id * config.num_gpus + local_rank

        torch.distributed.init_process_group(
            backend='nccl',
            world_size=config.world_size,
            rank=config.global_rank
        )

        config.local_rank = local_rank
        config.device = "cuda:{:d}".format(local_rank)

    start_time = time.time()

    dataset = build_dataset(config, 'test')
    if config.distributed:
        dist_sampler = get_distributed_sampler(dataset, config)
        val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=docvqa_collate_fn, pin_memory=True, sampler=dist_sampler)

    else:
        val_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=docvqa_collate_fn)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    evaluator = Evaluator(case_sensitive=False)
    accuracy_list, anls_list, answer_page_pred_acc_list, pred_answers, scores_by_samples = evaluate(val_data_loader, model, evaluator, config)
    accuracy, anls, answ_page_pred_acc = np.mean(accuracy_list), np.mean(anls_list), np.mean(answer_page_pred_acc_list)

    inf_time = time_stamp_to_hhmmss(time.time() - start_time, string=True)
    logger.log_val_metrics(accuracy, anls, answ_page_pred_acc, update_best=False)

    save_data = {
        "Model": config.model_name,
        "Model_weights": config.model_weights,
        "Dataset": config.dataset_name,
        "Page retrieval": getattr(config, 'page_retrieval', '-').capitalize(),
        "Inference time": inf_time,
        "Mean accuracy": accuracy,
        "Mean ANLS": anls,
        "Mean Retrieval precision": answ_page_pred_acc,
        "Scores by samples": scores_by_samples,
    }

    experiment_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(config.save_dir, 'results', "{:}_{:}_{:}__{:}.json".format(config.model_name, config.dataset_name, getattr(config, 'page_retrieval', '').lower(), experiment_date).replace('_none', ''))
    save_json(results_file, save_data)

    print("Results correctly saved in: {:s}".format(results_file))


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    config.return_answers = True
    config.return_scores_by_sample = True

    if config.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9957'
        torch.multiprocessing.spawn(run_evaluation, nprocs=args.num_gpus, args=(config,))

    else:
        run_evaluation(local_rank=0, config=config)

    print("\n\n\nPRINT AFTER EVALUATION\n\n\n")
