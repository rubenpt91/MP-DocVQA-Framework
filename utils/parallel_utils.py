import torch


def get_distributed_sampler(dataset, config):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=config.world_size, rank=config.global_rank
    )
    return sampler
