import torch
import transformers

from transformers import get_scheduler


def build_optimizer(model, length_train_loader, config):
    optimizer_class = getattr(transformers, "AdamW")
    optimizer = optimizer_class(model.model.parameters(), lr=float(config["lr"]))
    num_training_steps = config["train_epochs"] * length_train_loader
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_iterations"],
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler


def build_model(config):

    available_models = [
        "BertQA",
        "LayoutLMv2",
        "LayoutLMv3",
        "Longformer",
        "BigBird",
        "T5",
        "LT5",
        "Hi-LT5",
        "Hi-VT5",
    ]
    if config["model_name"].lower() == "bert" or config["model_name"].lower() == "bertqa":
        from models.BertQA import BertQA

        model = BertQA(config)

    elif config["model_name"].lower() == "longformer":
        from models.Longformer import Longformer

        # from models.Longformer_SQuAD import Longformer
        model = Longformer(config)

    elif config["model_name"].lower() == "bigbird":
        from models.BigBird import BigBird

        model = BigBird(config)

    elif config["model_name"].lower() == "longt5":
        from models.LongT5 import LongT5

        model = LongT5(config)

    elif config["model_name"].lower() == "layoutlmv2":
        from models.LayoutLMv2 import LayoutLMv2

        model = LayoutLMv2(config)

    elif config["model_name"].lower() == "layoutlmv3":
        from models.LayoutLMv3 import LayoutLMv3

        model = LayoutLMv3(config)

    elif config["model_name"].lower() == "t5":
        from models.T5 import T5

        model = T5(config)

    elif config["model_name"].lower() == "lt5":
        from models.LT5 import Proxy_LT5 as LT5

        model = LT5(config)

    elif config["model_name"].lower() in ["hilt5", "hi-lt5"]:
        from models.HiLT5 import Proxy_HiLT5 as HiLT5

        model = HiLT5(config)

    elif config["model_name"].lower() in ["hivt5", "hi-vt5"]:
        from models.HiVT5 import Proxy_HiVT5 as HiVT5

        model = HiVT5(config)

    elif config["model_name"].lower() in ["hilayoutlmv3", "hi-layoutlmv3"]:
        from models.HiLayoutLMv3 import Proxy_HiLayoutLMv3 as HiLayoutLMv3

        model = HiLayoutLMv3(config)

    else:
        raise ValueError(
            "Value '{:s}' for model selection not expected. Please choose one of {:}".format(
                config["model_name"], ", ".join(available_models)
            )
        )

    if config["device"] == "cuda" and config["data_parallel"] and torch.cuda.device_count() > 1:
        model.parallelize()

    model.model.to(config["device"])
    return model


def build_dataset(config, split):

    # Specify special params for data processing depending on the model used.
    dataset_kwargs = {}

    if config["model_name"].lower() in [
        "layoutlmv2",
        "layoutlmv3",
        "hilayoutlmv3",
        "hi-layoutlmv3",
        "lt5",
        "hilt5",
        "hi-lt5",
        "hivt5",
        "hi-vt5",
    ]:
        dataset_kwargs["get_raw_ocr_data"] = True

    if config["model_name"].lower() in [
        "layoutlmv2",
        "layoutlmv3",
        "hilayoutlmv3",
        "hi-layoutlmv3",
        "hivt5",
        "hi-vt5",
    ]:
        dataset_kwargs["use_images"] = True

    if config["model_name"].lower() in [
        "hilayoutlmv3",
        "hi-layoutlmv3",
        "hilt5",
        "hi-lt5",
        "hivt5",
        "hi-vt5",
    ]:
        dataset_kwargs["max_pages"] = config.get("max_pages", 1)
        dataset_kwargs["hierarchical_method"] = True

    # Build dataset
    if config["dataset_name"] == "SQuAD":
        from datasets.SQuAD import SQuAD

        dataset = SQuAD(config["imdb_dir"], split)

    elif config["dataset_name"] == "SP-DocVQA":
        from datasets.SP_DocVQA import SPDocVQA

        dataset = SPDocVQA(config["imdb_dir"], config["images_dir"], split, dataset_kwargs)

    elif config["dataset_name"] == "MP-DocVQA":
        from datasets.MP_DocVQA import MPDocVQA

        dataset = MPDocVQA(
            config["imdb_dir"],
            config["images_dir"],
            config["page_retrieval"],
            split,
            dataset_kwargs,
        )

    elif config["dataset_name"] == "DUDE":
        from datasets.DUDE import DUDE

        if split == "test":
            split = "val"
        DUDE_kwargs = {
            k: config[k]
            for k in [
                "none_strategy",
                "list_strategy",
                "atype_learning",
                "qtype_learning",
            ]
            if k in config
        }
        dataset = DUDE(
            config["imdb_dir"],
            config["images_dir"],
            config["page_retrieval"],
            split,
            dataset_kwargs,
            **DUDE_kwargs
        )

    else:
        raise ValueError

    return dataset
