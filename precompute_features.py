import os
import numpy as np
import json
from build_utils import build_dataset
from utils import parse_args, load_config
from models.HiVT5 import VisualEmbeddings
from transformers import PretrainedConfig
from tqdm import tqdm
from PIL import Image
import re


def precompute_visual_features(config):
    visual_config = PretrainedConfig()
    visual_config.update(
        {
            "return_visual_embedding": True,
            "visual_module_config": {
                "model": "dit",
                "model_weights": "microsoft/dit-base-finetuned-rvlcdip",
            },
        }
    )
    visual_embedder = VisualEmbeddings(
        visual_config, finetune=False
    )  # BS; 14x14+CLS (197); 768 (hidden size)

    config["precomputed_visual_feats"] = False

    # split_to_npz, directory mode
    CHUNKSIZE = 1000
    for split in ["val", "train"]:
        if split == "val":
            continue
        page_visual_featdict = {}
        count = 0

        for file in tqdm(os.listdir(os.path.join(config["images_dir"], split))):
            page_visual_featdict[f"{split}/{file}"] = count
            if count < 16000:  # skip
                count += 1
                continue
            print(count)
            image = Image.open(os.path.join(config["images_dir"], split, file)).convert("RGB")
            if count == 0 or count % CHUNKSIZE == 0:  # gives the reset
                collect_visual_features = visual_embedder([image], None).cpu().numpy()[0]
                count += 1
                continue

            page_visual_features = visual_embedder([image], None).cpu().numpy()[0]
            collect_visual_features = np.concatenate(
                (collect_visual_features, page_visual_features)
            )
            count += 1
            if count % CHUNKSIZE == 0:
                out = os.path.join(
                    config["images_dir"], f"{split}-visfeats_{int(count/CHUNKSIZE)}.npz"
                )
                np.savez_compressed(out, collect_visual_features)

        # deal with padding
        collect_visual_features = np.concatenate(
            (
                collect_visual_features,
                visual_embedder(np.ones((2, 2, 3), np.uint8) * 255, None).cpu().numpy()[0],
            )
        )
        out = os.path.join(config["images_dir"], f"{split}-visfeats_{int(count/CHUNKSIZE)+1}.npz")
        np.savez_compressed(out, collect_visual_features)
        from pdb import set_trace

        set_trace()
        # out = os.path.join(config["images_dir"], f"{split}-visfeats.npz")
        # np.savez_compressed(out, collect_visual_features)
        page_visual_featdict[f"PAD"] = count
        out = os.path.join(config["images_dir"], f"{split}-visfeats.json")
        with open(out, "w") as f:
            pass
            # json.dump(page_visual_featdict, f)

    return

    def ex_to_visual_features(ex):
        # --> specific features for padding saved in json
        image_names = [
            ex["image_names"][i].replace(config["images_dir"] + "/", "")
            for i in range(len(ex["image_names"]))
        ]
        document_visual_features = (
            visual_embedder([ex["images"][i] for i in range(len(ex["image_names"]))], None)
            .cpu()
            .numpy()
        )
        return image_names, document_visual_features

    train_dataset = build_dataset(config, "train")
    for i, ex in tqdm(enumerate(train_dataset)):
        if i == 0:
            collect_names, collect_visual_features = ex_to_visual_features(ex)
            continue
        image_names, document_visual_features = ex_to_visual_features(ex)
        collect_names.extend(image_names)
        collect_visual_features = np.concatenate(
            (collect_visual_features, document_visual_features)
        )
    val_dataset = build_dataset(config, "val")
    for ex in val_dataset:
        image_names, document_visual_features = ex_to_visual_features(ex)
        collect_names.extend(image_names)
        collect_visual_features = np.concatenate(
            (collect_visual_features, document_visual_features)
        )
    print(collect_visual_features.shape)
    print(len(collect_names))
    out = os.path.join(config["images_dir"], "visfeats.npz")
    np.savez_compressed(out, collect_visual_features)
    out = os.path.join(config["images_dir"], "visfeat_names.json")
    names_and_pad = {
        "image_names": collect_names,
        "PAD": visual_embedder(np.ones((2, 2, 3), np.uint8) * 255, None).cpu().numpy()[0].tolist(),
    }
    with open(out, "w") as f:
        json.dump(names_and_pad, f)


def merge_precomputed(config):
    a = np.zeros((16918, 197, 768), dtype=float) #precomputed size
    #catted = None
    count = 0
    for i, file in enumerate(os.listdir(config["images_dir"])):
        if not re.match("train-visfeats_\d+.npz", file):
            continue
        saved = np.load(os.path.join(config["images_dir"], file))["arr_0"].reshape((-1, 197, 768))
        until = int(saved.shape[0])
        print(file, saved.shape, count, count+until)
        a[count:count+until] = saved
        count += until
        # if i == 0:
        #     catted = saved
        # else:
        #     catted = np.concatenate((catted, saved))
    output_path = os.path.join(config["images_dir"], "train-visfeats.npz")
    np.savez_compressed(output_path, a)


def test_precomputed(config):
    config["precomputed_visual_feats"] = True
    val_dataset = build_dataset(config, "val")
    ex = val_dataset[0]
    print(val_dataset.precomputed_visual_feats)
    print(np.array(ex["images"]).shape)
    from pdb import set_trace

    set_trace()


def image_sizes(config):
    config["precomputed_visual_feats"] = False
    train_dataset = build_dataset(config, "train")
    image_sizes = []
    unique_names = []
    for i in tqdm(range(len(train_dataset))):
        ex = train_dataset[i]
        for j in range(len(ex["image_names"])):
            name = ex["image_names"][j]
            if name in unique_names:
                break
            unique_names.append(name)
            im = ex["images"][j]
            image_sizes.append(im.height * im.width)
    import pandas as pd

    df = pd.DataFrame(image_sizes, columns=["image_size"])
    print(df["image_size"].value_counts())
    from scipy.stats import describe

    print(describe(df["image_size"].tolist()))
    from pdb import set_trace

    set_trace()
    
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    merge_precomputed(config)
    # precompute_visual_features(config)
    # image_sizes(config)
    # test_precomputed(config)
