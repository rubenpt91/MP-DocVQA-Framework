import os
import numpy as np
import json
from build_utils import build_dataset
from utils import parse_args, load_config
from models.HiVT5 import VisualEmbeddings
from transformers import PretrainedConfig
from tqdm import tqdm

def precompute_visual_features(config):
    visual_config = PretrainedConfig()
    visual_config.update({'return_visual_embedding': True,'visual_module_config': {'model': 'dit', 'model_weights': 'microsoft/dit-base-finetuned-rvlcdip'}})
    visual_embedder = VisualEmbeddings(visual_config, finetune=False) # BS; 14x14+CLS (197); 768 (hidden size)
    
    def ex_to_visual_features(ex):
        # --> specific features for padding saved in json 
        image_names = [ex['image_names'][i].replace(config['images_dir']+"/",'') for i in range(len(ex['image_names']))]
        document_visual_features = visual_embedder([ex['images'][i] for i in range(len(ex['image_names']))], None).cpu().numpy()
        return image_names, document_visual_features

    train_dataset = build_dataset(config, 'train')
    for i, ex in tqdm(enumerate(train_dataset)):
        if i == 0:
            collect_names, collect_visual_features = ex_to_visual_features(ex)
            continue
        image_names, document_visual_features = ex_to_visual_features(ex)
        collect_names.extend(image_names)
        collect_visual_features = np.concatenate((collect_visual_features, document_visual_features))
    val_dataset = build_dataset(config, 'val')
    for ex in val_dataset:
        image_names, document_visual_features = ex_to_visual_features(ex)
        collect_names.extend(image_names)
        collect_visual_features = np.concatenate((collect_visual_features, document_visual_features))
    print(collect_visual_features.shape)
    print(len(collect_names))
    out = os.path.join(config['images_dir'], "visfeats.npz")
    np.savez_compressed(out, collect_visual_features)
    out = os.path.join(config['images_dir'], "visfeat_names.json")
    names_and_pad = {"image_names": collect_names, "PAD": visual_embedder(np.ones((2, 2, 3), np.uint8) * 255, None).cpu().numpy()[0].tolist()}
    with open(out, 'w') as f:
        json.dump(names_and_pad, f)

def test_precomputed(config):
    config["precomputed_visual_feats"] = True
    train_dataset = build_dataset(config, 'train')
    ex = train_dataset[0]
    print(train_dataset.precomputed_visual_feats)
    print(np.array(ex['images']).shape)
    from pdb import set_trace; set_trace()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    #precompute_visual_features(config)
    test_precomputed(config)