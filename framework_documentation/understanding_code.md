
# Understanding the framework and code

The framework has been implemented following a modular design, trying to make it the easiest possible to create new methods and datasets.

## Project structure 

It's quite straight-forward and self-explanatory:

```maarkdown
├── configs
│   ├── datasets
│   │   ├── SP-DocVQA.yml
│   │   ├── MP-DocVQA.yml
│   │   └── DUDE.yml
│   └── models
│       ├── BertQA.yml
│       ├── Longformer.yml
│       ├── BigBird.yml
│       ├── LayoutLMv2.yml
│       ├── LayoutLMv3.yml
│       ├── T5.yml
│       └── HiVT5.yml
├── datasets
│   ├── SP_DocVQA.py
│   ├── MP_DocVQA.py
│   └── DUDE.py
├── models
│   ├── BertQA.py
│   ├── Longformer.py
│   ├── BigBird.py
│   ├── LayoutLMv2.py
│   ├── LayoutLMv3.py
│   ├── T5.py
│   └── HiVT5.py
├── readme.md
├── environment.yml
├── utils.py
├── build_utils.py
├── logger.py
├── checkpoint.py
├── train.py
├── eval.py
├── metrics.py
└── visualization
    ├── fonts
    │   └── Arial.ttf
    ├── plot_attention_graphics.py
    ├── plot_enc_attention_img.py
    ├── store_attentions.py
    └── vis_utils.py

```

### Configs

In configs you will find the '_yml_' configuration files for the datasets and models. You can check the configuration files format in the [how to use section](how_to_use.md#datasets-configuration-files).


### Datasets

Here you will find the datasets' implementation. The dataset itself is quite simple since is based on the torch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). The most relevant keypoint is that all the datasets annotations are based on the imdb format of the Facebook MMF framework (You can download them [here](NotAvailableYet)). <br>
In the imdb files, the first record is the header with the version, split and creation date information. The rest is the data and contains the following information:

| Name                 | Description                                                                                                                        | Example                                                              |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| question_id          | ID corresponding to the question.                                                                                                  | 57349                                                                |
| question             | The question in string format.                                                                                                     | "What is the name of the company?"                                   |
| image_id             | Document ID. It matches with IDL documents IDs.                                                                                    | "snbx0223"                                                           |
| image_name           | List of images ids of all the pages used for the QA.                                                                               | \["snbx0223_p11", "snbx0223_p12", ..., "snbx0223_p22"\]              |
| imdb_doc_pages       | Number of pages considered for this QA.                                                                                            | 12                                                                   |
| total_doc_pages      | Total number of pages within the document.                                                                                         | 238                                                                  |
| ocr_tokens           | List of the OCR tokens for each page of the document. <br>It have the shape \[P, T\]                                               | \[\[lifestyle, apparel...case\], ..., \[itc, enduring...source:\] \] |
| ocr_normalized_boxes | Normalized \[0...1\] bounding boxes for the OCR tokens in the format \[left, top, right, bottom\]<br>It have the shape \[P, T\, 4] | \[\[\[0.07392304, 0.16985758, 0.2240048 , 0.19993986],...\], ...\]   |
| answers              | List of correct answers. In the imdb they are not lower cased. (It's performed in the dataloader).                                 | \["itc limited", "ITC Limited"\]                                     |
| answer_page_idx      | Page index where the answer is located.                                                                                            | 10                                                                   |

&nbsp;&nbsp;&nbsp;&nbsp;&ast; P: Number of pages. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&ast; T: Number of tokens.

## Models

The models are implemented by wrapping Huggingface models with an interface that will get the data from the dataloader and format it properly (tokenization, get start and end indices) to input to the model.

Note that the models are though to train either in _oracle_ or _concat_ set-ups, while _logits_ is only used in inference.

### Extractive methods

They require the start and end indices of the answer to train the models. This is performed in the `get_start_end_idx` function. But there are slight differences between the methods.

### Hi-VT5

Here we describe the main changes performed to add spatial and visual features. And to create the hierarchical architecture. To make it easier to understand, we follow a chronological order inside each subsection (as if you were debugging).

#### Spatial and Visual Features

* **Spatial features**: We define the [`SpatialEmbedding`](../models/HiLT5.py#L39) as four (x<sub>0</sub>, y<sub>0</sub>, x<sub>1</sub>, y<sub>1</sub>) `nn.Embedding` for 1000 (normalized bounding box coordinates scaled to 0-1000) and hidden size (512, 768, 1024). You can currently find 2 extra embeddings for width and height commented, since we didn't find any significant improvement using them.
  * In [lines 487-490](../models/HiVT5.py#L487) we get the **semantic embedding** of the words from the language model `self.shared`. Then we obtain the **spatial embedding** by sending the words' boxes to the `self.spatial_embedding` and sum up both representations. 


* **Visual features:** We define the [`VisualEmbedding`](../models/HiVT5.py#L39) using a DiT (alternatively ViT), freeze it and add a projection layer. We also define a function to create mock bounding boxes corresponding to the grid patches of the features extractor.
  * In [lines 492-498](../models/HiVT5.py#L492) the page images are sent to the `self.visual_embedding` with its corresponding mask (it might be necessary to add padding pages). Then, we get the visual boxes `self.visual_embeddings.get_visual_boxes`, send them to the `self.spatial_embedding`, and finally add the extracted visual features with the visual-spatial embedding.   
 

* Finally, in [lines 510-503](../models/HiVT5.py#L500)), spatial-semantic features are concatenated with the spatial-visual features. The same operation is performed for both masks, and they sent to the transformer encoder. Notice that all this processed is performed iteratively for each page of the document.

#### Hierarchical modifications

* In [lines 651](../models/HiVT5.py#L651) the special `[PAGE]` tokens are added into the tokenizer.
* In [line 689](../models/HiVT5.py#L689) in `prepare_vqa_input_ids` function. The `[PAGE]` tokens are prepended before the recognized OCR words. Note that the final output of this function, and accordingly the input to the model have a shape of [bs, P, X]. Where bs is batch_size, P is the maximum number of pages of the documents in the batch, and X is the OCR tokens or boxes.
* In [line 486](../models/HiVT5.py#L486), inside the core architecture (originally T5), we start iterating over the document's pages. Hence, all the operations will be applied page per page, but to all the samples in the batch at the same time (check indices `[:, page_idx]`)

* We get the spatial-visual-semantic representation and sent it to the transformer encoder as described in [Spatial and Visual Features](#spatial-and-visual-features) section. 

* In [lines 513-514](../models/HiVT5.py#L512) we keep the first **N** tokens of the encoder's output of each page, which corresponds to the `[PAGE]` tokens contextualized with the page information conditioned on the question. Hence, we build the summarized document representation from the concatenation of the contextualized `[PAGE]'`. To clarify, N=10 in the experiments we show in the paper. Therefore, if we have documents of 20 pages, the `document_embeddings` will have the shape `[bs, 200, H]`. Where H will be 512, 768 or 1024 depending on the model size (small, base, large).

* Then, in [lines 572-586](../models/HiVT5.py#L572) we send to the decoder the `document_embeddings` to get the final answer.

**Generation**
Generative models like T5 use the `model.generate()` function to perform inference.

<span style="color:red">**Check which modification I did in greedy search to make generate work**</span>

### Utils

Code with different functions. The most important ones might be `seed_everything`, to set the seed to torch, numpy, random, etc. And functions related to input arguments `parse_args` or  `check_config`, which checks that the configuration parameters are correct. For example, if you set _page_retrieval_ to _custom_ with Longformer, it's in this function where a ValueError is raised. Therefore, you might need to modify this if you create your own dataset or model.

### Build Utils

Very simple but important. Here you can find the functions `build_optimizer`, `build_dataset` and `build_model`. These functions are called from train and evaluation scripts to create its respective things. If you want to add a new optimizer or create a new dataset or model you'll need to include them in this functions. 

### Logger

Code dedicated to log to WandB all the losses and metrics. You can create your own new tags or change the project name in the `wb.init` function.

### Checkpoint

Code dedicated to store and load weights. Nothing remarkable.

### Metrics

Code dedicated to implement metrics such as Accuracy, ANLS or APPA. This is performed by the class `Evaluator` which stores variables to indicate if the evaluation must lower-case the ground truth, which is the ANLS threshold among others. It also keeps track of the best metrics reached and in which epochs.

