
# Understanding the framework and code

The framework has been implemented following a modular design, trying to make it the easiest possible to create new methods and datasets.

## Project structure 

It's quite straight-forward and self-explanatory:

&ndash; configs <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&ndash; datasets <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* MP-DocVQA.yml <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&ndash; models <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* HiVT5.yml <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br>

&ndash; datasets <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* SP-DocVQA.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* MP-DocVQA.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* DUDE.py <br>

&ndash; models <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* BertQA.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Longformer.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* BigBird.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* LayoutLMv2.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* LayoutLMv3.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* T5.py <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Hi-VT5.py <br>

&ndash; visualization <br>

&ast; utils.py <br>
&ast; build_utils.py <br>
&ast; checkpoint.py <br>
&ast; logger.py <br>
&ast; eval.py <br>
&ast; train.py <br>

### Configs

In configs you will find the '_yml_' configuration files for the datasets and models. You can check the configuration files format in the [how to use section](how_to_use#datasets-configuration-files). 


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

In the models and datasets directories you will find the datasets and models implementation.

## Models

The models are implemented by wrapping Huggingface models with an interface that will get the data from the dataloader and format it properly (tokenization, get start and end indices) to input to the model.

Note that the models are though to train either in _oracle_ or _concat_ set-ups, while _logits_ is only used in inference.

### Extractive methods

They require the start and end indices of the answer to train the models. This is performed in the `get_start_end_idx` function. But there are slight differences between the methods.

### Hi-VT5

Here we describe the main changes performed to add spatial and visual features. And to create the hierarchical architecture.

#### Spatial and Visual Features

* **Spatial features**: We define the [`SpatialEmbedding`]() as four (x<sub>0</sub>, y<sub>0</sub>, x<sub>1</sub>, y<sub>1</sub>) `nn.Embedding` for 1000 (normalized bounding box coordinates scaled to 0-1000) and hidden size (512, 768, 1024). You can currently find 2 extra embeddings for width and height commented, but we didn't find any significant improvement using them.
  * In models/HiVT5.py#L487 we get the **semantic embedding** of the words from the language model `self.shared`. Then we obtain the **spatial embedding** by sending the words' boxes to the `self.spatial_embedding` and sum up both representations. 


* **Visual features:** We define the [`VisualEmbedding`](#models/HiVT5.py#L39) using a DiT (alternatively ViT), freeze it and add a projection layer. We also define a function to create mock bounding boxes corresponding to the grid patches of the features extractor.
  * In #models/HiVT5.py#L493 the page images are sent to the `self.visual_embedding` with its corresponding mask (it might be necessary to add padding pages). Then, we get the visual boxes `self.visual_embeddings.get_visual_boxes`, send them to the `self.spatial_embedding`, and finally add the extracted visual features with the visual-spatial embedding.   
 

* Then in models/HiVT5.py#L500, spatial-semantic features are concatenated with the spatial-visual features, and sent to the transformer encoder. Notice that all this processed is performed iteratively for each page of the document.



* Builders
* Metrics
* Datasets (IMDBs)
* Hi-VT5
