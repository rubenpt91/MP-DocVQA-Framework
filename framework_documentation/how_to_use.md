# How to use

1. [Set-up environment](#set-up-environment)
2. [Train or evaluate](#train-and-evaluate)
3. [Configuration files and input arguments](#configuration-files-and-input-arguments)
   1. [Input Arguments](#input-arguments)
   2. [Datasets configuration files](#datasets-configuration-files)
   3. [Models configuration files](#models-configuration-files)
      1. [Retrieval module](#retrieval-module)
      2. [Visual Module](#visual-module)
      3. [Training parameters](#training-parameters)
4. [Visualization](#attention-visualization)

## Set-up environment

To start all the dependencies you only need to create a new conda environment with the provided yml file: <br>

```bash
$ conda env create -f environment.yml
$ conda activate mp_docvqa
```

## Train and evaluate

To use the framework you only need to call the `train.py` or `eval.py` scripts with the dataset and model you want to use. For example:

```python
python train.py --dataset MP-DocVQA --model HiVT5
```

The name of the dataset and the model **must** match the name of the configuration under the `configs/dataset` and `configs/models`. This allows to have different configs for the same dataset or model. For example in my case, I have `MP-DocVQA_local.yml`, and `MP-DocVQA_cluster.yml`. Depending on where to I run the script I use one or the other, where I specify the correct dataset path in each environment.

## Configuration files and input arguments

### Input arguments

| <div style="width:100px">Parameter </div> | <div style="width:150px">Input param </div> | Required 	  | Description                                                                                                                                                            |
|-------------------------------------------|---------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model                                     | `-m` `--model`                              | Yes         | Name of the model config file                                                                                                                                          |
| Dataset                                   | `-d` `--dataset`                            | Yes         | Name of the dataset config file                                                                                                                                        |
| Evaluation at start                       | `--no-eval-start`                           | No          | By default, before start training the framework performs an evaluation step to know the initial performance. By specifying this will skip the initial evaluation step. |
| Batch size                                | `-bs`, `--batch-size`                       | No          | Batch size*                                                                                                                                                            |
| Initialization seed                       | `--seed`                                    | No          | Initialization seed* **                                                                                                                                                |
| Parallelization                           | `--data-parallel`                           | No          | Specify utilizing multiple GPUs <br> **Currently not working**                                                                                                         |

- *Batch size and seed are specified in the configuration files. However, you can overwrite those parameters through the input parameters.
- **Although initialization seed is implemented. We have had different results with the same seed. If someone found the reason open an issue or email me :sweat_smile:

### Datasets configuration files

| Parameter      | Description                                                                                                                                                                                                                                                                                                                                                                                           | Values                         |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| dataset_name   | Name of the dataset to use.                                                                                                                                                                                                                                                                                                                                                                           | SP-DocVQA, MP-DocVQA, DUDE     |
| imdb_dir       | Path to the numpy annotations file.                                                                                                                                                                                                                                                                                                                                                                   | \<Path\>                       |
| images_dir     | Path to the images dir.                                                                                                                                                                                                                                                                                                                                                                               | \<Path\>                       |
| page_retrieval | Type of page retrieval system to be used. <br> - _Logits_ corresponds to the "Max conf." in the paper. <br> - _Oracle_ setup can't be used with _DUDE_ because it doesn't contain the answer page position. <br> - _Custom_ refers to the answer page prediction module. Therefore it can be used only with hierarchical models. <br> - If used in SP-DocVQA dataset, this parameter will be ignored. | Oracle, Concat, Logits, Custom |


### Models configuration files

| Parameter            | Description                                                                                                                                                                     | Values                                                          |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| model_name           | Name of the dataset to use.                                                                                                                                                     | BertQA, LayoutLMv2, LayoutLMv3, Longformer, BigBird, T5, Hi-VT5 |
| model_weights        | Path to the model weights dir. It can be either local path or huggingface weights id.                                                                                           | \<Path\>, \<Huggingface path\>                                  |
| page_tokens          | Number of [PAGE] tokens per page in **hierarchical** methods.                                                                                                                   | Integer: By default is 10 (as described in the paper)           |
| max_text_tokens      | Max number of text tokens **per page**. <br> <span style="color:red">**Currently this is implemented only in hierarchical methods**</span>                                      | Integer: Usually should be 512, 768 or 1024.                    |
| use_spatial_features | Boolean to ablate the **hierarchical** methods by using or not spatial features. <span style="color:red">**Implemented?**</span>                                                | True, False                                                     |
| use_visual_features  | Boolean to ablate the **hierarchical** methods by using or not visual features. <span style="color:red">**Implemented?**</span>                                                 | True, False                                                     |
| freeze_encoder       | Boolean to freeze the encoder in the **hierarchical** methods. This is used to train following the strategy described in the paper.                                             | True, False                                                     |
| save_dir             | Path where the checkpoints and log files will be saved.                                                                                                                         | \<Path\>                                                        |
| device               | Device to be used <span style="color:red">**Can I use cuda:1?**</span>                                                                                                          | CPU, cuda                                                       |
| data_parallel        | Use parallelism or not. <br> <span style="color:red">**CURRENTLY NOT IMPLEMENTED**</span>                                                                                       | True, False                                                     |
| retrieval_module     | Retrieval module parameters <br> Check section [Retrieval Module](#Retrieval Module) <br> <span style="color:red">**What if I don't want to have the retrieval module?**</span> |                                                                 |
| visual_module        | Visual module parameters <br> Check section [Visual Module](#Visual Module) <br> <span style="color:red">**What if I don't want to have the visual module?**</span>             |                                                                 |
| training_parameters  | The training parameters are specified in the model config file. <br> Check section [Training parameters](#Training parameters)                                                  | Oracle, Concat, Logits, Custom                                  |

#### Retrieval Module

* Retrieval module corresponds to the **Answer Page Prediction Module** described in the paper.
* This is used only for **Hierarchical** methods:

| Parameter     | Description                                                                                  | Values                  |
|---------------|----------------------------------------------------------------------------------------------|-------------------------|
| loss          | Loss to be used for the retrieval module. Currently only CrossEntropy is implemented.        | CrossEntropy            |
| loss_weight   | Scaling factor for the contribution of the Answer Page Prediction Module to the total loss.  | Float: 0.25 by default. |

#### Visual Module

* This is used only for **Hierarchical** methods:

| Parameter     | Description                                                                                                          | Values                         |
|---------------|----------------------------------------------------------------------------------------------------------------------|--------------------------------|
| model         | Name of the model to extract visual features to be used. <span style="color:red">**Is ViT still functional?**</span> | ViT, DiT                       |
| model_weights | Path to the model weights dir. It can be either local path or huggingface weights id.                                | \<Path\>, \<Huggingface path\> |


#### Training parameters

| Parameter           | Description                                                                                |
|---------------------|--------------------------------------------------------------------------------------------|
| lr                  | Learning rate.                                                                             |
| batch_size          | Batch size.                                                                                |
| train_epochs        | Number of epochs to train.                                                                 |
| warmup_iterations   | Number of iterations to perform learning rate warm-up.                                     |


## Attention visualization

<span style="color:red">**Currently this works only for Hi-VT5**</span>
