# Hierarchical multimodal transformers for Multipage DocVQA

This repository implements the baselines and Hi-VT5 methods for the paper [Hierarchical multimodal transformers for Multipage DocVQA](https://arxiv.org/abs/2212.05935).

**DISCLAIMER:** 
1. The Hi-VT5 method was originally trained in [MMF framework](https://mmf.sh/). Although it is a very optimized framework, is very complex and we had to modify some core functions and classes to make the project work. In consequence, we decided to move it outside such complex framework and build a simpler version. The code for Hi-VT5 is exactly the same, but the weights are not transferable. The current released weights achieve lower performance, but we are currently working on getting the same results as the obtained with MMF framework.
2. The framework is constantly evolving, and we can't test all the features for each update we perform. Please, if you find any error / bug / inconsistency in the framework open an issue an we'll try to fix it as soon as posible.

## How to use
To use the framework please check [How to use](framework_documentation/how_to_use.md#how-to-use) instructions.


## Dataset

The dataset is aimed to perform Visual Question Answering on multipage industry scanned documents. The questions and answers are reused from Single Page DocVQA (SP-DocVQA) dataset. The images also corresponds to the same in original dataset with previous and posterior pages with a limit of up to 20 pages per document.

If you want to download the dataset, you can do so in the DocVQA challenge in the [RRC portal](https://rrc.cvc.uab.es/?ch=17&com=introduction), downloads section. For this framework, you will need to download the IMDBs (which contains processed QAs and OCR) and the images. All the downloads must be performed through the RRC portal.

| Dataset 		   | Link	                                                                          |
|--------------|--------------------------------------------------------------------------------|
| SP-DocVQA 	  | [Link](https://rrc.cvc.uab.es/?ch=17&com=downloads)	SP-DocVQA (Task 1) section |
| MP-DocVQA 	  | [Link](https://rrc.cvc.uab.es/?ch=17&com=downloads)	MP-DocVQA (Task 4) section |
| DUDE 		      | [Link](https://rrc.cvc.uab.es/?ch=23&com=downloads)	                           |

## Metrics

**Average Normalized Levenshtein Similarity (ANLS)** <br>
The standard metric for text-based VQA tasks (ST-VQA and DocVQA). It evaluates the method's reasoning capabilities while smoothly penalizes OCR recognition errors.
Check [Scene Text Visual Question Answering](https://arxiv.org/abs/1905.13648) for more details.

**Answer Page Prediction Accuracy (APPA)** <br>
In the MP-DocVQA task, the models can provide the index of the page where the information required to answer the question is located. For this subtask accuracy is used to evaluate the predictions: i.e. if the predicted page is correct or not.
Check [Hierarchical multimodal transformers for Multi-Page DocVQA](https://arxiv.org/abs/2212.05935) for more details.


## Leaderboard

Extended experimentation can be found in Table 2 of [Hierarchical multimodal transformers for Multi-Page DocVQA](https://arxiv.org/pdf/2212.05935.pdf).
You can also check the live leaderboard at the [RRC Portal](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=4).

| Model 		    | Weights HF name								                                                    | Parameters 	|	ANLS 		| APPA		|
|:-----------------|:--------------------------------------------------------------------------------------------------|:-------------:|:-------------:|:---------:|
| Bert large	    | [rubentito/bert-large-mpdocvqa](https://huggingface.co/rubentito/bert-large-mpdocvqa)			    | 334M 			| 0.4183 		| 51.6177 	|
| Longformer base	| [rubentito/longformer-base-mpdocvqa](https://huggingface.co/rubentito/longformer-base-mpdocvqa)	| 148M			| 0.5287		| 71.1696 	|
| BigBird ITC base | [rubentito/bigbird-base-itc-mpdocvqa](https://huggingface.co/rubentito/bigbird-base-itc-mpdocvqa) | 131M			| 0.4929		| 67.5433 	|
| LayoutLMv3 base	| [rubentito/layoutlmv3-base-mpdocvqa](https://huggingface.co/rubentito/layoutlmv3-base-mpdocvqa)	| 125M 			| 0.4538		| 51.9426 	|
| T5 base			| [rubentito/t5-base-mpdocvqa](https://huggingface.co/rubentito/t5-base-mpdocvqa)		            | 223M 			| 0.5050		| 0.0000 	|
| Hi-VT5 			| [rubentito/hivt5-base-mpdocvqa](https://huggingface.co/rubentito/hivt5-base-mpdocvqa)             | 316M 			| **0.6201**	| **79.23**	|


## Limitations
1. Hi-VT5 is **quite slow** both for training and inference. Notice that each page is passed through the encoder and this can be done up to for 20 pages.
2. It might **require a lot of memory** to train if you use all the document's pages. Although the complexity of the model is the same as their independent components (T5 + DiT). It requires to compute and store the gradients of the encoder for each page. This could be addressed with gradient checkpointing or with the next point, but they are not implemented.

Notice that points 1 are 2 could be addressed by parallelizing the process of each page (or set of P pages) into different GPUs. But this is also not implemented.

3. If you follow the training strategy described in the paper to train with reduced number of pages. You **need** to know which are the **page/s** with the necessary information **to answer the question**. Which you might not have for other datasets (like DUDE).
