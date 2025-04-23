# in-scientific

## Overview

This repository is the official code for ["2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining"](https://arxiv.org/abs/2501.00958). It contains the implementation of pre-training LLaVA on our multimodal textbook (interleaved image-text corpora). Our dataset can be found in [Huggingface Dataset](https://huggingface.co/datasets/DAMO-NLP-SG/multimodal_textbook).

- Multimodal Textbook is a high-quality **pre-training corpus** that encompasses a wealth of foundational knowledge, which is presented in an **image-text interleaved format**.
- This textbook is constructed from **2.5 years of instructional videos**, amounting to 22,000 class hours, covering six fundamental subjects, including mathematics, physics, and others. 
- In multimodal textbooks, the text is transcribed from audio, and images are extracted from video's kekframe. They are closely aligned, and provide more coherent context.  
  


<img src="./src/page_fig.png" alt="Image" style="width: 900px;">  

## 🛠️ Installation

```
cd scientific
# create and activate an environment
conda create -n interleaved_textbook python=3.10 -y
conda activate interleaved_textbook

# install package
pip install --upgrade pip  
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118  
pip install -e .
pip install flash-attn --no-build-isolation
```


## Data Preparation
- Training Corpus: `multimodal_textbook.json` (11GB) + images folder (700GB)
- Benchmarks: OKVQA, TextVQA, scienceQ, Mathvista, mathvision, mathverse  in `./playground/data/eval/`

The full version of our dataset can be downloaded on our [Huggingface Dataset](https://huggingface.co/datasets/DAMO-NLP-SG/multimodal_textbook).


## Model Preparation
- clip-vit-large-patch14-336
- llava-v1.5-7b-sft

  
## 🔥 Training

### Pre-training with Llava-1.5
Similart to llava, `./llava/train/train_interleaved.py` is the training script for our pretraining.

```
cd scripts
./run_training_copy.sh
```
Note: 
> data_path: the interleaved dataset in scientific format. .
model_max_length: max length
max_num_images: The maximum number of images in the sample. Images exceeding this number will be ignored. 

```

