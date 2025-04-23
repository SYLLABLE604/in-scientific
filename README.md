# in-scientific

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

