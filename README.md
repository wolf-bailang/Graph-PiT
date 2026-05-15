# [Graph-PiT: Enhancing Structural Coherence in Part-Based Image Synthesis via Graph Priors](https://wolf-bailang.github.io/JunbinZhang/)

## Description

Achieving fine-grained and structurally sound controllability is a cornerstone of advanced visual generation. While recent part-based frameworks have pioneered component-based synthesis, they typically treat input parts as an unordered set. This neglect of intrinsic spatial and semantic relationships often results in compositions that lack structural integrity. To bridge this gap, we propose  Graph-PiT, a novel framework that explicitly models the structural dependencies of visual components using a graph prior. Specifically, we represent visual parts as nodes and their spatial-semantic relationships as edges. At the heart of our method is a Hierarchical Graph Neural Network (HGNN) module that performs relational message passing between coarse-grained super-nodes and fine-grained sub-node tokens, refining part embeddings before they enter the generative pipeline. Furthermore, we introduce a Graph Laplacian Regularization loss to ensure that connected components exhibit compatible latent features, effectively enforcing semantic glue between parts. Experimental results across four challenging domains—character, product, indoor layout, and jigsaw reconstruction-demonstrate that Graph-PiT significantly improves structural coherence and FID scores over the vanilla PiT baseline. Our approach not only enhances the plausibility of generated concepts but also offers a scalable and interpretable mechanism for complex, multi-part image synthesis.

关键词：image synthesis, diffusion models, graph neural networks, structural coherence. compositional AI

## Getting started with Graph-PiT

### Setup your environment

1. Clone the repo:

```bash
cd Graph-PiT
```

2. Install `uv`:

Instructions taken from [here](https://docs.astral.sh/uv/getting-started/installation/).

For linux systems this should be:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Install the dependencies:

```bash
uv sync
```

4. Activate your `.venv` and set the Python env:

```bash
source .venv/bin/activate
export PYTHONPATH=${PYTHONPATH}:${PWD}



conda create -n graph-pit python=3.12
source ~/.bashrc
conda activate graph-pit
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install accelerate==1.2.1 diffusers==0.32.1 einops==0.8.0 kornia==0.8.0 matplotlib==3.10.0 opencv-python==4.10.0.84 pandas==2.2.3 peft==0.14.0 protobuf==5.29.2 pyrallis==0.3.1 scikit-learn==1.6.1 scipy==1.15.0 sentencepiece==0.2.0 supervision==0.25.1 tensorboard==2.18.0 timm==1.0.12 transformers==4.47.1 wandb==0.19.1

huggingface-cli login
PIT-1  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

pip install modelscope
modelscope download --model black-forest-labs/FLUX.1-schnell --local_dir ./black-forest-labs/FLUX.1-schnell
modelscope download --model facebook/sam-vit-huge --local_dir ./facebook/sam-vit-huge
modelscope download --model microsoft/Florence-2-large --local_dir ./microsoft/Florence-2-large
modelscope download --model stabilityai/stable-diffusion-xl-base-1.0 --local_dir ./stabilityai/stable-diffusion-xl-base-1.0
modelscope download --model soulteary/h94-IP-Adapter --local_dir ./IP-Adapter

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

```

## Training Graph-PiT

### Data Generation

PiT assumes that the data is structured so that the the target images and part images are in the same directory with the naming convention being `image_name.jpg` for hte base image and `image_name_i.jpg` for the parts.

To use a generated data see the sample scripts

```bash
bash generate_character.sh
bash generate_IndoorLayout.sh
bash generate_Jigsaw.sh
bash generate_product.sh
```

### Data Processing

The generated dataset needs to be split into training and testing data, such as placing the original data in the product folder, the training data in the product_train folder, and the validation data in the product_val folder, and the test data in the product_test folder.

```bash
python scripts\data_process.py
```

### Training

For training see the `training/coach.py` file and the example below

```bash
bash train.sh
```

## Graph-PiT Inference

For inference see `scripts.infer.py` with the corresponding configs under `configs/infer`

```bash
bash test.sh
```

## Acknowledgments

Code is based on

- https://github.com/eladrich/PiT
