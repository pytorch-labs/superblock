# SuperBlock

SuperBlock combines two techniques for efficient neural network training and inference: Supermask and Block Compressed Sparse Row (BSR)

### Supermask
Supermask is a technique for applying structured sparsity to neural networks using a learned mask. It works by learning a continuous mask (scores) that is applied element-wise to the weights of a neural network layer. The mask scores are learned separately from the weights and are thresholded based on a target sparsity level to obtain a binary mask. The mask determines which weigths are kept and which are pruned, and is learned during training.

During inference, the binary mask is applied element-wise to the weights, pruning the weights that correspond to a 0 in the mask, resulting in a sparse network that can be efficiently computed. 

### Block compressed Sparse Row Format (BSR)
***INSERT NICE WORDS ABOUT BSR HERE***

## Setup
To use SuperBlock, you will need
* [latest PyTorch Nightly](https://pytorch.org/get-started/locally/)
* ImageNet2012 dataset

## Installation
* Clone this repo
  ```
  git clone https://github.com/pytorch-labs/superblock.git
  ```
* Create a new conda environment
  ```
  conda create -n superblock
  ```
* Install PyTorch Nightly
  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
  ```

## Training
Please refer to [TRAINING.md](TRAINING.md) for training from scratch. We use [Torchvision](https://github.com/pytorch/vision/tree/main/references/classification) as our framework for training. Supermask can be applied during training.

To apply supermask, we have the following arguments at our disposal,

* Apply Supermask to linear layers:
    ```
    --sparsity-linear
    --sp-linear-tile-size
    ```
* Apply Supermask to conv1x1 layers:
    ```
    --sparsity-conv1x1
    --sp-conv1x1-tile-size
    ```
* Apply Supermask to all other convolutional layers:
    ```
    --sparsity-conv
    --sp-conv-tile-size
    ```
* Skip first transformer layer and/or last linear layer (ViT only):
    ```
    --skip-last-layer-sparsity
    --skip-first-transformer-sparsity
    ```

For example, if you would like to train a `vit_b_16` from scratch using supermask, you can use the respective torchvision command found in [TRAINING.md](TRAINING.md) and append the supermask arguments:
```
torchrun --nproc_per_node=8 train.py\
    --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema\ 
    --sparsity-linear 0.9 --sp-linear-tile-size 32
```
Through this command, we are training a `vit_b_16` with 90% sparsity to linear layers using 32x32 tiles.

Please refer to the `get_args_parser` function in [train.py](train.py) for a full list of available arguments.

## Pretrained Weights

Instead of training from scratch, if you'd like to use the pretrained Supermask weights of `vit_b_16`, you can download them from here:

***INSERT LINK HERE***

## Evaluation

To run evaluation of a Supermask trained model, you can use [evaluate.py](evaluate.py).

* Offline sparsification with BSR:
    ```
    torchrun --nproc_per_node=8 evaluate.py  --model vit_b_16 --batch-size 256 --amp --sparsity-linear 0.9 --sp-linear-tile-size 32 --weights-path /path/to/model_299.pth  --data-path=/path/to/imagenet
    ```
    This command applies 90% sparsity to linear layers using 32x32 tiles, loads the model weights from model_299.pth, loads the ImageNet validation set located at the specified path, applies offline sparsification to the weights, and converts the sparse weights to BSR format with a block size of 256.

* Online sparsification without BSR:
  ```
  torchrun --nproc_per_node=8 evaluate.py --model vit_b_16 --batch-size 256 --amp --sparsity-linear 0.9 --sp-linear-tile-size 32 --weights-path /path/to/model_299.pth --data-path /path/to/imagenet
  ```
  This is similar to the previous command, but it does not apply offline sparsification or BSR conversion. Instead, the sparsity is applied on-the-fly during evaluation.

Please refer to the `get_args_parser` function in [evaluate.py](evaluate.py) for a full list of available arguments.

## Results (1xA100)

### Tile size=32, sparsity= 0.9, online, bsr=None
Test:  Total time: 0:02:20
Test:  Acc@1 0.088 Acc@5 0.540

### Tile size=32, sparsity= 0.9, offline, bsr=None
Test:  Total time: 0:02:14
Test:  Acc@1 0.088 Acc@5 0.540

### Tile size=32, sparsity= 0.9, offline, bsr=32
Test:  Total time: 0:01:40
Test:  Acc@1 0.088 Acc@5 0.540


## License
***INSERT LICENSE DETAILS HERE***