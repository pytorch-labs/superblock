# SuperBlock

SuperBlock combines two techniques for efficient neural network training and inference: Supermask and Block Compressed Sparse Row (BSR)

### Supermask
[Supermask](https://arxiv.org/abs/2207.00670) is a technique for applying structured sparsity to neural networks using a learned mask. It works by learning a continuous mask (scores) that is applied element-wise to the weights of a neural network layer. The mask scores are learned separately from the weights and are thresholded based on a target sparsity level to obtain a binary mask. The mask determines which weigths are kept and which are pruned, and is learned during training.

During inference, the binary mask is applied element-wise to the weights, pruning the weights that correspond to a 0 in the mask, resulting in a sparse network that can be efficiently computed. 

### Block compressed Sparse Row Format (BSR)
[The BSR format](https://pytorch.org/docs/main/sparse.html#sparse-bsr-tensor) is a sparse matrix representation that stores dense sub-blocks of non-zero elements instead of individual non-zero elements. The matrix is divided into equal-sized blocks, and only the non-zero blocks are stored.

The BSR format is efficient for sparse matrices with a block structure, where non-zero elements tend to cluster in dense sub-blocks. It reduces storage requirements and enables efficient matrix operations on the non-zero blocks.

Currently, the BSR format is optimized for Nvidia A100 GPU(s) only.

## Setup
To use SuperBlock, you will need
* [latest PyTorch Nightly](https://pytorch.org/get-started/locally/)

To train the model or evaluate accuracy, you will need:
* ImageNet2012 dataset

## Installation
* Clone this repo
  ```
  git clone https://github.com/pytorch-labs/superblock.git
  cd superblock
  ```
* Create a new conda environment
  ```
  conda create -n superblock
  conda activate superblock
  ```
* Install PyTorch Nightly
  ```
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
  ```


## Benchmarking
Baseline:
```
python benchmark.py \
  --model vit_b_16 \
  --batch-size 256 \
  > /dev/null
```
Result:
```
535.5450390625
```


80% sparsity random weights
```
python benchmark.py --model vit_b_16 \
  --batch-size 256 \
  --sparsity-linear 0.8 \
  --sp-linear-tile-size 64 \
  --sparsify-weights \
  --bsr 64 \
  > /dev/null
```
Result:
```
393.5461328125
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
* Skip the first transformer layer and/or last linear layer (ViT only):
    ```
    --skip-last-layer-sparsity
    --skip-first-transformer-sparsity
    ```

For example, if you would like to train a `vit_b_16` from scratch using Supermask, you can use the respective torchvision command found in [TRAINING.md](TRAINING.md) and append the supermask arguments:
```
torchrun --nproc_per_node=8 train.py\
    --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema\ 
    --sparsity-linear 0.9 --sp-linear-tile-size 32
```
Through this command, we are training a `vit_b_16` with 90% sparsity to linear layers using 32x32 tiles.

Please `python train.py --help` for a full list of available arguments.

## Evaluation

To run an evaluation of a Supermask-trained model, you can use [evaluate.py](evaluate.py).

* Offline sparsification with BSR:
    ```
    torchrun --nproc_per_node=8 evaluate.py  --model vit_b_16 --batch-size 256 --amp --sparsity-linear 0.9 --sp-linear-tile-size 32 --weights-path /path/to/model_299.pth  --data-path /path/to/imagenet --sparsify-weights --bsr 32
    ```
    This command applies 90% sparsity to linear layers using 32x32 tiles, loads the model weights from model_299.pth, loads the ImageNet validation set located at the specified path, applies offline sparsification to the weights, and converts the sparse weights to BSR format with a block size of 32. It is recommended to set `--bsr`      the same as tile size.

* Online sparsification without BSR:
  ```
  torchrun --nproc_per_node=8 evaluate.py --model vit_b_16 --batch-size 256 --amp --sparsity-linear 0.9 --sp-linear-tile-size 32 --weights-path /path/to/model.pth --data-path /path/to/imagenet
  ```
  This is similar to the previous command, but it does not apply offline sparsification or BSR conversion. Instead, the sparsity is applied on-the-fly during evaluation.

Please refer to the `get_args_parser` function in [evaluate.py](evaluate.py) for a full list of available arguments.

Results (1x A100):
* Sparsity= 0.9, Tile Size = 32, Online Sparsification, BSR = None
  ```
  Test:  Total time: 0:01:47
  Test:  Acc@1 76.078 Acc@5 92.654
  ```

* Sparsity= 0.9, Tile Size = 32, Offline Sparsification, BSR = None
  ```
  Test:  Total time: 0:01:45
  Test:  Acc@1 76.078 Acc@5 92.654
  ```

* Sparsity= 0.9, Tile Size = 32, Offline Sparsification, BSR = 32
  ```
  Test:  Total time: 0:01:18
  Test:  Acc@1 76.078 Acc@5 92.654
  ```

## Pretrained Weights

Download:
Instead of training from scratch, if you'd like to use the Supermask weights of `vit_b_16` trained on privacy mitigated Imagenet-blurred, you can download them:
```
mkdir checkpoints
# 80% sparsity, block size 32
wget https://huggingface.co/facebook/superblock-vit-b-16-sp0.80-ts32/resolve/main/pytorch_model.bin -O checkpoints/superblock-vit-b-16-sp0.80-ts32.pth
# 80% sparsity, block size 64
wget https://huggingface.co/facebook/superblock-vit-b-16-sp0.80-ts64/resolve/main/pytorch_model.bin -O checkpoints/superblock-vit-b-16-sp0.80-ts64.pth
```

Benchmark:
80% sparsity, block size 64
```
python benchmark.py --model vit_b_16 \
  --batch-size 256 \
  --sparsity-linear 0.8 \
  --sp-linear-tile-size 64 \
  --sparsify-weights \
  --bsr 64 \
  --weights-path ./checkpoints/superblock-vit-b-16-sp0.80-ts64.pth \
  > /dev/null
```
Result:
```
394.2301953125
```

Evaluate:
80% sparsity, block size 32
```
torchrun --nproc_per_node=8 evaluate.py --model vit_b_16 --batch-size 256 --amp --sparsity-linear 0.8 --sp-linear-tile-size 32 --weights-path checkpoints/superblock-vit-b-16-sp0.80-ts32.pth --data-path /path/to/imagenet
```
Results (1x A100):
```
  Test:  Total time: X
  Test:  Acc@1 78.040 Acc@5 93.756
```

80% sparsity, block size 64
```
torchrun --nproc_per_node=8 evaluate.py --model vit_b_16 --batch-size 256 --amp --sparsity-linear 0.8 --sp-linear-tile-size 64 --weights-path checkpoints/superblock-vit-b-16-sp0.80-ts64.pth --data-path /path/to/imagenet
```
Results (1x A100):
```
  Test:  Total time: X
  Test:  Acc@1 77.998 Acc@5 93.694
```

## License
SuperBlock is released under the [MIT license](https://github.com/pytorch-labs/superblock?tab=MIT-1-ov-file#readme).
