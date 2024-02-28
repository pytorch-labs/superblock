model_299.pth: https://drive.google.com/file/d/1GLKWOJYrzfq38I9bCYZ7v7_teTHe9k9o/view?usp=drive_link (this has tile size 64 and 0.8 sparsity)

## To Train:
Refer to TRAINING.md

## To Evaluate:
```
torchrun --nproc_per_node=2 evaluate.py  --model vit_b_16 --batch-size 256 --amp --model-ema --sparsity-linear 0.8 --sp-linear-tile-size 16 --weights-path ./model_299.pth --sparsify-weights --bsr 256
```
