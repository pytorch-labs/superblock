model_299.pth: https://drive.google.com/file/d/1GLKWOJYrzfq38I9bCYZ7v7_teTHe9k9o/view?usp=drive_link (ViT_b_16 trained from scratch on ImageNet with tile size 64 and 0.8 sparsity)

## To Train:
Refer to TRAINING.md to train on ImageNet using block sparsity

## To Evaluate:
```
python evaluate.py  --model vit_b_16 --batch-size 256 --amp --sparsity-linear 0.9 --sp-linear-tile-size 32 --weights-path /home/jessecai/local/supermask/model_300.pth  --data-path=/home/jessecai/local/DATA/imagenet2012 --sparsify-weights --bsr 32
```


## Tile size=32, sparsity= 0.9, online, bsr=None
Test:  Total time: 0:02:20
Test:  Acc@1 0.088 Acc@5 0.540

## Tile size=32, sparsity= 0.9, offline, bsr=None
Test:  Total time: 0:02:14
Test:  Acc@1 0.088 Acc@5 0.540

## Tile size=32, sparsity= 0.9, offline, bsr=32
Test:  Total time: 0:01:40
Test:  Acc@1 0.088 Acc@5 0.540
