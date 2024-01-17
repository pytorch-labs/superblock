model_299.pth: https://drive.google.com/file/d/1GLKWOJYrzfq38I9bCYZ7v7_teTHe9k9o/view?usp=drive_link (this has tile size 64 and 0.8 sparsity)

```
python benchmark_supermask_model.py vit_b_16 ./model_299.pth --bsr 64 --sparsity-linear 0.8 --sp-linear-tile-size 64
```
