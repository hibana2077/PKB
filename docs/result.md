# Results

## Prev SOTA(1st and 2nd) and Ours - UFGVC datasets

| Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing |
|--------|-----|------|----------|---------|---------|
| CLE-ViT | 0.6333 | 0.4717 | 0.7850 | 0.7521 | 0.8214 |
| Mix-ViT | 0.6042 | 0.5617 | 0.7994| 0.5100 | 0.7630 |
| PKB-ViT | 0.700 | 0.657 | 0.838 | 0.682 | 0.8418 |

## Other UFG datasets

| Method | CUB_200 | Stanford_Cars | Standford_Dogs |
|--------|---------|---------------|----------------|
| PKB-ViT | 0.907 | ? | ? |
| PKB-R50 | 0.858 | ? | ? |
| PKB-R34 | 0.787 | ? | ? |
| Horospherical-Smart-R34-3d | 0.574 | ? | ? |
| Horospherical-Smart-R34-50d | 0.592 | ? | ? |
| LSDA-R50 | 0.867 | 0.943 | ? |

## Ablation Study - PKB and Original Augmentations

| Model | Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing | CUB_200 | Stanford_Cars |
|-------|--------|----------|---------|---------|-----------|-----------|---------|---------------|
| Resnet50 | Original Augmentations | 0.525 | 0.3883 | 0.7021 | 0.6715 | 0.2559 | 0.845 | ? |
| Resnet50 | PKB + Original Augmentations | 0.600 | 0.520 | 0.750 | 0.700 | 0.695 | 0.858 | ? |
