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
| PKB-R50 | 0.858 | 0.917 | ? |
| PKB-R34 | 0.787 | ? | ? |
| Horospherical-Smart-R34-3d | 0.574 | 0.802 | ? |
| Horospherical-Smart-R34-50d | 0.592 | 0.829 | ? |
| ISDA-R50 | 0.853 | 0.932 | ? |
| LSDA-R50 | 0.867 | 0.943 | ? |

## Ablation Study - PKB and Original Augmentations

| Model | Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing | CUB_200 | Stanford_Cars |
|-------|--------|----------|---------|---------|-----------|-----------|---------|---------------|
| Tiny_ViT | Original Augmentations | 0.675 | 0.4717 | 0.7850 | 0.478 | 0.792 | 0.896 | 0.945 |
| Tiny_ViT | PKB + Original Augmentations | 0.700 | 0.657 | 0.838 | 0.682 | 0.8418 | 0.907 | ? |

## Ablation Study - PKB and Null Baseline

| Model | Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing | CUB_200 | Stanford_Cars |
|-------|--------|----------|---------|---------|-----------|-----------|---------|---------------|
| Tiny_ViT | cutout |
| Tiny_ViT | fullblur |