# Results

## Prev SOTA(1st and 2nd) and Ours - UFGVC datasets

| Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing |
|--------|-----|------|----------|---------|---------|
| CLE-ViT | 0.6333 | 0.4717 | 0.7850 | 0.7521 | 0.8214 |
| Mix-ViT | 0.6042 | 0.5617 | 0.7994| 0.5100 | 0.7630 |
| PKB-ViT | 0.700 | 0.657 | 0.838 | 0.682 | 0.8418 |

## SoyAgeing

| Method | SoyAgeing | SoyAgeing_R1 | SoyAgeing_R3 | SoyAgeing_R4 | SoyAgeing_R5 | SoyAgeing_R6 |
|--------|-----------|--------------|--------------|--------------|--------------|--------------|
|ViT | 0.792 | 0.824 | 0.819 | 0.802 | 0.820 | 0.695 |
| PKB-ViT | 0.8418 | 0.830 | 0.860 | 0.873 | 0.883 | 0.763 |
| CLE-ViT | 0.8214 | 0.8010 | 0.8333 | 0.8424 | 0.8636 | 0.7596 |
| Mix-ViT | 0.7630 | 0.7929 | 0.7717 | 0.7798 | 0.7919 | 0.6788 |
| ResNet50 | 0.6784 | 0.731 | 0.685 | 0.671 | 0.713 | 0.592 |
| ResNet34 | 0.629 | 0.668 | 0.653 | 0.640 | 0.660 | 0.524 |
| PKB-R50 | 0.736 | 0.770 | 0.745 | 0.740 | 0.781 | 0.642 |
| PKB-R34 | 0.683 | 0.719 | 0.680 | 0.706 | 0.721 | 0.587 |

## Other UFG datasets

| Method | CUB_200 | Stanford_Cars | Standford_Dogs |
|--------|---------|---------------|----------------|
| PKB-ViT | 0.907 | 0.946 | ? |
| PKB-R50 | 0.858 | 0.917 | ? |
| PKB-R34 | 0.787 | 0.922 | ? |
| Horospherical-Smart-R34-3d | 0.574 | 0.802 | ? |
| Horospherical-Smart-R34-50d | 0.592 | 0.829 | ? |
| ISDA-R50 | 0.853 | 0.932 | ? |
| LSDA-R50 | 0.867 | 0.943 | ? |

## Ablation Study - PKB and Original Augmentations

| Model | Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing | CUB_200 | Stanford_Cars |
|-------|--------|----------|---------|---------|-----------|-----------|---------|---------------|
| Tiny_ViT | Original Augmentations | 0.675 | 0.4717 | 0.7850 | 0.478 | 0.792 | 0.896 | 0.945 |
| Tiny_ViT | PKB + Original Augmentations | 0.700 | 0.657 | 0.838 | 0.682 | 0.8418 | 0.907 | 0.946 |

## Ablation Study - R50 and R34

| Model | Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing | CUB_200 | Stanford_Cars |
|-------|--------|----------|---------|---------|-----------|-----------|---------|---------------|
| ResNet50 | Original Augmentations | 0.438 | ? | ? | ? | ? | ? | ? |
| ResNet50 | PKB + Original Augmentations | 0.479 | ? | ? | ? | ? | ? | ? |
| ResNet34 | PKB + Original Augmentations | 0.650 | 0.600 | 0.810 | 0.650 | 0.820 | 0.787 | 0.922 |

## Ablation Study - PKB and Null Baseline

| Model | Method | Cotton80 | Soybean | SoyGene | SoyGlobal | SoyAgeing | CUB_200 | Stanford_Cars |
|-------|--------|----------|---------|---------|-----------|-----------|---------|---------------|
| Tiny_ViT | cutout |
| Tiny_ViT | fullblur |