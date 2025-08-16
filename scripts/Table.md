# Code Mapping

## J: Resnet50

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|J000 | Cotton80 | Resnet50 | Base | --color-jitter --hflip --rotate | None | 0.438 | 0.692 |
|J001 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 20 --pkb-placement random | 0.442 | 0.683 |
|J002 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 20 --pkb-placement random | 0.438 | 0.704 |
|J003 | Cotton80 | Resnet50 | cutout | --color-jitter --hflip --rotate | None | 0.367 | 0.667 |
|J004 | Cotton80 | Resnet50 | fullblur | --color-jitter --hflip --rotate | None | 0.100 | 0.217 |
|J005 | SoyAgeing-R1 | Resnet50 | Base | --color-jitter --hflip --rotate | None | 0.731 | 0.904 |
|J006 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 4 --pkb-placement random | 0.446 | 0.692 |
|J007 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 4 --pkb-placement random | 0.429 | 0.667 |
|J008 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 4 --pkb-placement random | 0.429 | 0.700 |
|J009 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 4 --pkb-placement random | 0.433 | 0.688 |

## A: Tiny ViT

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|A000 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.675 | 0.887 |
|A001 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement random | 0.679 | 0.879 |
|A002 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.696 | 0.879 |
|A003 | SoyAgeing-R1 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.824 | 0.958 |
|A004 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.819 | 0.954 |
|A005 | SoyAgeing-R4 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.802 | 0.940 |
|A006 | SoyAgeing-R5 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.820 | 0.956 |
|A007 | SoyAgeing-R6 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.695 | 0.899 |
|A008 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.520 | 0.800 |
|A009 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.896 | 0.985 |

## E: EfficientNet

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|E000 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.408 | 0.713 |
|E001 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | PKB | --color-jitter --hflip --rotate --train-crop 224 | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.425 | 0.700 |

## F: EVA02

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|F000 | Cotton80 | eva02_small_patch14_336.mim_in22k_ft_in1k | Base | --color-jitter --hflip --rotate --train-crop 336 | None | 0.054 | 0.129 |

## S: Swin Transformer

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|S000 | Cotton80 | swin_base_patch4_window12_384.ms_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.438 | 0.704 |
|S001 | Cotton80 | swin_base_patch4_window12_384.ms_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --batch-size 8 --pkb-placement random | 0.283 | 0.542 |

## C: ConvNeXt

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|C000 | Cotton80 | convnextv2_tiny.fcmae_ft_in22k_in1k_384 | Base | --color-jitter --hflip --rotate | None | 0.483 | 0.750 |
|C001 | Cotton80 | convnextv2_tiny.fcmae_ft_in22k_in1k_384 | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.504 | 0.767 |

## EXA: Experiment A - Tiny vit on each UFG dataset

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|EXA000 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.696 | 0.887 |
|EXA001 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.537 | 0.807 |
|EXA002 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 16 --pkb-placement random | 0.510(Time out) | 0.798(Time out) |
|EXA003 | SoyAgeing-R1 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.830 | 0.953 |
|EXA004 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.829 | 0.955 |
|EXA005 | SoyAgeing-R4 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.815 | 0.948 |
|EXA006 | SoyAgeing-R5 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.823 | 0.957 |
|EXA007 | SoyAgeing-R6 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.710 | 0.895 |
|EXA008 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random | 0.898(Time out) | 0.982(Time out) |

## EXB: Experiment B - Classic CNN models on each UFG dataset

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|EXB000 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.433 | 0.729 |
|EXB001 | Soybean | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.440 | 0.708 |
|EXB002 | SoyAgeing-R1 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.753 | 0.921 |
|EXB010 | Cotton80 | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.358 | 0.633 |
|EXB011 | Soybean | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.358 | 0.607 |
|EXB012 | SoyAgeing-R1 | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.639 | 0.858 |
|EXB020 | Cotton80 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.500 | 0.742 |
|EXB021 | Soybean | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.503 | 0.760 |
|EXB022 | SoyAgeing-R1 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.809 | 0.945 |
|EXB023 | CUB-200 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.790 | 0.940 |

## M: Experiment C - PKB parameters search

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
| M000 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | Base| --color-jitter --hflip --rotate --train-crop 384 | None| 0.404 | 0.692 |
| M001 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 6  --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 4  --pkb-placement random| 0.438 | 0.725 |
| M002 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 6  --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8  --pkb-placement random| 0.500 | 0.754 |
| M003 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 8  --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8  --pkb-placement random| 0.454 | 0.738 |
| M004 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 8  --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8  --pkb-placement random| 0.442 | 0.758 |
| M005 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 8  --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8  --pkb-placement random| 0.487 | 0.758 |
| M006 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 8  --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8  --pkb-placement random| 0.467 | 0.713 |
| M007 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 8  --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8  --pkb-placement random| 0.517 | 0.767 |
| M008 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 6  --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 4  --pkb-placement dispersed | 0.504 | 0.717 |
| M009 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 8  --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 4  --pkb-placement dispersed | 0.471 | 0.692 |
| M010 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8  --pkb-placement dispersed | 0.454 | 0.767 |
| M011 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 12 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 4  --pkb-placement random| 0.496 | 0.738 |
| M012 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 6  --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8  --pkb-placement random| 0.454 | 0.717 |
| M013 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 7  --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8  --pkb-placement random| 0.521 | 0.783 |

## N: Experiments D - Additional Experiments

| Experiment | Dataset | Model | Augmentation | Details | PKB Parameters | Validation Accuracy @1 | Validation Accuracy @5 |
|------------|---------|-------|--------------|---------|----------------|-----------------------|-----------------------|
| N001 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 7 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed| 0.475 | 0.733 |
| N002 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate --train-crop 384 | --pkb-n 6 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed| 0.504 | 0.758 |