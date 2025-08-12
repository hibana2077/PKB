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

## A: Tiny ViT

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|A000 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.675 | 0.887 |
|A001 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement random | 0.679 | 0.879 |
|A002 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.696 | 0.879 |
|A003 | SoyAgeing-R1 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.824 | 0.958 |
|A004 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.819 | 0.954 |
|A005 | SoyAgeing-R4 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | Run yet | Run yet |
|A006 | SoyAgeing-R5 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | Run yet | Run yet |
|A007 | SoyAgeing-R6 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | Run yet | Run yet |


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
|EXA002 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 16 --pkb-placement random | Run yet | Run yet |
|EXA003 | SoyAgeing-R1 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | Run yet | Run yet |

## EXB: Experiment B - Classic CNN models on each UFG dataset

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|EXB000 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.433 | 0.729 |
|EXB001 | Soybean | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.440 | 0.708 |
|EXB002 | SoyAgeing-R1 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | Run yet | Run yet |
|EXB010 | Cotton80 | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.358 | 0.633 |
|EXB011 | Soybean | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.358 | 0.607 |
|EXB012 | SoyAgeing-R1 | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | Run yet | Run yet |
|EXB020 | Cotton80 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.500 | 0.742 |
|EXB021 | Soybean | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.503 | 0.760 |
|EXB022 | SoyAgeing-R1 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | Run yet | Run yet |