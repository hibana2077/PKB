# Code Mapping

## J: Resnet50

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|J000 | Cotton80 | Resnet50 | Base | --color-jitter --hflip --rotate | None | 0.438 | 0.692 |
|J001 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 20 --pkb-placement random | 0.442 | 0.683 |
|J002 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 20 --pkb-placement random | 0.438 | 0.704 |
|J003 | Cotton80 | Resnet50 | cutout | --color-jitter --hflip --rotate | None | 0.367 | 0.667 |
|J004 | Cotton80 | Resnet50 | fullblur | --color-jitter --hflip --rotate | None | 0.100 | 0.217 |

## A: Tiny ViT

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|A000 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.675 | 0.887 |
|A001 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement random | 0.679 | 0.879 |

## E: EfficientNet

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|E000 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate | None |  |  |

## S: Swin Transformer

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|S000 | Cotton80 | swin_base_patch4_window12_384.ms_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None |  |  |