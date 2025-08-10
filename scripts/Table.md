# Code Mapping

## J: Resnet50

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|J000 | Cotton80 | Resnet50 | Base | --color-jitter --hflip --rotate | None | 0.438 | 0.692 |
|J001 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 20 --pkb-placement random | 0.442 | 0.683 |
|J002 | Cotton80 | Resnet50 | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 20 --pkb-placement random |  |  |
|J003 | Cotton80 | Resnet50 | cutout | --color-jitter --hflip --rotate | None |  |  |
|J004 | Cotton80 | Resnet50 | fullblur | --color-jitter --hflip --rotate | None |  |  |

## A: Tiny ViT

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|A000 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None |  |  |
|A001 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 20 --pkb-placement random |  |  |
