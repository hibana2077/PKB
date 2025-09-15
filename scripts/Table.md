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
|J009 | Cotton80 | Resnet50 | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.446 | 0.671 |
|J010 | CUB-200 | Resnet50 | Base | --hflip --rotate | None | 0.849 | 0.962 |
|J011 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.853 | 0.962 |
|J012 | CUB-200 | Resnet34 | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random --train-crop 224| 0.787 | 0.947 |
|J013 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement random | 0.856 | 0.961 |
|J014 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement random | 0.849 | 0.960 |
|J015 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.15 --pkb-sigma 1.5 --pkb-views 4 --pkb-placement random | 0.857 | 0.960 |
|J016 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.12 --pkb-sigma 1.5 --pkb-views 4 --pkb-placement random | 0.853 | 0.959 |
|J017 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.18 --pkb-sigma 1.2 --pkb-views 6 --pkb-placement random | 0.853 | 0.960 |
|J018 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.16 --pkb-sigma 1.5 --pkb-views 4 --pkb-placement random | 0.855 | 0.961 |
|J019 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.14 --pkb-sigma 1.4 --pkb-views 4 --pkb-placement random | 0.854 | 0.961 |
|J020 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 15 --pkb-a-frac 0.16 --pkb-sigma 1.5 --pkb-views 4 --pkb-placement random | 0.853 | 0.961 |
|J021 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.15 --pkb-sigma 1.4 --pkb-views 4 --pkb-placement random | 0.854 | 0.962 |
|J022 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.16 --pkb-sigma 1.5 --pkb-views 4 --pkb-placement random | 0.855 | 0.961 |
|J023 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 17 --pkb-a-frac 0.14 --pkb-sigma 1.5 --pkb-views 4 --pkb-placement random | 0.856 | 0.962 |
|J024 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.15 --pkb-sigma 1.8 --pkb-views 4 --pkb-placement random | 0.853 | 0.961 |
|J025 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 17 --pkb-a-frac 0.13 --pkb-sigma 1.4 --pkb-views 4 --pkb-placement random | 0.852 | 0.963 |
|J026 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.14 --pkb-sigma 1.4 --pkb-views 4 --pkb-placement random | 0.853 | 0.958 |
|J027 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.14 --pkb-sigma 1.45 --pkb-views 4 --pkb-placement random | 0.853 | 0.960 |
|J028 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 17 --pkb-a-frac 0.15 --pkb-sigma 1.4 --pkb-views 4 --pkb-placement random | 0.852 | 0.960 |
|J029 | Stanford_Cars | Resnet50 | Base | --hflip --rotate | None | 0.918 | 0.985 |
|J030 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.10 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.915 | 0.984 |
|J031 | Cotton80 | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.30 --pkb-sigma 3.0 --pkb-views 12 --pkb-placement random | 0.442 | 0.704 |
|J032 | Cotton80 | Resnet50 | PKB | --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 12 --pkb-placement random | 0.479 | 0.708 |
|J033 | Cotton80 | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.30 --pkb-sigma 2.5 --pkb-views 10 --pkb-placement random | 0.479 | 0.708 |
|J034 | Cotton80 | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.25 --pkb-sigma 3.5 --pkb-views 8 --pkb-placement random | 0.429 | 0.692 |
|J035 | Cotton80 | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.20 --pkb-sigma 4.0 --pkb-views 6 --pkb-placement random | 0.433 | 0.646 |
|J036 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.06 --pkb-sigma 0.6 --pkb-views 8 --pkb-placement random | 0.915 | 0.983 |
|J037 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.03 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.915 | 0.985 |
|J038 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.08 --pkb-sigma 0.5 --pkb-views 8 --pkb-placement random | 0.914 | ? |
|J039 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.12 --pkb-sigma 0.9 --pkb-views 8 --pkb-placement random | 0.917 | ? |
|J040 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.05 --pkb-sigma 0.3 --pkb-views 8 --pkb-placement random | 0.913 | 0.983 |
|J041 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.04 --pkb-sigma 0.8 --pkb-views 8 --pkb-placement dispersed | 0.917 | 0.984 |
|J042 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.05 --pkb-sigma 0.5 --pkb-views 8 --pkb-placement contiguous | 0.915 | 0.985 |
|J043 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.03 --pkb-sigma 0.9 --pkb-views 8 --pkb-placement random | 0.915 | 0.983 |
|J044 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.07 --pkb-sigma 0.6 --pkb-views 8 --pkb-placement dispersed | 0.913 | 0.984 |
|J045 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.05 --pkb-sigma 0.7 --pkb-views 8 --pkb-placement contiguous | 0.914 | 0.985 |
|J046 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.025 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement dispersed | 0.914 | 0.985 |
|J047 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.1 --pkb-sigma 0.4 --pkb-views 8 --pkb-placement random | 0.915 | 0.984 |
|J048 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 15 --pkb-a-frac 0.045 --pkb-sigma 0.75 --pkb-views 8 --pkb-placement contiguous | 0.914 | 0.984 |
|J049 | Stanford_Cars | Resnet50 | PKB | --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.06 --pkb-sigma 0.9 --pkb-views 8 --pkb-placement dispersed | 0.916 | 0.984 |

## G: G-series - CUB-200 fine-grid search

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|G01 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.10 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.852 | 0.962 |
|G02 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 15 --pkb-a-frac 0.12 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.856 | 0.963 |
|G03 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.10 --pkb-sigma 1.2 --pkb-views 6 --pkb-placement random | 0.851 | 0.961 |
|G04 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 16 --pkb-a-frac 0.11 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.855 | 0.966 |
|G05 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 17 --pkb-a-frac 0.10 --pkb-sigma 1.0 --pkb-views 6 --pkb-placement random | 0.855 | 0.961 |
|G06 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.09 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.858 | 0.966 |
|G07 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.09 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.855 | 0.961 |
|G08 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 22 --pkb-a-frac 0.09 --pkb-sigma 0.9 --pkb-views 8 --pkb-placement random | 0.855 | 0.962 |
|G09 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.11 --pkb-sigma 0.9 --pkb-views 8 --pkb-placement random | 0.852 | 0.961 |
|G10 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.08 --pkb-sigma 1.1 --pkb-views 8 --pkb-placement contiguous | 0.853 | 0.962 |
|G11 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 19 --pkb-a-frac 0.08 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement contiguous | 0.850 | 0.963 |
|G12 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.09 --pkb-sigma 1.2 --pkb-views 7 --pkb-placement contiguous | 0.858 | 0.963 |
|G13 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 17 --pkb-a-frac 0.09 --pkb-sigma 1.1 --pkb-views 8 --pkb-placement contiguous | 0.854 | 0.961 |
|G14 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.08 --pkb-sigma 1.0 --pkb-views 7 --pkb-placement contiguous | 0.855 | 0.964 |
|G15 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement contiguous | 0.844 | 0.963 |
|G16 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 7 --pkb-placement contiguous | 0.842 | 0.958 |
|G17 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 6 --pkb-placement contiguous | 0.840 | 0.959 |
|G18 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 5 --pkb-placement contiguous | 0.848 | 0.959 |
|G19 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 11 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 5 --pkb-placement contiguous | 0.846 | 0.960 |
|G20 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 5 --pkb-placement contiguous | 0.842 | 0.958 |
|G21 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 13 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 5 --pkb-placement contiguous | 0.843 | 0.957 |
|G22 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 14 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 5 --pkb-placement contiguous | 0.844 | 0.959 |
|G23 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 15 --pkb-a-frac 0.28 --pkb-sigma 4.0 --pkb-views 5 --pkb-placement contiguous | 0.840 | 0.957 |
|G24 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 19 --pkb-a-frac 0.09 --pkb-sigma 1.2 --pkb-views 8 --pkb-placement contiguous | 0.854 | 0.961 |
|G25 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 21 --pkb-a-frac 0.08 --pkb-sigma 1.0 --pkb-views 7 --pkb-placement contiguous | 0.852 | 0.962 |
|G26 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.085 --pkb-sigma 1.1 --pkb-views 8 --pkb-placement contiguous | 0.855 | 0.961 |
|G27 | CUB-200 | Resnet50 | PKB | --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.075 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement contiguous | 0.855 | 0.961 |

## P: P-series - CUB-200 parameter search

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
| P01  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 19 --pkb-a-frac 0.085 --pkb-sigma 1.2 --pkb-views 8 --pkb-placement contiguous  | 0.854 | 0.961 |
| P02  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.09 --pkb-sigma 1.15 --pkb-views 8 --pkb-placement contiguous  | 0.857 | 0.962 |
| P03  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.085 --pkb-sigma 1.1 --pkb-views 8 --pkb-placement contiguous  | 0.855 | 0.963 |
| P04  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 17 --pkb-a-frac 0.09 --pkb-sigma 1.2 --pkb-views 8 --pkb-placement contiguous   | 0.853 | 0.962 |
| P05  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.085 --pkb-sigma 1.2 --pkb-views 8 --pkb-placement contiguous  | 0.855 | 0.962 |
| P06  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 21 --pkb-a-frac 0.085 --pkb-sigma 1.15 --pkb-views 8 --pkb-placement contiguous | 0.857 | 0.963 |
| P07  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 19 --pkb-a-frac 0.09 --pkb-sigma 1.1 --pkb-views 7 --pkb-placement contiguous   | 0.854 | 0.963 |
| P08  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 20 --pkb-a-frac 0.085 --pkb-sigma 1.2 --pkb-views 7 --pkb-placement contiguous  | 0.853 | 0.962 |
| P09  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 18 --pkb-a-frac 0.08 --pkb-sigma 1.2 --pkb-views 8 --pkb-placement contiguous   | 0.856 | 0.963 |
| P10  | CUB-200 | Resnet50 | PKB| --hflip --rotate | --pkb-n 19 --pkb-a-frac 0.085 --pkb-sigma 1.15 --pkb-views 8 --pkb-placement contiguous | 0.852 | 0.963 |

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
|A010 | Stanford_Cars | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.945 | 0.992 |

## E: EfficientNet

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|E000 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.408 | 0.713 |
|E001 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | PKB | --color-jitter --hflip --rotate --train-crop 224 | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.425 | 0.700 |
|E002 | SoyGene | efficientnet_b0.ra4_e3600_r224_in1k | PKB | --color-jitter --hflip --rotate --train-crop 224 | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.582 | 0.801 |
|E003 | SoyGene | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.581 | 0.813 |

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
|EXA009 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random | 0.679 | 0.883 |
|EXA010 | SoyGene | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.773 | 0.931 |
|EXA011 | SoyGene | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.778 | 0.932 |
|EXA012 | SoyGlobal | tiny_vit_21m_384.dist_in22k_ft_in1k | Base | --color-jitter --hflip --rotate | None | 0.478 | 0.716 |
|EXA013 | Stanford_Cars | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.10 --pkb-sigma 1.0 --pkb-views 8 --pkb-placement random | 0.946 | 0.992 |

## EXB: Experiment B - Classic CNN models on each UFG dataset

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
|EXB000 | Cotton80 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.433 | 0.729 |
|EXB001 | Soybean | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.440 | 0.708 |
|EXB002 | SoyAgeing-R1 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.753 | 0.921 |
|EXB003 | CUB-200 | efficientnet_b0.ra4_e3600_r224_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.761 | 0.922 |
|EXB010 | Cotton80 | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.358 | 0.633 |
|EXB011 | Soybean | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.358 | 0.607 |
|EXB012 | SoyAgeing-R1 | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.639 | 0.858 |
|EXB013 | CUB-200 | inception_v3.tf_adv_in1k | Base | --color-jitter --hflip --rotate --train-crop 299 | None | 0.787 | 0.935 |
|EXB020 | Cotton80 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.500 | 0.742 |
|EXB021 | Soybean | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.503 | 0.760 |
|EXB022 | SoyAgeing-R1 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.809 | 0.945 |
|EXB023 | CUB-200 | densenet161.tv_in1k | Base | --color-jitter --hflip --rotate --train-crop 224 | None | 0.790 | 0.940 |

## M: Experiment C - PKB parameters search

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|------|---------|-------|------------|--------|----------|-----------|-----------|
| M000 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | Base| --color-jitter --hflip --rotate | None| 0.404 | 0.692 |
| M001 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement random| 0.438 | 0.725 |
| M002 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random| 0.500 | 0.754 |
| M003 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random| 0.454 | 0.738 |
| M004 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random| 0.442 | 0.758 |
| M005 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random| 0.487 | 0.758 |
| M006 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random| 0.467 | 0.713 |
| M007 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random| 0.517 | 0.767 |
| M008 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement dispersed | 0.504 | 0.717 |
| M009 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 4 --pkb-placement dispersed | 0.471 | 0.692 |
| M010 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement dispersed | 0.454 | 0.767 |
| M011 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 12 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 4 --pkb-placement random| 0.496 | 0.738 |
| M012 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random| 0.454 | 0.717 |
| M013 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate | --pkb-n 7 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random| 0.521 | 0.783 |

## N: Experiments D - Additional Experiments

| Experiment | Dataset | Model | Augmentation | Details | PKB Parameters | Validation Accuracy @1 | Validation Accuracy @5 |
|------------|---------|-------|--------------|---------|----------------|-----------------------|-----------------------|
| N001 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 7 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement dispersed| 0.475 | 0.733 |
| N002 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 6 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed| 0.504 | 0.758 |
| N003 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 7 --pkb-a-frac 0.40 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random| 0.458 | 0.758 |
| N004 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 7 --pkb-a-frac 0.50 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random| 0.467 | 0.738 |
| N005 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 7 --pkb-a-frac 0.25 --pkb-sigma 6.0 --pkb-views 8 --pkb-placement random| 0.537 | 0.775 |
| N006 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 8 --pkb-placement random| 0.542 | 0.771 |
| N007 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 5.0 --pkb-views 8 --pkb-placement random| 0.479 | 0.717 |
| N008 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 6.0 --pkb-views 8 --pkb-placement random| 0.508 | 0.750 |
| N009 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 7.0 --pkb-views 8 --pkb-placement random| 0.512 | 0.746 |
| N010 | Cotton80 | mobilenetv4_hybrid_medium.ix_e550_r384_in1k | PKB| --color-jitter --hflip --rotate  | --pkb-n 10 --pkb-a-frac 0.25 --pkb-sigma 8.0 --pkb-views 8 --pkb-placement random| 0.500 | 0.758 |

## TVA: Tiny ViT Parameter Experiments A - Cotton80

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TVA001 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 1.5 --pkb-views 8 --pkb-placement random | 0.667 | 0.883 |
| TVA002 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.688 | 0.879 |
| TVA003 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.667 | 0.900 |
| TVA004 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.700 | 0.883 |
| TVA005 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random | 0.675 | 0.883 |
| TVA006 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.683 | 0.879 |
| TVA007 | Cotton80 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | None | --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.700 | 0.875 |

## TVB: Tiny ViT Parameter Experiments B - Soybean

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TVB001 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 1.5 --pkb-views 8 --pkb-placement random | 0.530 | 0.797 |
| TVB002 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.545 | 0.803 |
| TVB003 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.540 | 0.805 |
| TVB004 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.540 | 0.790 |
| TVB005 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random | 0.568 | 0.817 |
| TVB006 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.543 | 0.800 |
| TVB007 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random | 0.657 | 0.890 |
| TVB008 | Soybean | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | None | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.602 | 0.868 |

## TVC: Tiny ViT Parameter Experiments C - SoyAgeing-R3

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TVC001 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 1.5 --pkb-views 8 --pkb-placement random | 0.817 | 0.945 |
| TVC002 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.828 | 0.951 |
| TVC003 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.825 | 0.949 |
| TVC004 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.816 | 0.948 |
| TVC005 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random | 0.821 | 0.946 |
| TVC006 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.818 | 0.949 |
| TVC007 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.24 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.826 | 0.949 |
| TVC008 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.26 --pkb-sigma 1.8 --pkb-views 8 --pkb-placement random | 0.823 | 0.947 |
| TVC009 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.22 --pkb-sigma 2.2 --pkb-views 8 --pkb-placement random | 0.823 | 0.949 |
| TVC010 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.24 --pkb-sigma 1.9 --pkb-views 8 --pkb-placement random | 0.823 | 0.949 |
| TVC011 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.828 | 0.956 |
| TVC012 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter --hflip --rotate | --pkb-n 10 --pkb-a-frac 0.15 --pkb-sigma 2.1 --pkb-views 10 --pkb-placement random | 0.824 | 0.952 |
| TVC013 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --color-jitter | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.809 | 0.947 |
| TVC014 | SoyAgeing-R3 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | None | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.860 | 0.957 |

## TVD: Tiny ViT Parameter Experiments D - CUB-200

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TVD001 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | None | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.901 | 0.981 |
| TVD002 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | None | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.900 | 0.978 |
| TVD003 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | None | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.899 | 0.982 |
| TVD004 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.904 | 0.978 |
| TVD005 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.907 | 0.980 |
| TVD006 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.905 | 0.977 |
| TVD007 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 2 --pkb-placement random | 0.904 | 0.978 |
| TVD008 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 4.0 --pkb-views 2 --pkb-placement dispersed | 0.905 | 0.978 |
| TVD009 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.25 --pkb-sigma 5.0 --pkb-views 2 --pkb-placement contiguous | 0.904 | 0.978 |
| TVD010 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.15 --pkb-sigma 5.0 --pkb-views 2 --pkb-placement contiguous | 0.907 | 0.978 |
| TVD011 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 4 --pkb-a-frac 0.35 --pkb-sigma 5.0 --pkb-views 2 --pkb-placement contiguous | 0.901 | 0.976 |
| TVD012 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.18 --pkb-sigma 5.0 --pkb-views 2 --pkb-placement contiguous | ? | ? |
| TVD013 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.19 --pkb-sigma 5.0 --pkb-views 2 --pkb-placement contiguous | ? | ? |
| TVD014 | CUB-200 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.25 --pkb-sigma 5.0 --pkb-views 2 --pkb-placement contiguous | ? | ? |

## TVE: Tiny ViT Parameter Experiments E - SoyGene

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|TVE001 | SoyGene | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.813 | 0.957 |
|TVE002 | SoyGene | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.801 | 0.956 |
|TVE003 | SoyGene | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 4 --pkb-placement dispersed | 0.838 | 0.961 |

## TVF: Tiny ViT Parameter Experiments F - SoyAgeing-R{4,5,6}

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|TVF001 | SoyAgeing-R4 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.873 | 0.975 |
|TVF002 | SoyAgeing-R5 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.883 | 0.981 |
|TVF003 | SoyAgeing-R6 | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.763 | 0.916 |

## TVG: Tiny ViT Parameter Experiments G - SoyGlobal

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|TVG001 | SoyGlobal | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement random | 0.682 | 0.867 |
|TVG002 | SoyGlobal | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement dispersed | 0.661 | 0.646 |
|TVG003 | SoyGlobal | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 2.0 --pkb-views 10 --pkb-placement contiguous | 0.638 | 0.846 |

## TVH: Tiny ViT Parameter Experiments H - NAbird

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|TVH001 | NAbird | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 10 --pkb-placement random | ? | ? |
|TVH002 | NAbird | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 10 --pkb-placement dispersed | ? | ? |
|TVH003 | NAbird | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 3.0 --pkb-views 10 --pkb-placement contiguous | ? | ? |
|TVH004 | NAbird | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 5.0 --pkb-views 10 --pkb-placement random | ? | ? |
|TVH005 | NAbird | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 5.0 --pkb-views 10 --pkb-placement dispersed | ? | ? |
|TVH006 | NAbird | tiny_vit_21m_384.dist_in22k_ft_in1k | PKB | --hflip --rotate | --pkb-n 8 --pkb-a-frac 0.20 --pkb-sigma 5.0 --pkb-views 10 --pkb-placement contiguous | ? | ? |

## R34 - PKB on UFGVC {Cotton80, Soybean, SoyAgeing-R{1,3,4,5,6}, SoyGene, SoyGlobal}

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|R3400 | Cotton80 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.30 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.350 | 0.646 |
|R3401 | Cotton80 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.321 | 0.650 |
|R3402 | Cotton80 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.325 | 0.617 |
|R3403 | Soybean | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random | 0.413 | 0.735 |
|R3404 | Soybean | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.380 | 0.693 |
|R3405 | Soybean | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.453 | 0.735 |
|R3406 | SoyAgeing-R1 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.702 | 0.901 |
|R3407 | SoyAgeing-R1 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.701 | 0.890 |
|R3408 | SoyAgeing-R1 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.719 | 0.909 |
|R3409 | SoyAgeing-R3 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.680 | 0.894 |
|R3410 | SoyAgeing-R3 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.626 | 0.869 |
|R3411 | SoyAgeing-R3 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.665 | 0.892 |
|R3412 | SoyAgeing-R4 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.706 | 0.905 |
|R3413 | SoyAgeing-R4 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.696 | 0.912 |
|R3414 | SoyAgeing-R4 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.686 | 0.902 |
|R3415 | SoyAgeing-R5 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.719 | 0.940 |
|R3416 | SoyAgeing-R5 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.721 | 0.926 |
|R3417 | SoyAgeing-R5 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.713 | 0.928 |
|R3418 | SoyAgeing-R6 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.587 | 0.837 |
|R3419 | SoyAgeing-R6 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.565 | 0.843 |
|R3420 | SoyAgeing-R6 | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.584 | 0.829 |
|R3421 | SoyGene | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.530 | 0.804 |
|R3422 | SoyGene | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.439 | 0.729 |
|R3423 | SoyGene | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.563 | 0.829 |
|R3424 | SoyGlobal | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.380 | 0.624 |
|R3425 | SoyGlobal | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | ? | ? |
|R3426 | SoyGlobal | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | ? | ? |
|R3427 | Stanford_Cars | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.913 | 0.983 |
|R3428 | Stanford_Cars | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.915 | 0.984 |
|R3429 | Stanford_Cars | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.912 | 0.985 |
|R3430 | NAbird | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | ? | ? |
|R3431 | NAbird | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | ? | ? |
|R3432 | NAbird | resnet34.a1_in1k | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | ? | ? |
|R3433 | NAbird | resnet34.a1_in1k | Base | --hflip --rotate | None | ? | ? |

## R50 - PKB on UFGVC {Cotton80, Soybean, SoyAgeing-R{1,3,4,5,6}, SoyGene, SoyGlobal}

| Code | Dataset | Model | Aug Method | Detail | PKB parm | Val Acc@1 | Val Acc@5 |
|R5003 | Soybean | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 3.0 --pkb-views 8 --pkb-placement random | 0.480 | 0.755 |
|R5004 | Soybean | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.522 | 0.807 |
|R5005 | Soybean | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.498 | 0.770 |
|R5006 | SoyAgeing-R1 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.770 | 0.917 |
|R5007 | SoyAgeing-R1 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.760 | 0.923 |
|R5008 | SoyAgeing-R1 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.757 | 0.931 |
|R5009 | SoyAgeing-R3 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.745 | 0.919 |
|R5010 | SoyAgeing-R3 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.731 | 0.920 |
|R5011 | SoyAgeing-R3 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.730 | 0.917 |
|R5012 | SoyAgeing-R4 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.740 | 0.920 |
|R5013 | SoyAgeing-R4 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.740 | 0.925 |
|R5014 | SoyAgeing-R4 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.740 | 0.914 |
|R5015 | SoyAgeing-R5 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.769 | 0.948 |
|R5016 | SoyAgeing-R5 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.780 | 0.951 |
|R5017 | SoyAgeing-R5 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.781 | 0.946 |
|R5018 | SoyAgeing-R6 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | 0.642 | 0.870 |
|R5019 | SoyAgeing-R6 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | 0.640 | 0.869 |
|R5020 | SoyAgeing-R6 | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | 0.641 | 0.855 |
|R5021 | SoyGene | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | ? | ? |
|R5022 | SoyGene | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | ? | ? |
|R5023 | SoyGene | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | ? | ? |
|R5024 | SoyGlobal | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | ? | ? |
|R5025 | SoyGlobal | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | ? | ? |
|R5026 | SoyGlobal | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | ? | ? |
|R5027 | NAbird | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement random | ? | ? |
|R5028 | NAbird | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement contiguous | ? | ? |
|R5029 | NAbird | resnet50 | PKB | --hflip --rotate | --pkb-n 6 --pkb-a-frac 0.25 --pkb-sigma 2.0 --pkb-views 8 --pkb-placement dispersed | ? | ? |
|R5030 | NAbird | resnet50 | Base | --hflip --rotate | None | ? | ? |

## Ablation Study - blur and 
- 改P