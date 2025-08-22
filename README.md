# PKB

## Augmentation Demo

Use `demo_augmentations.py` to visualize before/after effects of PatchKeepBlur and related augmentations on a subset of any supported dataset.

Basic example (PowerShell):

```pwsh
python demo_augmentations.py --dataset cotton80 --split train --aug pkb --num-samples 6 --out-dir outputs/aug_demo_pkb
```

Available augmentations:

- `pkb` (PatchKeepBlur)
- `cutout` (PatchCutout)
- `fullblur` (Full image Gaussian blur)
- `multiview-pkb` (Multiple distinct PKB views in one row)

Key options:

- `--n` grid side (n x n)
- `--a` number of blurred (or cutout) patches (overrides fraction)
- `--a-fraction` fraction of patches affected when `--a` not given
- `--sigma` Gaussian blur sigma
- `--placement` random | dispersed | contiguous
- `--views` number of views for multiview-pkb
- `--indices` explicit comma list of dataset indices (e.g. `--indices 0,12,55`)
- `--no-random` use first N samples instead of random

Outputs: PNG files and a small markdown gallery in the specified output directory.

```pwsh
python demo_augmentations.py --dataset soybean --split train --aug multiview-pkb --views 4 --num-samples 4 --placement dispersed
```

Images are named: `sample_{rank}_idx{original_index}_cls_{class_name}.png`.
