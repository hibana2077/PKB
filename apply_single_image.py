#!/usr/bin/env python3
"""
Apply a single augmentation from src/augmentations/pkb.py to a single image and save output.
Usage examples:
  pwsh> python .\scripts\apply_single_image.py -i images/samples/cotton.jpg -a 3 -n 4 --placement dispersed -o out.jpg --aug pkb
  pwsh> python .\scripts\apply_single_image.py -i images/samples/cotton.jpg --aug cutout --fill 255,0,0 -o out_cutout.jpg
"""
import argparse
import os
import sys
from typing import Optional, Tuple

from PIL import Image

# Ensure project root is on sys.path so we can import src.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.augmentations.pkb import PatchKeepBlur, PatchCutout, FullImageBlur, MultiViewPatchKeepBlur
except Exception as e:
    raise RuntimeError(f"Failed to import augmentations: {e}")


def parse_fill(value: str) -> Tuple[int, int, int]:
    parts = value.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('fill must be R,G,B')
    return tuple(int(p) for p in parts)


def maybe_int(v: Optional[str]) -> Optional[int]:
    return None if v is None else int(v)


def maybe_float(v: Optional[str]) -> Optional[float]:
    return None if v is None else float(v)


def main():
    p = argparse.ArgumentParser(description='Apply a PKB-style augmentation to a single image')
    p.add_argument('-i', '--image', required=True, help='Path to input image')
    p.add_argument('-o', '--out', default=None, help='Path to save output (if omitted, appends suffix)')
    p.add_argument('--aug', choices=['pkb', 'cutout', 'blur'], default='pkb', help='Augmentation to apply')
    p.add_argument('--views', type=int, default=1, help='Number of distinct PKB views to generate (only for pkb). If >1 uses MultiViewPatchKeepBlur')
    p.add_argument('-n', type=int, default=4, help='Grid n (n x n)')
    p.add_argument('-a', type=int, default=None, help='Number of patches to affect (overrides a_fraction)')
    p.add_argument('--a_fraction', type=float, default=None, help='Fraction of patches to affect (if -a omitted)')
    p.add_argument('--sigma', type=float, default=2.0, help='Gaussian blur sigma (for pkb and blur)')
    p.add_argument('--placement', choices=['random', 'dispersed', 'contiguous'], default='random', help='Patch placement')
    p.add_argument('--seed', type=int, default=None, help='RNG seed')
    p.add_argument('--fill', type=parse_fill, default=(127,127,127), help='Fill color for cutout as R,G,B')
    p.add_argument('--highlight', action='store_true', help='Draw a colored border around blurred patches (hex color default E30918)')
    p.add_argument('--highlight_color', default='E30918', help='Hex color for highlight (e.g. E30918 or #E30918)')
    p.add_argument('--show', action='store_true', help='Open saved image after processing (PIL.Image.show)')

    args = p.parse_args()

    if not os.path.exists(args.image):
        p.error(f'Input image not found: {args.image}')

    img = Image.open(args.image).convert('RGB')

    out_path = args.out
    if out_path is None:
        base, ext = os.path.splitext(args.image)
        out_path = f"{base}.{args.aug}{ext}"

    if args.aug == 'pkb':
        if args.views and args.views > 1:
            aug = MultiViewPatchKeepBlur(n=args.n, a=args.a, a_fraction=args.a_fraction, sigma=args.sigma, placement=args.placement, views=args.views, seed=args.seed, post_transform=None, highlight=args.highlight, highlight_color=args.highlight_color)
            res = aug(img)
        else:
            aug = PatchKeepBlur(n=args.n, a=args.a, a_fraction=args.a_fraction, sigma=args.sigma, placement=args.placement, seed=args.seed, highlight=args.highlight, highlight_color=args.highlight_color)
            res = aug(img)
    elif args.aug == 'cutout':
        aug = PatchCutout(n=args.n, a=args.a, a_fraction=args.a_fraction, placement=args.placement, fill=args.fill, seed=args.seed)
        res = aug(img)
    elif args.aug == 'blur':
        aug = FullImageBlur(sigma=args.sigma)
        res = aug(img)
    else:
        raise SystemExit('Unknown augmentation')

    # If augmentation returns a list (rare), save enumerated files
    if isinstance(res, list):
        saved_paths = []
        for idx, v in enumerate(res):
            if not hasattr(v, 'save'):
                print(f"Skipping element {idx}: not a PIL.Image")
                continue
            pth = os.path.splitext(out_path)[0] + f".{idx}" + os.path.splitext(out_path)[1]
            v.save(pth)
            saved_paths.append(pth)
        print('Saved:', ', '.join(saved_paths))
        if args.show and saved_paths:
            Image.open(saved_paths[0]).show()
        return

    # If result is a PIL image-like, save
    if hasattr(res, 'save'):
        res.save(out_path)
        print(f'Saved: {out_path}')
        if args.show:
            res.show()
        return

    # Otherwise, unknown return type (e.g., torch.Tensor)
    try:
        # Try to handle torch tensor if available
        import torch
        if isinstance(res, torch.Tensor):
            # Expect (C,H,W) or (H,W,C) or (V,C,H,W)
            t = res
            if t.ndim == 4:
                # save each view
                for vi in range(t.shape[0]):
                    tv = t[vi]
                    if tv.shape[0] in (1,3):
                        tv = tv
                    else:
                        tv = tv.permute(2,0,1)
                    from torchvision.transforms.functional import to_pil_image
                    pil = to_pil_image(tv)
                    pth = os.path.splitext(out_path)[0] + f".{vi}" + os.path.splitext(out_path)[1]
                    pil.save(pth)
                print('Saved tensor views as images')
                return
            elif t.ndim == 3:
                if t.shape[0] in (1,3):
                    from torchvision.transforms.functional import to_pil_image
                    pil = to_pil_image(t)
                    pil.save(out_path)
                    print(f'Saved tensor as image: {out_path}')
                    return
    except Exception:
        pass

    raise RuntimeError(f'Cannot save augmentation output of type {type(res)}')


if __name__ == '__main__':
    main()
