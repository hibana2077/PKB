import argparse
from pathlib import Path
import random
from typing import Optional, List

import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset.ufgvc import UFGVCDataset
from src.augmentations.pkb import PatchKeepBlur, PatchCutout, FullImageBlur, MultiViewPatchKeepBlur


def build_base_transform(size: int):
    return transforms.Resize((size, size))


def parse_args():
    p = argparse.ArgumentParser(description='Demo augmentation before/after comparison')
    p.add_argument('--dataset', default='cotton80')
    p.add_argument('--split', default='train')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--out-dir', default='./outputs/aug_demo')
    p.add_argument('--num-samples', type=int, default=8, help='Number of samples to visualize')
    p.add_argument('--indices', type=str, default='', help='Comma separated explicit indices to use (overrides random).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--size', type=int, default=384, help='Resize shorter side (square) before aug')
    # augmentation choice
    p.add_argument('--aug', choices=['pkb', 'cutout', 'fullblur', 'multiview-pkb'], default='pkb')
    # shared PKB params
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--a', type=int, default=None)
    p.add_argument('--a-fraction', type=float, default=0.25)
    p.add_argument('--sigma', type=float, default=2.0)
    p.add_argument('--placement', choices=['random', 'dispersed', 'contiguous'], default='random')
    p.add_argument('--views', type=int, default=3, help='For multiview-pkb number of augmented views.')
    p.add_argument('--no-random', action='store_true', help='Use first N samples instead of random selection if --indices not provided.')
    return p.parse_args()


def build_augmentation(args):
    common_kwargs = dict(n=args.n, a=args.a, a_fraction=args.a_fraction, sigma=args.sigma, placement=args.placement, seed=args.seed)
    if args.aug == 'pkb':
        return PatchKeepBlur(**common_kwargs)
    if args.aug == 'cutout':
        return PatchCutout(n=args.n, a=args.a, a_fraction=args.a_fraction, placement=args.placement, seed=args.seed)
    if args.aug == 'fullblur':
        return FullImageBlur(sigma=args.sigma)
    if args.aug == 'multiview-pkb':
        # multiview expects post_transform to maybe convert to tensor; we only need PIL so leave None
        return MultiViewPatchKeepBlur(views=args.views, post_transform=None, **common_kwargs)
    raise ValueError('Unknown augmentation')


def select_indices(total: int, args) -> List[int]:
    if args.indices:
        idxs = [int(x) for x in args.indices.split(',') if x.strip()]
        return [i for i in idxs if 0 <= i < total]
    if args.no_random:
        return list(range(min(args.num_samples, total)))
    rng = random.Random(args.seed)
    return rng.sample(range(total), k=min(args.num_samples, total))


def pil_to_tensor_for_show(img):
    return transforms.ToTensor()(img)


def show_and_save(original, augmented_list, out_path: Path, title: str):
    cols = 1 + len(augmented_list)
    plt.figure(figsize=(3 * cols, 3))
    imgs = [original] + augmented_list
    for i, im in enumerate(imgs):
        ax = plt.subplot(1, cols, i + 1)
        ax.imshow(im)
        ax.axis('off')
        ax.set_title('orig' if i == 0 else f'aug{i}', fontsize=10)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset without transform so we can apply base transform before augmentation
    dataset_plain = UFGVCDataset(dataset_name=args.dataset, root=args.data_root, split=args.split, transform=None)
    base_tf = build_base_transform(args.size)
    aug = build_augmentation(args)

    indices = select_indices(len(dataset_plain), args)
    print(f'Using indices: {indices}')

    for rank, idx in enumerate(indices):
        img, label = dataset_plain[idx]
        cls_name = dataset_plain.get_class_name(idx)
        img_proc = base_tf(img)

        if args.aug == 'multiview-pkb':
            views = aug(img_proc)  # list of PIL images
            if torch.is_tensor(views):
                # unlikely since post_transform None
                raise ValueError('Expected PIL views; got tensor. Provide post_transform=None.')
            augmented_list = views if isinstance(views, list) else [views]
        else:
            augmented_list = [aug(img_proc)]

        # Convert to displayable arrays (PIL already ok)
        out_path = out_dir / f'sample_{rank:02d}_idx{idx}_cls_{cls_name}.png'
        show_and_save(img_proc, augmented_list, out_path, f'{args.aug} | cls={cls_name} idx={idx}')

    # simple markdown summary
    md_path = out_dir / 'README.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# Augmentation Demo ({args.aug})\n\n')
        f.write(f'Dataset: **{args.dataset}** split **{args.split}**\n\n')
        f.write('Samples generated:\n\n')
        for rank, idx in enumerate(indices):
            pattern = f'sample_{rank:02d}_idx{idx}_cls_'
            for file in out_dir.glob(pattern + '*'):
                if file.suffix.lower() == '.png':
                    f.write(f'![]({file.name})\n\n')
    print(f'Done. Images saved to {out_dir}')


if __name__ == '__main__':
    main()
