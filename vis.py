import argparse
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # ensure no GUI needed
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    import umap  # type: ignore
except ImportError:  # graceful fallback
    umap = None

import timm

from src.dataset.ufgvc import UFGVCDataset

THEME_PRIMARY = '#e30918'  # digital red
THEME_BLACK = '#000000'
THEME_WHITE = '#FFFFFF'


def build_val_transform(resize_side: int, crop: int):
    return transforms.Compose([
        transforms.Resize((resize_side, resize_side)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(args, num_classes: int):
    if args.model == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
    elif args.model == 'vit':
        model = timm.create_model('vit_small_patch16_384', pretrained=False, img_size=args.train_crop, num_classes=num_classes)
    else:
        model = timm.create_model(args.model, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def collect_indices_by_class(labels: List[int], k: int) -> List[int]:
    if k <= 0:
        return list(range(len(labels)))
    selected = []
    per_class_counter: Dict[int, int] = {}
    for idx, y in enumerate(labels):
        c = int(y)
        per_class_counter.setdefault(c, 0)
        if per_class_counter[c] < k:
            selected.append(idx)
            per_class_counter[c] += 1
    return selected


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device, layer: str = 'features', max_samples: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []

    # Hook for head input if needed
    captured: List[torch.Tensor] = []
    hook_handle = None

    if layer == 'head_input':
        # Try to find classifier module
        classifier = model.get_classifier() if hasattr(model, 'get_classifier') else None
        if classifier is None:
            # fallback: last module attribute named 'fc' or 'head'
            for name in ['fc', 'head', 'classifier']:
                if hasattr(model, name):
                    classifier = getattr(model, name)
                    break
        if classifier is None:
            raise ValueError('Cannot automatically locate classifier for head_input capture.')

        def hook_fn(m, inp, out):
            captured.append(inp[0].detach())

        hook_handle = classifier.register_forward_hook(hook_fn)

    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.numpy()
        if layer == 'features':
            if hasattr(model, 'forward_features'):
                f = model.forward_features(images)
                # For ViT-like, forward_features may return (B, tokens, C)
                if f.dim() == 3:  # (B, N, C) -> take CLS token
                    f = f[:, 0]
                elif f.dim() == 4:  # (B, C, H, W) -> global pool
                    f = f.mean(dim=[2, 3])
            else:
                # fallback: use penultimate output via hook? Simpler: use logits pre-softmax
                f = model(images)
            feats.append(f.cpu())
        else:  # head_input
            _ = model(images)  # triggers hook
            if not captured:
                continue
            f = captured[-1]
            feats.append(f.cpu())
        labels.append(torch.from_numpy(targets))
        count += images.size(0)
        if max_samples > 0 and count >= max_samples:
            break
    if hook_handle is not None:
        hook_handle.remove()
    feats_t = torch.cat(feats, dim=0)
    labels_t = torch.cat(labels, dim=0)
    return feats_t.numpy(), labels_t.numpy()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def apply_pca(x: np.ndarray, dim: int) -> np.ndarray:
    if dim <= 0 or x.shape[1] <= dim:
        return x
    pca = PCA(n_components=dim, random_state=0)
    return pca.fit_transform(x)


def embed_tsne(x: np.ndarray, seed: int, perplexity: Optional[float] = None) -> np.ndarray:
    n = x.shape[0]
    if perplexity is None:
        perplexity = min(30, max(5, n // 3))
    tsne = TSNE(n_components=2, init='pca', random_state=seed, perplexity=perplexity, learning_rate='auto')
    return tsne.fit_transform(x)


def embed_umap(x: np.ndarray, seed: int) -> np.ndarray:
    if umap is None:
        raise ImportError('umap-learn is not installed. Please add it to requirements.')
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(x)


def plot_embedding(emb: np.ndarray, labels: np.ndarray, title: str, out_path: Path, classes: List[str]):
    plt.figure(figsize=(10, 10), facecolor=THEME_BLACK)
    ax = plt.gca()
    ax.set_facecolor(THEME_BLACK)
    unique = np.unique(labels)
    # Build a discrete color map leaning on red; cycle if many classes
    base_cmap = plt.get_cmap('tab20')
    colors = []
    for i, _ in enumerate(unique):
        if i == 0:
            colors.append(THEME_PRIMARY)
        else:
            colors.append(base_cmap(i % base_cmap.N))
    for i, cls in enumerate(unique):
        pts = emb[labels == cls]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, color=colors[i], alpha=0.85, label=str(classes[cls]))
    ax.set_title(title, color=THEME_WHITE, fontsize=26, pad=20, weight='bold')
    ax.tick_params(colors=THEME_WHITE, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME_PRIMARY)
    legend = ax.legend(fontsize=12, facecolor=THEME_BLACK, edgecolor=THEME_PRIMARY, framealpha=0.9, loc='best')
    for text in legend.get_texts():
        text.set_color(THEME_WHITE)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=plt.gcf().get_facecolor())
    plt.close()


def k_mid_selection(features: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError('k must be > 0 for k-mid')
    # Using KMeans centroids and then nearest samples to centroids as representatives
    km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
    km.fit(features)
    centers = km.cluster_centers_
    # find representative index per cluster
    reps = []
    for i in range(k):
        idxs = np.where(km.labels_ == i)[0]
        cluster_feats = features[idxs]
        dists = np.linalg.norm(cluster_feats - centers[i], axis=1)
        rep_local = idxs[np.argmin(dists)]
        reps.append(rep_local)
    return centers, np.array(reps, dtype=int)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._forward_hook)
        self._bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove(); self._bwd.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        loss = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        if self.activations is None or self.gradients is None:
            raise RuntimeError('GradCAM hooks did not capture tensors.')
        # Grad-CAM weight: global-average pooling over spatial dims
        grads = self.gradients  # (B, C, H, W)
        acts = self.activations
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        # normalize each CAM to 0-1
        cam = cam - cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam = cam / (cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-6)
        return cam  # (B,1,H,W)


def overlay_cam(image: torch.Tensor, cam: torch.Tensor) -> np.ndarray:
    # image: (C,H,W) normalized; convert to uint8
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    heatmap = cam.squeeze().cpu().numpy()
    heatmap = plt.get_cmap('jet')(heatmap)[:, :, :3]
    overlay = 0.5 * img + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def auto_select_conv_layer(model: nn.Module) -> nn.Module:
    # pick the last Conv2d encountered
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError('No Conv2d layer found for Grad-CAM (likely a ViT). Specify a layer or use a CNN.')
    return last


def run_gradcam(args, model: nn.Module, device: torch.device, dataset: UFGVCDataset, out_dir: Path):
    if args.gradcam_samples <= 0:
        return
    if hasattr(model, 'module'):
        model_core = model.module
    else:
        model_core = model
    layer = auto_select_conv_layer(model_core) if args.gradcam_layer == 'auto' else eval(f'model_core.{args.gradcam_layer}')
    cam_engine = GradCAM(model_core, layer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    ensure_dir(out_dir)
    saved = 0
    for img, label in loader:
        img = img.to(device)
        cam = cam_engine(img)
        overlay = overlay_cam(img[0], cam[0])
        out_path = out_dir / f'gradcam_{saved:03d}_class{int(label)}.png'
        from imageio import imwrite  # lightweight dependency via imageio (not in req) -> fallback to plt
        try:
            imwrite(out_path, overlay)
        except Exception:
            plt.figure(figsize=(4, 4))
            plt.imshow(overlay)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
        saved += 1
        if saved >= args.gradcam_samples:
            break
    cam_engine.remove()


@torch.no_grad()
def run_vit_attention(args, model: nn.Module, device: torch.device, dataset: UFGVCDataset, out_dir: Path):
    """Generate attention rollout heatmaps for ViT-like models.
    Saves overlays analogous to Grad-CAM using attention rollout from CLS token.
    """
    if args.vit_attn_samples <= 0:
        return
    # Heuristic: require modules named 'blocks' (timm ViT) with attention submodules having 'attn_drop'
    if hasattr(model, 'module'):
        model_core = model.module
    else:
        model_core = model
    if not hasattr(model_core, 'blocks'):
        print('ViT attention requested but model has no attribute blocks; skipping.')
        return
    blocks = list(model_core.blocks)
    attn_hooks = []

    def make_hook(store_list):
        def _hook(module, inp, out):  # out: (B, heads, T, T)
            store_list.append(out.detach().cpu())
        return _hook

    ensure_dir(out_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    saved = 0
    for img, label in loader:
        attn_mats = []
        # register hooks for this sample
        for b in blocks:
            if hasattr(b, 'attn') and hasattr(b.attn, 'attn_drop'):
                attn_hooks.append(b.attn.attn_drop.register_forward_hook(make_hook(attn_mats)))
        img = img.to(device)
        _ = model_core(img)
        # remove hooks
        for h in attn_hooks:
            h.remove()
        attn_hooks.clear()
        if not attn_mats:
            print('No attention matrices captured; aborting ViT attention generation.')
            break
        # attention rollout
        # Order of attn_mats corresponds to forward order
        rollout = None
        for A in attn_mats:  # A: (1, heads, T, T)
            A_mean = A.mean(1)  # (1,T,T)
            I = torch.eye(A_mean.size(-1)).unsqueeze(0)
            A_aug = A_mean + I
            A_aug = A_aug / A_aug.sum(dim=-1, keepdim=True)
            if rollout is None:
                rollout = A_aug
            else:
                rollout = A_aug @ rollout  # matrix multiply
        # CLS attention to patches
        # rollout shape (1,T,T); take CLS row (index 0) excluding CLS token itself
        cls_attn = rollout[0, 0, 1:]  # (T-1,)
        tokens = cls_attn.shape[0]
        side = int(tokens ** 0.5)
        if side * side != tokens:
            # fallback reshape by trying nearest square; pad if necessary
            side = int(np.ceil(tokens ** 0.5))
            pad_len = side * side - tokens
            cls_attn = torch.cat([cls_attn, cls_attn.new_zeros(pad_len)], dim=0)
        heat = cls_attn.view(1, 1, side, side)
        heat = heat / (heat.max() + 1e-6)
        # upscale to image size
        heat_up = F.interpolate(heat, size=img.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        heat_up = heat_up.cpu().numpy()
        # Overlay (use custom red colormap)
        overlay = vit_overlay(img[0], heat_up)
        out_path = out_dir / f'vit_attn_{saved:03d}_class{int(label)}.png'
        try:
            from imageio import imwrite
            imwrite(out_path, overlay)
        except Exception:
            plt.figure(figsize=(4, 4))
            plt.imshow(overlay)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
        saved += 1
        if saved >= args.vit_attn_samples:
            break


def vit_overlay(image: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    """Overlay attention heatmap on original normalized image using theme colors.
    Emphasize primary red; convert to uint8 array.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    # Create red heatmap: scale heatmap to 0..1 then map to (R,G,B)
    h = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    red_map = np.stack([h, np.zeros_like(h), np.zeros_like(h)], axis=-1)
    # Blend: where heat high, push towards red; else keep original
    overlay = img * (1 - 0.6 * h[..., None]) + red_map * 0.6
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    # model/data
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', default='cotton80')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--model', default='resnet50')
    p.add_argument('--resize-side', type=int, default=440)
    p.add_argument('--train-crop', type=int, default=384)
    p.add_argument('--split', default='test')
    # feature extraction
    p.add_argument('--feature-layer', choices=['features', 'head_input'], default='features')
    p.add_argument('--max-samples', type=int, default=-1, help='Max total samples (after class sampling). -1=all')
    p.add_argument('--sample-per-class', type=int, default=-1)
    # embeddings
    p.add_argument('--do-tsne', action='store_true')
    p.add_argument('--do-umap', action='store_true')
    p.add_argument('--do-pca', action='store_true')
    p.add_argument('--pca-dim', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    # k-mid (representative selection via k-means)
    p.add_argument('--do-kmid', action='store_true')
    p.add_argument('--kmid-k', type=int, default=16)
    # grad-cam
    p.add_argument('--do-gradcam', action='store_true')
    p.add_argument('--gradcam-layer', default='auto')
    p.add_argument('--gradcam-samples', type=int, default=8)
    # ViT attention maps
    p.add_argument('--do-vit-attn', action='store_true')
    p.add_argument('--vit-attn-samples', type=int, default=8)
    # runtime
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default='./outputs/vis')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # dataset
    val_tf = build_val_transform(args.resize_side, args.train_crop)
    dataset = UFGVCDataset(dataset_name=args.dataset, root=args.data_root, split=args.split, transform=val_tf)
    num_classes = len(dataset.classes)
    model = load_model(args, num_classes).to(device)

    # DataLoader (full then subselect indices if needed)
    base_indices = list(range(len(dataset)))
    if args.sample_per_class > 0:
        # build quick labels list
        labels_list = [dataset[i][1] for i in range(len(dataset))]
        base_indices = collect_indices_by_class(labels_list, args.sample_per_class)
    if args.max_samples > 0 and len(base_indices) > args.max_samples:
        base_indices = base_indices[:args.max_samples]

    subset = torch.utils.data.Subset(dataset, base_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Extract features once if any embedding or kmid requested
    features = None
    labels = None
    if any([args.do_tsne, args.do_umap, args.do_pca, args.do_kmid]):
        features, labels = extract_features(model, loader, device, layer=args.feature_layer, max_samples=-1)
        # optional PCA pre-processing (except pure PCA visualization which will be handled later)
        if args.pca_dim > 0 and (args.do_tsne or args.do_umap):
            features_pca = apply_pca(features, args.pca_dim)
        else:
            features_pca = features
        out_dir = Path(args.out_dir)
        ensure_dir(out_dir)
        meta = {'checkpoint': args.checkpoint, 'dataset': args.dataset, 'split': args.split, 'feature_layer': args.feature_layer,
                'num_features': features.shape[1], 'num_samples': int(features.shape[0])}
        with open(out_dir / 'meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        # t-SNE
        if args.do_tsne:
            emb = embed_tsne(features_pca, seed=args.seed)
            np.save(out_dir / 'tsne.npy', emb)
            plot_embedding(emb, labels, 't-SNE Embedding', out_dir / 'tsne.png', dataset.classes)
        # UMAP
        if args.do_umap:
            if umap is None:
                print('UMAP requested but umap-learn not installed.')
            else:
                emb = embed_umap(features_pca, seed=args.seed)
                np.save(out_dir / 'umap.npy', emb)
                plot_embedding(emb, labels, 'UMAP Embedding', out_dir / 'umap.png', dataset.classes)
        # PCA direct plot
        if args.do_pca:
            pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(features)
            np.save(out_dir / 'pca2.npy', pca2)
            plot_embedding(pca2, labels, 'PCA (2D)', out_dir / 'pca2.png', dataset.classes)
        # k-mid
        if args.do_kmid:
            centers, reps = k_mid_selection(features, args.kmid_k, args.seed)
            np.save(out_dir / 'kmid_centers.npy', centers)
            np.save(out_dir / 'kmid_indices.npy', reps)
            with open(out_dir / 'kmid_indices.txt', 'w') as f:
                for r in reps:
                    f.write(str(int(r)) + '\n')

    # Grad-CAM (works best for CNNs). Use original full dataset subset if earlier subset applied.
    if args.do_gradcam:
        gradcam_dir = Path(args.out_dir) / 'gradcam'
        run_gradcam(args, model, device, dataset, gradcam_dir)
    if args.do_vit_attn:
        vit_dir = Path(args.out_dir) / 'vit_attn'
        run_vit_attention(args, model, device, dataset, vit_dir)

    print('Visualization tasks complete.')


if __name__ == '__main__':
    main()
