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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

try:
    import umap  # type: ignore
except ImportError:  # graceful fallback
    umap = None

import timm

from src.dataset.ufgvc import UFGVCDataset

THEME_PRIMARY = '#e30918'  # digital red
THEME_BLACK = '#000000'
THEME_WHITE = '#FFFFFF'


def _tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor (ImageNet mean/std) to HWC uint8 RGB."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    img = (image * std + mean).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _save_uint8_image(out_path: Path, array: np.ndarray):
    """Save HWC uint8 RGB image to disk with best-effort backends."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Prefer imageio if available
        from imageio import imwrite  # type: ignore
        imwrite(out_path, array)
    except Exception:
        # Fallback to matplotlib
        plt.figure(figsize=(4, 4))
        plt.imshow(array)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


def build_val_transform(resize_side: int, crop: int):
    return transforms.Compose([
        transforms.Resize((resize_side, resize_side)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model_with_params(model_name: str, checkpoint: str, train_crop: int, num_classes: int):
    """Create and load a timm model by name and checkpoint path.
    Falls back to generic timm.create_model if model_name not in special cases.
    """
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
    elif model_name in {'vit', 'vit_small_patch16_384'}:
        model = timm.create_model('vit_small_patch16_384', pretrained=False, img_size=train_crop, num_classes=num_classes)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(checkpoint, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_model(args, num_classes: int):
    return load_model_with_params(args.model, args.checkpoint, args.train_crop, num_classes)


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
    # Clamp n_components to valid range considering both samples and features
    if dim <= 0:
        return x
    n_samples, n_features = x.shape[0], x.shape[1]
    allowed_max = min(n_samples, n_features)
    n_components = min(dim, allowed_max)
    # If PCA won't reduce dimensionality, skip
    if n_components >= n_features:
        return x
    pca = PCA(n_components=n_components, random_state=0)
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


def plot_embedding(
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    classes: List[str],
    legend_fontsize: int = 12,
    marker_size: int = 18,
):
    # General plot: white background, black text, black frame; discrete colors per class
    plt.figure(figsize=(10, 10), facecolor=THEME_WHITE)
    ax = plt.gca()
    ax.set_facecolor(THEME_WHITE)
    unique = np.unique(labels)
    # Build a discrete color map leaning on red; cycle if many classes
    base_cmap = plt.get_cmap('tab20')
    colors = []
    for i, _ in enumerate(unique):
        colors.append(base_cmap(i % base_cmap.N))
    for i, cls in enumerate(unique):
        pts = emb[labels == cls]
        ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, color=colors[i], alpha=0.85, label=str(classes[cls]))
    ax.set_title(title, color=THEME_BLACK, fontsize=26, pad=20, weight='bold')
    ax.tick_params(colors=THEME_BLACK, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME_BLACK)
    ax.legend(
        fontsize=legend_fontsize,
        facecolor=THEME_WHITE,
        edgecolor=THEME_BLACK,
        framealpha=0.95,
        loc='best',
        labelcolor=THEME_BLACK,
        markerscale=1.2,
    )
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=plt.gcf().get_facecolor())
    plt.close()


def plot_embedding_tsne_gradient(emb: np.ndarray, title: str, out_path: Path, theme_hex: str = THEME_PRIMARY):
    """t-SNE plot with theme-colored gradient dots on white background.
    Color encodes a simple scalar from the 2D coords to avoid repeated discrete colors.
    """
    plt.figure(figsize=(10, 10), facecolor=THEME_WHITE)
    ax = plt.gca()
    ax.set_facecolor(THEME_WHITE)
    # Build a scalar for color mapping from coordinates (normalized x+y)/2
    x = emb[:, 0]
    y = emb[:, 1]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-6)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-6)
    s_val = 0.5 * (x_n + y_n)
    # Create a light-to-theme colormap
    light_rgb = (1.0, 0.92, 0.92)  # very light red-ish
    cmap = LinearSegmentedColormap.from_list('theme_grad', [light_rgb, theme_hex])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=s_val, cmap=cmap, s=18, alpha=0.9, edgecolors='none')
    ax.set_title(title, color=THEME_BLACK, fontsize=26, pad=20, weight='bold')
    ax.tick_params(colors=THEME_BLACK, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME_BLACK)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=plt.gcf().get_facecolor())
    plt.close()


def plot_embedding_tsne_gradient_with_class_legend(
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    classes: List[str],
    legend_fontsize: int = 20,
    marker_size: int = 28,
    theme_hex: str = THEME_PRIMARY,
):
    """t-SNE plot:
    - Dot color: theme gradient based on 2D coords (same scalar mapping for all points)
    - Class legend: distinguished by marker shape (not color)
    - Larger markers for readability
    """
    plt.figure(figsize=(10, 10), facecolor=THEME_WHITE)
    ax = plt.gca()
    ax.set_facecolor(THEME_WHITE)

    x = emb[:, 0]
    y = emb[:, 1]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-6)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-6)
    s_val = 0.5 * (x_n + y_n)

    # Light-to-theme colormap
    light_rgb = (1.0, 0.92, 0.92)
    cmap = LinearSegmentedColormap.from_list('theme_grad', [light_rgb, theme_hex])

    unique = np.unique(labels)
    # Cycle through marker shapes to represent classes
    marker_cycle = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*', 'h', 'H']

    for i, cls in enumerate(unique):
        pts = emb[labels == cls]
        cls_mask = (labels == cls)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=s_val[cls_mask],
            cmap=cmap,
            s=marker_size,
            alpha=0.9,
            edgecolors='none',
            marker=marker_cycle[i % len(marker_cycle)],
            label=str(classes[cls]),
        )

    ax.set_title(title, color=THEME_BLACK, fontsize=26, pad=20, weight='bold')
    ax.tick_params(colors=THEME_BLACK, labelsize=16)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME_BLACK)

    # Build proxy legend handles to reflect marker shapes with theme color
    from matplotlib.lines import Line2D
    handles = []
    for i, cls in enumerate(unique):
        handle = Line2D(
            [], [],
            linestyle='None',
            marker=marker_cycle[i % len(marker_cycle)],
            markerfacecolor=theme_hex,
            markeredgecolor=THEME_BLACK,
            markersize=max(8, marker_size // 2),
            label=str(classes[cls]),
        )
        handles.append(handle)
    ax.legend(
        handles=handles,
        fontsize=legend_fontsize,
        facecolor=THEME_WHITE,
        edgecolor=THEME_BLACK,
        framealpha=0.95,
        loc='best',
        labelcolor=THEME_BLACK,
    )
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
    # image: (C,H,W) normalized; cam: (Hc,Wc) or (1,Hc,Wc)
    # 1) Unnormalize image to [0,1] for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = (image.cpu() * std + mean).clamp(0, 1)
    H, W = img_t.shape[-2:]
    img = img_t.permute(1, 2, 0).numpy()  # (H,W,3)

    # 2) Upsample CAM to image size and normalize to [0,1]
    if cam.dim() == 2:
        cam_4d = cam.unsqueeze(0).unsqueeze(0)  # (1,1,Hc,Wc)
    elif cam.dim() == 3:  # (1,Hc,Wc)
        cam_4d = cam.unsqueeze(0)  # (1,1,Hc,Wc)
    else:
        raise ValueError(f"Unexpected CAM dims: {tuple(cam.shape)}")
    cam_up = F.interpolate(cam_4d.float(), size=(H, W), mode='bilinear', align_corners=False).squeeze()
    # Ensure 2D array
    if cam_up.dim() == 3:
        cam_up = cam_up[0]
    cam_up = cam_up - cam_up.min()
    cam_up = cam_up / (cam_up.max() + 1e-6)
    heatmap = plt.get_cmap('jet')(cam_up.cpu().numpy())[:, :, :3]  # (H,W,3) in [0,1]

    # 3) Blend heatmap with image
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


def run_gradcam(
    args,
    model: nn.Module,
    device: torch.device,
    dataset: UFGVCDataset,
    out_dir: Path,
    indices: Optional[List[int]] = None,
) -> int:
    if args.gradcam_samples <= 0:
        return 0
    if hasattr(model, 'module'):
        model_core = model.module
    else:
        model_core = model
    layer = auto_select_conv_layer(model_core) if args.gradcam_layer == 'auto' else eval(f'model_core.{args.gradcam_layer}')
    cam_engine = GradCAM(model_core, layer)
    if indices is not None:
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    ensure_dir(out_dir)
    orig_dir = out_dir / 'original'
    if getattr(args, 'orig_pic', False):
        ensure_dir(orig_dir)
    saved = 0
    for img, label in loader:
        img = img.to(device)
        cam = cam_engine(img)
        overlay = overlay_cam(img[0], cam[0])
        # Support tensor labels with batch_size=1
        try:
            label_int = int(label.item() if torch.is_tensor(label) else label)
        except Exception:
            label_int = -1
        out_path = out_dir / f'gradcam_{saved:03d}_class{label_int}.png'
        # Save original image if requested
        if getattr(args, 'orig_pic', False):
            orig_img = _tensor_to_uint8_image(img[0])
            orig_path = orig_dir / f'gradcam_{saved:03d}_class{label_int}.png'
            _save_uint8_image(orig_path, orig_img)
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
        # If indices provided, iterate through all; otherwise respect --gradcam-samples
        if indices is None and saved >= args.gradcam_samples:
            break
    cam_engine.remove()
    return saved


@torch.no_grad()
def run_vit_attention(
    args,
    model: nn.Module,
    device: torch.device,
    dataset: UFGVCDataset,
    out_dir: Path,
    indices: Optional[List[int]] = None,
) -> int:
    """Generate attention rollout heatmaps for ViT-like models.
    Saves overlays analogous to Grad-CAM using attention rollout from CLS token.
    """
    if args.vit_attn_samples <= 0:
        return 0
    # Support both timm ViT (has `blocks`) and TinyViT (has `stages` with TinyVitBlock).
    if hasattr(model, 'module'):
        model_core = model.module
    else:
        model_core = model

    attn_hooks = []

    def make_hook(store_list):
        def _hook(module, inp, out):  # out: (B, heads, T, T)
            store_list.append(out.detach().cpu())
        return _hook

    ensure_dir(out_dir)
    orig_dir = out_dir / 'original'
    if getattr(args, 'orig_pic', False):
        ensure_dir(orig_dir)
    if indices is not None:
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    saved = 0

    # Case A: Standard ViT with global attention and cls token
    if hasattr(model_core, 'blocks'):
        blocks = list(model_core.blocks)
        for img, label in loader:
            attn_mats = []
            for b in blocks:
                if hasattr(b, 'attn') and hasattr(b.attn, 'attn_drop'):
                    attn_hooks.append(b.attn.attn_drop.register_forward_hook(make_hook(attn_mats)))
            img = img.to(device)
            _ = model_core(img)
            for h in attn_hooks:
                h.remove()
            attn_hooks.clear()
            if not attn_mats:
                print('No attention matrices captured; aborting ViT attention generation.')
                break
            rollout = None
            for A in attn_mats:  # A: (1, heads, T, T)
                A_mean = A.mean(1)  # (1,T,T)
                I = torch.eye(A_mean.size(-1)).unsqueeze(0)
                A_aug = A_mean + I
                A_aug = A_aug / A_aug.sum(dim=-1, keepdim=True)
                rollout = A_aug if rollout is None else A_aug @ rollout
            cls_attn = rollout[0, 0, 1:]
            tokens = cls_attn.shape[0]
            side = int(tokens ** 0.5)
            if side * side != tokens:
                side = int(np.ceil(tokens ** 0.5))
                pad_len = side * side - tokens
                cls_attn = torch.cat([cls_attn, cls_attn.new_zeros(pad_len)], dim=0)
            heat = cls_attn.view(1, 1, side, side)
            heat = heat / (heat.max() + 1e-6)
            heat_up = F.interpolate(heat, size=img.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            heat_up = heat_up.cpu().numpy()
            overlay = vit_overlay(img[0], heat_up)
            try:
                label_int = int(label.item() if torch.is_tensor(label) else label)
            except Exception:
                label_int = -1
            out_path = out_dir / f'vit_attn_{saved:03d}_class{label_int}.png'
            if getattr(args, 'orig_pic', False):
                orig_img = _tensor_to_uint8_image(img[0])
                orig_path = orig_dir / f'vit_attn_{saved:03d}_class{label_int}.png'
                _save_uint8_image(orig_path, orig_img)
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
            if indices is None and saved >= args.vit_attn_samples:
                break

    # Case B: TinyViT style with window attention under `stages`
    elif hasattr(model_core, 'stages'):
        # choose the last block of the last stage as representative
        stages = list(model_core.stages)
        if not stages:
            print('Model has stages but empty; skipping vit attention.')
            return
        last_stage = stages[-1]
        if not hasattr(last_stage, 'blocks') or len(last_stage.blocks) == 0:
            print('Last stage has no blocks; skipping vit attention.')
            return
        target_block = last_stage.blocks[-1]
        if not hasattr(target_block, 'attn') or not hasattr(target_block.attn, 'qkv'):
            print('TinyViT block without attn.qkv; cannot capture attention.')
            return
        window_size = getattr(target_block, 'window_size', None)
        if window_size is None:
            print('TinyViT window_size not found; cannot map windows.')
            return

        qkv_store = {}
        feat_shape = {}

        def qkv_hook(m, inp, out):
            # out: (B*nW, tokens, 3*dim)
            qkv_store['out'] = out.detach().cpu()

        def feat_hook(m, inp, out):
            # capture spatial size at block input (B, C, H, W)
            x = inp[0]
            if x.dim() == 4:
                feat_shape['HW'] = (int(x.shape[-2]), int(x.shape[-1]))

        # hook qkv and a conv to get H,W
        h1 = target_block.attn.qkv.register_forward_hook(qkv_hook)
        # prefer local_conv input for HW; fallback to downsample or stage conv
        local_conv = getattr(target_block, 'local_conv', None)
        if local_conv is not None and hasattr(local_conv, 'conv'):
            h2 = local_conv.conv.register_forward_hook(feat_hook)
        else:
            # generic: hook the block itself to get (B,C,H,W) if possible
            h2 = target_block.register_forward_hook(feat_hook)

        for img, label in loader:
            qkv_store.clear(); feat_shape.clear()
            img = img.to(device)
            _ = model_core(img)
            if 'out' not in qkv_store or 'HW' not in feat_shape:
                print('Failed to capture TinyViT qkv or feature shape; abort.')
                break

            qkv = qkv_store['out']  # cpu tensor
            Hs, Ws = feat_shape['HW']
            ws = int(window_size)
            if Hs % ws != 0 or Ws % ws != 0:
                print(f'Feature size {(Hs, Ws)} not divisible by window_size {ws}; using fallback resize.')
            B = 1  # loader uses batch_size=1
            nW = qkv.shape[0] // B
            tokens = qkv.shape[1]
            dim3 = qkv.shape[2]
            dim = dim3 // 3
            # infer heads
            num_heads = getattr(target_block.attn, 'num_heads', 1)
            head_dim = dim // num_heads
            # compute attention per window
            qkv = qkv.view(B, nW, tokens, 3, num_heads, head_dim).permute(0, 1, 3, 4, 2, 5).contiguous()
            # qkv shape: (B, nW, 3, heads, tokens, head_dim)
            q = qkv[:, :, 0]
            k = qkv[:, :, 1]
            # attn: (B, nW, heads, tokens, tokens)
            attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            attn_mean = attn.mean(dim=2)  # (B, nW, T, T)
            # token importance as mean over queries -> (B, nW, T)
            token_imp = attn_mean.mean(dim=2)
            # fold windows back to (H_s, W_s)
            if Hs % ws == 0 and Ws % ws == 0 and (nW == (Hs // ws) * (Ws // ws)) and (tokens == ws * ws):
                nWh = Hs // ws
                nWw = Ws // ws
                grid = token_imp.view(B, nWh, nWw, ws, ws)
                heat = grid.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, Hs, Ws)
            else:
                # fallback: assume square windows layout
                nWh = int(np.sqrt(nW))
                nWw = max(1, nW // max(1, nWh))
                # pad if needed
                need = nWh * nWw - nW
                if need > 0:
                    pad = torch.zeros(B, need, tokens, dtype=token_imp.dtype)
                    token_imp = torch.cat([token_imp, pad], dim=1)
                # reshape and tile windows
                grid = token_imp.view(B, nWh, nWw, int(np.sqrt(tokens)), int(np.sqrt(tokens)))
                Hs2 = nWh * grid.shape[3]
                Ws2 = nWw * grid.shape[4]
                heat = grid.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, Hs2, Ws2)
                # resize to approximate stage size
                heat = F.interpolate(heat, size=(Hs, Ws), mode='bilinear', align_corners=False)

            # normalize and upsample to image
            heat = heat / (heat.amax(dim=[2, 3], keepdim=True) + 1e-6)
            heat_up = F.interpolate(heat, size=img.shape[-2:], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            overlay = vit_overlay(img[0], heat_up)
            try:
                label_int = int(label.item() if torch.is_tensor(label) else label)
            except Exception:
                label_int = -1
            out_path = out_dir / f'vit_attn_{saved:03d}_class{label_int}.png'
            if getattr(args, 'orig_pic', False):
                orig_img = _tensor_to_uint8_image(img[0])
                orig_path = orig_dir / f'vit_attn_{saved:03d}_class{label_int}.png'
                _save_uint8_image(orig_path, orig_img)
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
            if indices is None and saved >= args.vit_attn_samples:
                break

        h1.remove(); h2.remove()
    else:
        print('Model is not a ViT/TinyViT style (no blocks or stages); skipping vit attention.')
    return saved


def vit_overlay(image: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    """Overlay attention heatmap on original normalized image using a standard colormap (jet).
    Returns an HWC uint8 image.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    h = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    cmap = plt.get_cmap('jet')
    heat_rgb = cmap(h)[..., :3]  # [0,1]
    overlay = 0.5 * img + 0.5 * heat_rgb
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ---- Embedding quality metrics and plots ----
def _safe_silhouette(emb2d: np.ndarray, labels: np.ndarray) -> Optional[float]:
    try:
        # silhouette requires at least 2 clusters and fewer than n_samples clusters
        if len(np.unique(labels)) < 2 or emb2d.shape[0] < 3:
            return None
        return float(silhouette_score(emb2d, labels, metric='euclidean'))
    except Exception:
        return None


def _knn_overall_accuracy(emb2d: np.ndarray, labels: np.ndarray, k: int = 10) -> Optional[float]:
    n = emb2d.shape[0]
    if n < 2:
        return None
    k_eff = max(1, min(k, n - 1))
    try:
        nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean')
        nn.fit(emb2d)
        dists, indices = nn.kneighbors(emb2d)  # includes self at [:,0]
        neigh_idx = indices[:, 1:]  # drop self
        neigh_labels = labels[neigh_idx]
        # majority vote
        from scipy import stats
        mode_labels = stats.mode(neigh_labels, axis=1, keepdims=False).mode
        acc = (mode_labels == labels).mean()
        return float(acc)
    except Exception:
        # fallback simple L2, no scipy
        try:
            from collections import Counter
            acc_cnt = 0
            for i in range(n):
                diff = emb2d - emb2d[i]
                dist = (diff[:, 0]**2 + diff[:, 1]**2)
                order = np.argsort(dist)
                neigh = order[1:1 + k_eff]
                counts = Counter(labels[neigh].tolist())
                pred = max(counts.items(), key=lambda x: x[1])[0]
                if pred == labels[i]:
                    acc_cnt += 1
            return acc_cnt / n
        except Exception:
            return None


def _plot_knn_overall(acc: Optional[float], title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar([0], [acc if acc is not None else 0.0], color='#1f77b4')
    plt.xticks([0], ['kNN@10'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title(title + ('' if acc is None else f' ({acc:.3f})'))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_knn_compare(acc_base: Optional[float], acc_pkb: Optional[float], title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    vals = [acc_base if acc_base is not None else 0.0, acc_pkb if acc_pkb is not None else 0.0]
    colors = ['#1f77b4', '#ff7f0e']
    plt.bar([0, 1], vals, color=colors)
    plt.xticks([0, 1], ['Base', 'PKB'])
    plt.ylim(0, 1)
    plt.ylabel('kNN@10 Accuracy')
    sup = f" (base={vals[0]:.3f}, pkb={vals[1]:.3f})"
    plt.title(title + sup)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_embedding_compare(
    emb: np.ndarray,
    labels: np.ndarray,
    model_flags: np.ndarray,
    classes: List[str],
    title: str,
    out_path: Path,
    marker_size: int = 18,
):
    """Plot combined embedding for two models.
    model_flags: 0 for Base, 1 for PKB. Classes colored; models differ by marker.
    """
    plt.figure(figsize=(10, 10), facecolor=THEME_WHITE)
    ax = plt.gca()
    ax.set_facecolor(THEME_WHITE)
    unique_classes = np.unique(labels)
    base_cmap = plt.get_cmap('tab20')
    class_colors = {c: base_cmap(i % base_cmap.N) for i, c in enumerate(unique_classes)}
    markers = {0: 'o', 1: 's'}
    for c in unique_classes:
        idx_base = np.where((labels == c) & (model_flags == 0))[0]
        idx_pkb = np.where((labels == c) & (model_flags == 1))[0]
        if len(idx_base) > 0:
            pts = emb[idx_base]
            ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, color=class_colors[c], marker=markers[0], alpha=0.85, label=None)
        if len(idx_pkb) > 0:
            pts = emb[idx_pkb]
            ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, color=class_colors[c], marker=markers[1], alpha=0.85, label=None)
    # Legends
    from matplotlib.lines import Line2D
    class_handles = [Line2D([], [], linestyle='None', marker='o', color=class_colors[c], label=str(classes[c])) for c in unique_classes]
    model_handles = [Line2D([], [], linestyle='None', marker=markers[m], color='k', label=('Base' if m == 0 else 'PKB')) for m in [0, 1]]
    first_legend = ax.legend(handles=class_handles, title='Classes', fontsize=12, loc='upper right', framealpha=0.95)
    ax.add_artist(first_legend)
    ax.legend(handles=model_handles, title='Models', fontsize=12, loc='lower right', framealpha=0.95)
    ax.set_title(title, color=THEME_BLACK, fontsize=26, pad=20, weight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME_BLACK)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor=plt.gcf().get_facecolor())
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    # model/data
    p.add_argument('--checkpoint', default=None, help='Single-model checkpoint (required unless --compare-two-models)')
    p.add_argument('--checkpoint-base', default=None, help='Base model checkpoint (for two-model compare)')
    p.add_argument('--checkpoint-pkb', default=None, help='PKB model checkpoint (for two-model compare)')
    p.add_argument('--dataset', default='cotton80')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--model', default='resnet50')
    p.add_argument('--model-base', default=None, help='Base model name (defaults to --model)')
    p.add_argument('--model-pkb', default=None, help='PKB model name (defaults to --model)')
    p.add_argument('--resize-side', type=int, default=440)
    p.add_argument('--train-crop', type=int, default=384)
    p.add_argument('--split', default='test')
    p.add_argument('--first-n-classes', type=int, default=-1, help='Use only the first N classes (dataset.classes order). -1=all')
    p.add_argument('--compare-two-models', action='store_true', help='Compare Base vs PKB: visualize indices where PKB correct, Base wrong')
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
    # originals saving
    p.add_argument('--orig-pic', action='store_true', help='Save original input images alongside overlays')
    # runtime
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir', default='./outputs/vis')
    return p.parse_args()


@torch.no_grad()
def get_predictions(model: nn.Module, dataset: UFGVCDataset, indices: List[int], device: torch.device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a subset of dataset indices and return (preds, labels)."""
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    model = model.to(device)
    model.eval()
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        # targets may be tensor already
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)
        labels_list.append(targets.cpu())
        preds_list.append(preds)
    return torch.cat(preds_list).numpy(), torch.cat(labels_list).numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # dataset
    val_tf = build_val_transform(args.resize_side, args.train_crop)
    dataset = UFGVCDataset(dataset_name=args.dataset, root=args.data_root, split=args.split, transform=val_tf)
    num_classes = len(dataset.classes)

    # DataLoader (full then subselect indices if needed)
    base_indices = list(range(len(dataset)))
    # Limit to first-N classes (by dataset.classes order)
    if args.first_n_classes > 0:
        n = min(args.first_n_classes, len(dataset.classes))
        allowed = set(dataset.classes[:n])
        # Use internal dataframe to avoid loading images
        class_names_series = dataset.data['class_name'] if hasattr(dataset, 'data') else None
        if class_names_series is not None:
            base_indices = [i for i, cn in enumerate(class_names_series.tolist()) if cn in allowed]
        else:
            # Fallback (may load images): keep indices whose label maps to first n classes
            base_indices = [i for i in range(len(dataset)) if dataset[i][1] < n]

    if args.sample_per_class > 0:
        # Build labels list without loading images if possible
        if hasattr(dataset, 'data'):
            candidate_indices = base_indices
            class_names = dataset.data.iloc[candidate_indices]['class_name'].tolist()
            labels_list = [dataset.class_to_idx[cn] for cn in class_names]
            selected_rel = collect_indices_by_class(labels_list, args.sample_per_class)
            base_indices = [candidate_indices[i] for i in selected_rel]
        else:
            # Fallback (may load images)
            labels_list = [dataset[i][1] for i in base_indices]
            selected_rel = collect_indices_by_class(labels_list, args.sample_per_class)
            base_indices = [base_indices[i] for i in selected_rel]
    if args.max_samples > 0 and len(base_indices) > args.max_samples:
        base_indices = base_indices[:args.max_samples]

    subset = torch.utils.data.Subset(dataset, base_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Extract features once if any embedding or kmid requested (only for single-model mode)
    features = None
    labels = None
    if not args.compare_two_models and any([args.do_tsne, args.do_umap, args.do_pca, args.do_kmid]):
        if args.checkpoint is None:
            raise ValueError('Embeddings requested: please provide --checkpoint for single-model mode (omit --compare-two-models).')
        model_single = load_model(args, num_classes).to(device)
        features, labels = extract_features(model_single, loader, device, layer=args.feature_layer, max_samples=-1)
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
            sil = _safe_silhouette(emb, labels)
            title = 't-SNE Embedding' + ('' if sil is None else f' (sil={sil:.3f})')
            plot_embedding(emb, labels, title, out_dir / 'tsne.png', dataset.classes)
            # kNN metrics
            knn_acc = _knn_overall_accuracy(emb, labels, k=10)
            _plot_knn_overall(knn_acc, 't-SNE kNN accuracy', out_dir / 'tsne_knn.png')
            with open(out_dir / 'tsne_metrics.json', 'w', encoding='utf-8') as f:
                json.dump({'silhouette': sil, 'knn_acc@10': knn_acc}, f, indent=2)
        # UMAP
        if args.do_umap:
            if umap is None:
                print('UMAP requested but umap-learn not installed.')
            else:
                emb = embed_umap(features_pca, seed=args.seed)
                np.save(out_dir / 'umap.npy', emb)
                sil = _safe_silhouette(emb, labels)
                title = 'UMAP Embedding' + ('' if sil is None else f' (sil={sil:.3f})')
                plot_embedding(emb, labels, title, out_dir / 'umap.png', dataset.classes)
                knn_acc = _knn_overall_accuracy(emb, labels, k=10)
                _plot_knn_overall(knn_acc, 'UMAP kNN accuracy', out_dir / 'umap_knn.png')
                with open(out_dir / 'umap_metrics.json', 'w', encoding='utf-8') as f:
                    json.dump({'silhouette': sil, 'knn_acc@10': knn_acc}, f, indent=2)
        # PCA direct plot
        if args.do_pca:
            pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(features)
            np.save(out_dir / 'pca2.npy', pca2)
            sil = _safe_silhouette(pca2, labels)
            title = 'PCA (2D)' + ('' if sil is None else f' (sil={sil:.3f})')
            plot_embedding(pca2, labels, title, out_dir / 'pca2.png', dataset.classes)
            knn_acc = _knn_overall_accuracy(pca2, labels, k=10)
            _plot_knn_overall(knn_acc, 'PCA(2D) kNN accuracy', out_dir / 'pca2_knn.png')
            with open(out_dir / 'pca2_metrics.json', 'w', encoding='utf-8') as f:
                json.dump({'silhouette': sil, 'knn_acc@10': knn_acc}, f, indent=2)
        # k-mid
        if args.do_kmid:
            centers, reps = k_mid_selection(features, args.kmid_k, args.seed)
            np.save(out_dir / 'kmid_centers.npy', centers)
            np.save(out_dir / 'kmid_indices.npy', reps)
            with open(out_dir / 'kmid_indices.txt', 'w') as f:
                for r in reps:
                    f.write(str(int(r)) + '\n')

    # Two-model comparison: find samples PKB correct and Base wrong; then visualize those only
    if args.compare_two_models:
        # Validate inputs
        if not args.checkpoint_base or not args.checkpoint_pkb:
            raise ValueError('When --compare-two-models is set, please provide --checkpoint-base and --checkpoint-pkb')
        model_base_name = args.model_base if args.model_base is not None else args.model
        model_pkb_name = args.model_pkb if args.model_pkb is not None else args.model
        model_base = load_model_with_params(model_base_name, args.checkpoint_base, args.train_crop, num_classes).to(device)
        model_pkb = load_model_with_params(model_pkb_name, args.checkpoint_pkb, args.train_crop, num_classes).to(device)

        # Evaluate on the same candidate pool (base_indices)
        eval_indices = base_indices
        preds_base, labels_eval = get_predictions(model_base, dataset, eval_indices, device, args.batch_size)
        preds_pkb, _ = get_predictions(model_pkb, dataset, eval_indices, device, args.batch_size)
        correct_pkb = preds_pkb == labels_eval
        wrong_base = preds_base != labels_eval
        chosen_mask = np.logical_and(correct_pkb, wrong_base)
        chosen_indices = [eval_indices[i] for i, flag in enumerate(chosen_mask) if flag]

        # t-SNE comparison: select classes where PKB accuracy > Base accuracy
        if args.do_tsne:
            from collections import defaultdict
            class_total = defaultdict(int)
            class_correct_base = defaultdict(int)
            class_correct_pkb = defaultdict(int)
            for y_true, yb, yp in zip(labels_eval, preds_base, preds_pkb):
                c = int(y_true)
                class_total[c] += 1
                if yb == y_true:
                    class_correct_base[c] += 1
                if yp == y_true:
                    class_correct_pkb[c] += 1
            better_classes = []
            for c in class_total.keys():
                tb = class_total[c]
                if tb <= 0:
                    continue
                acc_b = class_correct_base[c] / tb
                acc_p = class_correct_pkb[c] / tb
                if acc_p > acc_b:
                    better_classes.append(c)
            if len(better_classes) == 0:
                print('t-SNE compare: no classes where PKB > Base; skipping t-SNE comparison.')
            else:
                better_set = set(better_classes)
                sel_indices = [idx for idx, y in zip(eval_indices, labels_eval) if int(y) in better_set]
                # Build loaders for selected indices
                subset_sel = torch.utils.data.Subset(dataset, sel_indices)
                loader_sel = DataLoader(subset_sel, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                # Features for both models
                feats_b, labels_b = extract_features(model_base, loader_sel, device, layer=args.feature_layer, max_samples=-1)
                feats_p, labels_p = extract_features(model_pkb, loader_sel, device, layer=args.feature_layer, max_samples=-1)
                # Optional PCA before t-SNE
                if args.pca_dim > 0:
                    feats_b_p = apply_pca(feats_b, args.pca_dim)
                    feats_p_p = apply_pca(feats_p, args.pca_dim)
                else:
                    feats_b_p, feats_p_p = feats_b, feats_p
                X = np.vstack([feats_b_p, feats_p_p])
                model_flags = np.concatenate([np.zeros(len(feats_b_p), dtype=int), np.ones(len(feats_p_p), dtype=int)])
                labels_comb = np.concatenate([labels_b, labels_p])
                emb = embed_tsne(X, seed=args.seed)
                out_root = Path(args.out_dir) / 'compare' / 'tsne'
                ensure_dir(out_root)
                np.save(out_root / 'tsne_compare.npy', emb)
                sil_overall = _safe_silhouette(emb, labels_comb)
                title = 't-SNE Compare (PKB>Base classes)'
                if sil_overall is not None:
                    title += f' (sil={sil_overall:.3f})'
                plot_embedding_compare(emb, labels_comb, model_flags, dataset.classes, title, out_root / 'tsne_compare.png')
                # kNN (overall per model on their own embeddings)
                emb_b = emb[:len(feats_b_p)]
                emb_p = emb[len(feats_b_p):]
                acc_b = _knn_overall_accuracy(emb_b, labels_b, k=10)
                acc_p = _knn_overall_accuracy(emb_p, labels_p, k=10)
                _plot_knn_compare(acc_b, acc_p, 't-SNE Compare kNN', out_root / 'tsne_compare_knn.png')
                with open(out_root / 'tsne_compare_metrics.json', 'w', encoding='utf-8') as f:
                    json.dump({'silhouette_overall': sil_overall, 'knn_acc_base@10': acc_b, 'knn_acc_pkb@10': acc_p, 'better_classes': [int(c) for c in better_classes]}, f, indent=2)

        out_root = Path(args.out_dir) / 'compare'
        base_out = out_root / 'base'
        pkb_out = out_root / 'pkb'
        # Grad-CAM
        if args.do_gradcam:
            run_gradcam(args, model_base, device, dataset, base_out / 'gradcam', indices=chosen_indices)
            run_gradcam(args, model_pkb, device, dataset, pkb_out / 'gradcam', indices=chosen_indices)
        # ViT attention
        if args.do_vit_attn:
            run_vit_attention(args, model_base, device, dataset, base_out / 'vit_attn', indices=chosen_indices)
            run_vit_attention(args, model_pkb, device, dataset, pkb_out / 'vit_attn', indices=chosen_indices)
    else:
        # Grad-CAM (works best for CNNs). Use original full dataset subset if earlier subset applied.
        if args.do_gradcam or args.do_vit_attn:
            if args.checkpoint is None:
                raise ValueError('Single-model attention requested: please provide --checkpoint (omit --compare-two-models).')
            model_single = load_model(args, num_classes).to(device)
            if args.do_gradcam:
                gradcam_dir = Path(args.out_dir) / 'gradcam'
                run_gradcam(args, model_single, device, dataset, gradcam_dir)
            if args.do_vit_attn:
                vit_dir = Path(args.out_dir) / 'vit_attn'
                run_vit_attention(args, model_single, device, dataset, vit_dir)

    print('Visualization tasks complete.')


if __name__ == '__main__':
    main()