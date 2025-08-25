# Important Algorithms

## PKB

```py
class PatchKeepBlur:
    """PatchKeep-Blur (PKB) augmentation.
    Splits image into n x n patches, blurs a subset, keeps rest sharp.
    """
    def __init__(self, n: int = 4, a: Optional[int] = None, a_fraction: Optional[float] = 0.25,
                 sigma: float = 2.0, placement: str = 'random', seed: Optional[int] = None):
        assert n > 1, 'n must be > 1'
        self.n = n
        total = n * n
        if a is None:
            if a_fraction is None:
                raise ValueError('Provide a or a_fraction')
            a = max(1, min(total - 1, int(round(a_fraction * total))))
        if not (0 < a < total):
            raise ValueError(f'a must be in (0,{total}) got {a}')
        self.a = a
        self.sigma = sigma
        self.placement = placement
        self.base_rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.a == 0:
            return img
        w, h = img.size
        pw = w // self.n
        ph = h // self.n
        if pw == 0 or ph == 0:
            return img
        indices = self._select_indices()
        return self._apply_indices(img, indices)

    def _apply_indices(self, img: Image.Image, indices: Iterable[int]) -> Image.Image:
        """Apply blur to specified patch indices (no randomness inside)."""
        out = img.copy()
        w, h = img.size
        pw = w // self.n
        ph = h // self.n
        blur_filter = ImageFilter.GaussianBlur(self.sigma)
        for idx in indices:
            r, c = divmod(idx, self.n)
            left = c * pw
            upper = r * ph
            right = (c + 1) * pw if c < self.n - 1 else w
            lower = (r + 1) * ph if r < self.n - 1 else h
            patch = out.crop((left, upper, right, lower))
            patch = patch.filter(blur_filter)
            out.paste(patch, (left, upper))
        return out

    def _select_indices(self) -> List[int]:
        total = self.n * self.n
        if self.placement == 'random':
            return self.base_rng.sample(range(total), self.a)
        if self.placement == 'dispersed':
            return self._select_dispersed()
        if self.placement == 'contiguous':
            return self._select_contiguous()
        raise ValueError('Unknown placement')

    def _select_dispersed(self) -> List[int]:
        coords = [(i // self.n, i % self.n) for i in range(self.n * self.n)]
        remaining = list(range(self.n * self.n))
        first = self.base_rng.choice(remaining)
        selected = [first]
        remaining.remove(first)
        def dist2(a, b):
            return (a[0]-b[0])**2 + (a[1]-b[1])**2
        while len(selected) < self.a and remaining:
            best_idx = None
            best_d = -1
            for cand in remaining:
                d = min(dist2(coords[cand], coords[s]) for s in selected)
                if d > best_d:
                    best_d = d
                    best_idx = cand
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    def _select_contiguous(self) -> List[int]:
        start = self.base_rng.randrange(self.n * self.n)
        selected = {start}
        frontier = [start]
        while len(selected) < self.a and frontier:
            current = self.base_rng.choice(frontier)
            r, c = divmod(current, self.n)
            neighbors = []
            if r > 0: neighbors.append((r-1, c))
            if r < self.n-1: neighbors.append((r+1, c))
            if c > 0: neighbors.append((r, c-1))
            if c < self.n-1: neighbors.append((r, c+1))
            self.base_rng.shuffle(neighbors)
            for nr, nc in neighbors:
                idx = nr * self.n + nc
                if idx not in selected:
                    selected.add(idx)
                    frontier.append(idx)
                    if len(selected) >= self.a:
                        break
            if all((nr * self.n + nc) in selected for nr, nc in neighbors):
                frontier.remove(current)
        if len(selected) < self.a:
            remaining = [i for i in range(self.n * self.n) if i not in selected]
            self.base_rng.shuffle(remaining)
            for i in remaining[:self.a - len(selected)]:
                selected.add(i)
        return list(selected)


class MultiViewPatchKeepBlur:
    """Generate multiple distinct PKB views (unique patch index sets) and stack as tensor.

    Returns torch.Tensor shape (V, C, H, W) after provided post_transform.
    Assumes pre_transform produces a PIL.Image, post_transform converts to tensor & normalizes.
    """
    def __init__(self, n: int = 4, a: Optional[int] = None, a_fraction: Optional[float] = 0.25,
                 sigma: float = 2.0, placement: str = 'random', views: int = 2,
                 seed: Optional[int] = None, post_transform=None, max_resample_factor: int = 20,
                 highlight: bool = False, highlight_color: str = 'E30918'):
        assert views > 1, 'Use PatchKeepBlur for single view'
        self.views = views
        self.post_transform = post_transform
        self.pkb = PatchKeepBlur(n=n, a=a, a_fraction=a_fraction, sigma=sigma, placement=placement, seed=seed,
                                 highlight=highlight, highlight_color=highlight_color)
        self.max_attempts = max_resample_factor * views

    def __call__(self, img: Image.Image):
        # img should already be augmented & cropped to final size before this call.
        collected = []
        used: Set[Tuple[int,...]] = set()
        attempts = 0
        while len(collected) < self.views and attempts < self.max_attempts:
            indices = self.pkb._select_indices()
            key = tuple(sorted(indices))
            if key in used:
                attempts += 1
                continue
            used.add(key)
            view = self.pkb._apply_indices(img, indices)
            if self.post_transform:
                view = self.post_transform(view)
            collected.append(view)
        # If uniqueness space exhausted early, duplicate last to keep tensor size consistent.
        if len(collected) == 0:
            raise RuntimeError('MultiViewPatchKeepBlur produced 0 views')
        while len(collected) < self.views:
            collected.append(collected[-1])
        # Stack if tensors, else return list
        if isinstance(collected[0], torch.Tensor):
            return torch.stack(collected, dim=0)
        return collected
```

## Training Loop

```py
for images, targets in loader:
        targets = targets.to(device)
        if multiview:
            # images shape (B, V, C, H, W)
            b, v, c, h, w = images.shape
            images = images.view(b * v, c, h, w).to(device)
            targets_exp = targets.unsqueeze(1).repeat(1, v).view(-1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets_exp)
            loss.backward(); optimizer.step()
```

