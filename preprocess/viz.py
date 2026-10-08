"""Drawing helpers shared by the preprocess test scripts (rendering only, no logic)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw


def palette(n: int) -> np.ndarray:
    g = torch.Generator().manual_seed(0)
    return (torch.rand(max(n, 1), 3, generator=g) * 0.6 + 0.4).numpy()


def _erode(m: np.ndarray, k: int) -> np.ndarray:
    for _ in range(k):
        p = np.pad(m, 1)
        m = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m


def outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Inner boundary band of a bool mask, `width` pixels wide."""
    return mask & ~_erode(mask, width)


def overlay_masks(image: Image.Image, masks: torch.Tensor, keep: list[int], labels: list[str] | None = None,
                  alpha: float = 0.55, outline_width: int = 2) -> Image.Image:
    """keep: indices drawn as filled colour (+ outline); the rest as outline only."""
    img = np.asarray(image.convert("RGB")).astype(np.float32) / 255
    m = masks.cpu().numpy().astype(bool)
    colors = palette(len(m))
    keep_set = set(keep)
    for i in range(len(m)):
        if i in keep_set:
            img[m[i]] = img[m[i]] * (1 - alpha) + colors[i] * alpha
    for i in range(len(m)):
        ol = outline(m[i], outline_width)
        img[ol] = colors[i] if i in keep_set else colors[i] * 0.8
    out = Image.fromarray((img * 255).astype(np.uint8))
    if labels:
        d = ImageDraw.Draw(out)
        for i in range(len(m)):
            ys, xs = np.nonzero(m[i])
            if len(ys) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            tw = d.textlength(labels[i])
            on = i in keep_set
            d.rectangle([cx - 2, cy - 12, cx + tw + 2, cy + 2], fill=(0, 0, 0) if on else (70, 70, 70))
            d.text((cx, cy - 11), labels[i], fill=(255, 255, 255) if on else (190, 190, 190))
    return out


def draw_pointer(image: Image.Image, xy, radius: int = 6, color=(255, 255, 0)) -> Image.Image:
    d = ImageDraw.Draw(image)
    x, y = float(xy[0]), float(xy[1])
    d.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=3)
    d.line([x - 2 * radius, y, x + 2 * radius, y], fill=color, width=1)
    d.line([x, y - 2 * radius, x, y + 2 * radius], fill=color, width=1)
    return image


def hstack(images: list[Image.Image], titles: list[str] | None = None, pad: int = 6) -> Image.Image:
    top = 16 if titles else 0
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (w, h + top), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    x = 0
    for k, im in enumerate(images):
        canvas.paste(im, (x, top))
        if titles:
            d.text((x + 4, 2), titles[k], fill=(255, 255, 255))
        x += im.width + pad
    return canvas
