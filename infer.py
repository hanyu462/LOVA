"""Single-image inference with a user-given pointer.

    python infer.py --ckpt runs/C/last.pth --image foo.jpg --pointer 230,180 --out out.png
    python infer.py --ckpt runs/C/last.pth --image foo.jpg --pointer 230,180 --r-mode ones --out out_full.png

Pointer is given in ORIGINAL image pixels. Output PNG = [image | R map | masks], same layout as
eval_pointer.py --viz. For an interactive (click) version see demo.py, which reuses Predictor.

Note: V0 gates by masking, so latency does not depend on R.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from lova.data.common import MEAN, STD
from lova.models.head import postprocess
from lova.models.model import LOVAv0

COCO_CLASSES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic-light fire-hydrant stop-sign "
    "parking-meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
    "tie suitcase frisbee skis snowboard sports-ball kite baseball-bat baseball-glove skateboard surfboard "
    "tennis-racket bottle wine-glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
    "hot-dog pizza donut cake chair couch potted-plant bed dining-table toilet tv laptop mouse remote keyboard "
    "cell-phone microwave oven toaster sink refrigerator book clock vase scissors teddy-bear hair-drier toothbrush"
).split()


def prepare_image(img: Image.Image, size: int):
    """PIL RGB -> (image [3,S,S] normalized, valid4 [1,S/4,S/4], meta). Same as COCOPointerDataset eval path."""
    w, h = img.size
    s = size / max(h, w)
    nh, nw = max(4, round(h * s / 4) * 4), max(4, round(w * s / 4) * 4)
    im = torch.from_numpy(np.array(img.resize((nw, nh), Image.BILINEAR))).permute(2, 0, 1).float() / 255
    im = (im - MEAN) / STD
    image = torch.zeros(3, size, size)
    image[:, :nh, :nw] = im
    valid4 = torch.zeros(1, size // 4, size // 4)
    valid4[:, :nh // 4, :nw // 4] = 1
    meta = dict(orig_size=(h, w), new_size=(nh, nw), sx=nw / w, sy=nh / h)
    return image, valid4, meta


def make_r_override(mode: str, shape, device):
    """mode: pred | ones | zeros | const:<v>. (oracle needs GT, not available here.)"""
    if mode == "pred":
        return None
    if mode == "ones":
        return torch.ones(shape, device=device)
    if mode == "zeros":
        return torch.zeros(shape, device=device)
    if mode.startswith("const:"):
        return torch.full(shape, float(mode.split(":")[1]), device=device)
    raise ValueError(mode)


class Predictor:
    """Load once, call many times: predictor(pil_image, (x, y)) -> dict."""

    def __init__(self, ckpt_path: str, device: str | None = None, amp: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LOVAv0(**ckpt["config"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.size = ckpt.get("args", {}).get("img_size", 512)
        self.num_classes = ckpt["config"]["num_classes"]
        self.class_names = COCO_CLASSES if self.num_classes == len(COCO_CLASSES) else [f"cls{k}" for k in range(self.num_classes)]
        self.amp = amp and self.device.type == "cuda"
        self._cache = {}  # id(image) -> prepared tensors (so repeated clicks skip preprocessing)
        self(Image.new("RGB", (self.size, self.size)), (0, 0))  # warm-up (CUDA init, autotune) so the first real call is fast

    def prepare(self, img: Image.Image):
        key = id(img)
        if key not in self._cache:
            self._cache = {key: prepare_image(img.convert("RGB"), self.size)}
        return self._cache[key]

    @torch.no_grad()
    def __call__(self, img: Image.Image, pointer_xy, r_mode: str = "pred", score_thr: float = 0.05):
        """pointer_xy: (x, y) in ORIGINAL image pixels. Returns dict with masks at original resolution."""
        image, valid4, meta = self.prepare(img)
        h, w = meta["orig_size"]
        nh, nw = meta["new_size"]
        px, py = float(pointer_xy[0]) * meta["sx"], float(pointer_xy[1]) * meta["sy"]
        px, py = min(max(px, 0.0), nw - 1e-3), min(max(py, 0.0), nh - 1e-3)
        img_t = image[None].to(self.device)
        ptr = torch.tensor([[px, py]], device=self.device)
        r_over = make_r_override(r_mode, (1, 1, self.size // 4, self.size // 4), self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = self.model(img_t, ptr, r_override=r_over)
        pred = postprocess(out, score_thr=score_thr)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        # R and masks back to original resolution (crop padding first).
        r4 = out["r"][0].float()  # [1,S/4,S/4]
        r_full = F.interpolate(r4[None], size=(self.size, self.size), mode="bilinear", align_corners=False)[0, :, :nh, :nw]
        r_full = F.interpolate(r_full[None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        if len(pred["scores"]):
            m = F.interpolate(pred["masks"][None].float(), scale_factor=4, mode="bilinear", align_corners=False)[0][:, :nh, :nw]
            masks = F.interpolate(m[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        else:
            masks = torch.zeros(0, h, w, device=self.device)
        mean_r = torch.stack([(r_full * (mk > 0.5)).sum() / (mk > 0.5).sum().clamp(min=1) for mk in masks]) \
            if len(masks) else torch.zeros(0, device=self.device)
        centers = pred["centers"] / torch.tensor([meta["sx"], meta["sy"]], device=self.device)
        return dict(
            r=r_full.cpu(), masks=masks.cpu(), scores=pred["scores"].float().cpu(), labels=pred["labels"].cpu(),
            centers=centers.cpu(), mean_r=mean_r.cpu(), latency_ms=latency_ms,
            r_mean=float((r4 * valid4.to(self.device)).sum() / valid4.sum()),
            vcompute=float(self.model.backbone.virtual_compute(r4[None], valid4[None].to(self.device))[0]),
            pointer_input=(px, py), meta=meta,
        )

    def detections(self, res, det_thr: float = 0.3):
        """Human-readable list sorted by score."""
        rows = []
        for k in range(len(res["scores"])):
            if res["scores"][k] < det_thr:
                continue
            rows.append(dict(label=self.class_names[int(res["labels"][k])], score=float(res["scores"][k]),
                             mean_r=float(res["mean_r"][k]), center=[float(v) for v in res["centers"][k]],
                             area=int((res["masks"][k] > 0.5).sum())))
        return rows


def _palette(n: int):
    gen = torch.Generator().manual_seed(0)
    return (torch.rand(n, 3, generator=gen) * 0.6 + 0.4)


def render_r(img: Image.Image, r: torch.Tensor, pointer_xy=None) -> Image.Image:
    """R heatmap (red = high, blue = low) blended on the image."""
    base = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255
    rr = r[None]
    panel = torch.cat([rr, torch.zeros_like(rr), 1 - rr], 0) * 0.6 + base * 0.4
    out = Image.fromarray((panel.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    if pointer_xy is not None:
        _draw_pointer(out, pointer_xy)
    return out


def render_masks(img: Image.Image, res, class_names, det_thr: float = 0.3, pointer_xy=None, labels=True) -> Image.Image:
    base = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255
    keep = (res["scores"] >= det_thr).nonzero()[:, 0].tolist()
    colors = _palette(max(len(res["scores"]), 1))
    over = base.clone()
    for k in reversed(keep):  # high score drawn last (on top)
        mm = res["masks"][k] > 0.5
        over[:, mm] = over[:, mm] * 0.4 + colors[k].view(3, 1) * 0.6
    out = Image.fromarray((over.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    if labels:
        for k in keep:
            cx, cy = res["centers"][k].tolist()
            txt = f"{class_names[int(res['labels'][k])]} {float(res['scores'][k]):.2f} R={float(res['mean_r'][k]):.2f}"
            tw = d.textlength(txt)
            d.rectangle([cx - 2, cy - 12, cx + tw + 2, cy + 2], fill=(0, 0, 0))
            d.text((cx, cy - 11), txt, fill=tuple(int(v * 255) for v in colors[k].tolist()))
    if pointer_xy is not None:
        _draw_pointer(out, pointer_xy)
    return out


def _draw_pointer(pil: Image.Image, xy, radius: int = 6):
    d = ImageDraw.Draw(pil)
    x, y = float(xy[0]), float(xy[1])
    d.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 255, 0), width=3)
    d.line([x - 2 * radius, y, x + 2 * radius, y], fill=(255, 255, 0), width=1)
    d.line([x, y - 2 * radius, x, y + 2 * radius], fill=(255, 255, 0), width=1)


def render_panel(img: Image.Image, res, class_names, det_thr: float = 0.3, pointer_xy=None) -> Image.Image:
    """[image | R | masks] side by side."""
    img = img.convert("RGB")
    a = img.copy()
    if pointer_xy is not None:
        _draw_pointer(a, pointer_xy)
    b = render_r(img, res["r"], pointer_xy)
    c = render_masks(img, res, class_names, det_thr, pointer_xy)
    w, h = img.size
    canvas = Image.new("RGB", (3 * w, h))
    for i, p in enumerate((a, b, c)):
        canvas.paste(p, (i * w, 0))
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--pointer", required=True, help="x,y in original image pixels")
    p.add_argument("--r-mode", default="pred", help="pred | ones | zeros | const:<v>")
    p.add_argument("--det-thr", type=float, default=0.3)
    p.add_argument("--out", default="infer_out.png")
    p.add_argument("--device", default=None)
    p.add_argument("--repeat", type=int, default=1, help="extra timed runs for a stable latency number")
    a = p.parse_args()

    pred = Predictor(a.ckpt, a.device)
    img = Image.open(a.image)
    xy = tuple(float(v) for v in a.pointer.split(","))
    res = pred(img, xy, a.r_mode)
    lat = [res["latency_ms"]] + [pred(img, xy, a.r_mode)["latency_ms"] for _ in range(a.repeat - 1)]
    render_panel(img, res, pred.class_names, a.det_thr, xy).save(a.out)

    print(f"pointer (orig px) = {xy}   r_mode = {a.r_mode}   device = {pred.device}")
    print(f"R mean = {res['r_mean']:.3f}   vcompute = {res['vcompute']:.3f}   latency = {np.median(lat):.1f} ms (median of {len(lat)})")
    rows = pred.detections(res, a.det_thr)
    print(f"{len(rows)} detections (score >= {a.det_thr}):")
    for r in rows:
        print(f"  {r['label']:>14} score={r['score']:.3f} R_in={r['mean_r']:.2f} center=({r['center'][0]:.0f},{r['center'][1]:.0f}) area={r['area']}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
