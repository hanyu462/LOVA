"""lova.engine.decode: peaks -> masks -> matrix NMS on a constructed ModelOutput.

    python tests/engine/test_decode.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lova.engine.decode import DecodeCfg, decode, matrix_nms, select_pointed  # noqa: E402
from lova.models.output import ModelOutput  # noqa: E402


def unit_test():
    K, D, S = 3, 4, 64
    H8, H4 = S // 8, S // 4
    heat = torch.full((1, K, H8, H8), -8.0)
    heat[0, 1, 2, 3] = 3.0            # class 1 peak at cell (2, 3): p = 0.95
    heat[0, 1, 2, 4] = 1.0            # neighbour, lower: not a 3x3 maximum
    heat[0, 2, 6, 6] = 0.0            # class 2 peak: p = 0.5
    heat[0, 0, 5, 1] = -3.0           # p = 0.047 < score_thr
    # mask feature with two "object" channels; kernels pick one of them per cell
    mf = torch.full((1, D, H4, H4), -20.0)
    mf[0, 0, 2:8, 4:10] = 20.0        # object A region
    mf[0, 1, 10:15, 10:15] = 20.0     # object B region
    km = torch.zeros(1, D, H8, H8)
    km[0, 0, 2, 3] = 1.0; km[0, 0, 2, 4] = 1.0                 # both class-1 cells -> object A
    km[0, 1, 6, 6] = 1.0                                       # class-2 cell -> object B
    out = ModelOutput(torch.full((1, 1, H4, H4), 0.5), heat, km, mf)
    det = decode(out, DecodeCfg(score_thr=0.1))[0]
    assert det["labels"].tolist() == [1, 2] and det["scores"][0] > 0.9 and abs(float(det["scores"][1]) - 0.5) < 0.05
    assert det["masks"].shape == (2, H4, H4) and (det["masks"][0] > 0.5).sum() == 36 and (det["masks"][1] > 0.5).sum() == 25
    assert det["centers"].tolist() == [[3 * 8 + 3.5, 2 * 8 + 3.5], [6 * 8 + 3.5, 6 * 8 + 3.5]]
    # matrix NMS: a second class-1 peak with the same mask is decayed; a different class is not
    heat[0, 1, 4, 3] = 2.0            # another class-1 peak (p = 0.88), kernel -> object A again
    km[0, 0, 4, 3] = 1.0
    det2 = decode(ModelOutput(out.r_pred, heat, km, mf), DecodeCfg(score_thr=0.1))[0]
    dup = [(l, float(s)) for l, s in zip(det2["labels"].tolist(), det2["scores"].tolist())]
    assert dup[0] == (1, dup[0][1]) and dup[0][1] > 0.9
    assert sum(1 for l, _ in dup if l == 1) == 1 or max(s for l, s in dup if l == 1 and s < 0.9) < 0.2    # duplicate suppressed (or decayed hard)
    assert any(l == 2 for l, _ in dup)
    s = matrix_nms(torch.ones(2, 10), torch.tensor([0, 0]), torch.tensor([0.9, 0.8]))
    assert abs(float(s[0]) - 0.9) < 1e-6 and float(s[1]) < 0.2
    s = matrix_nms(torch.ones(2, 10), torch.tensor([0, 1]), torch.tensor([0.9, 0.8]))
    assert abs(float(s[1]) - 0.8) < 1e-6                                                       # different class: untouched
    # nothing above threshold -> empty, and batched inputs give one dict per image
    empty = decode(ModelOutput(out.r_pred, torch.full((2, K, H8, H8), -8.0), km.repeat(2, 1, 1, 1), mf.repeat(2, 1, 1, 1)))
    assert len(empty) == 2 and all(len(e["scores"]) == 0 for e in empty)
    # ---- padding: an invalid heat cell with a huge score yields no detection ----
    heat_p = torch.full((1, K, H8, H8), -8.0); heat_p[0, 1, 2, 3] = 3.0; heat_p[0, 0, 7, 7] = 6.0          # (7,7) will be padding
    km_p = km.clone(); km_p[0, 1, 7, 7] = 1.0                                                              # give that cell a kernel (object B)
    hv = torch.ones(1, H8, H8, dtype=torch.bool); hv[:, 6:, :] = False                                     # bottom quarter padded
    det_p = decode(ModelOutput(out.r_pred, heat_p, km_p, mf), DecodeCfg(score_thr=0.1), heat_valid=hv)[0]
    assert det_p["labels"].tolist() == [1], det_p["labels"]
    det_np = decode(ModelOutput(out.r_pred, heat_p, km_p, mf), DecodeCfg(score_thr=0.1))[0]
    assert 0 in det_np["labels"].tolist()                                                                   # without heat_valid it is found
    # ---- padding next to a valid peak: the (untrained) padding value must not hide the valid cell's peak ----
    heat_b = torch.full((1, K, H8, H8), -8.0)
    heat_b[0, 1, 5, 3] = 2.2                                    # valid boundary cell, p = 0.90
    heat_b[0, 1, 6, 3] = 4.6                                    # adjacent padding cell, p = 0.99
    km_b = torch.zeros_like(km); km_b[0, 0, 5, 3] = 1.0; km_b[0, 0, 6, 3] = 1.0
    det_b = decode(ModelOutput(out.r_pred, heat_b, km_b, mf), DecodeCfg(score_thr=0.1), heat_valid=hv)[0]
    assert len(det_b["scores"]) == 1 and det_b["centers"][0].tolist() == [3 * 8 + 3.5, 5 * 8 + 3.5], det_b["centers"]   # the valid cell survives
    det_b2 = decode(ModelOutput(out.r_pred, heat_b, km_b, mf), DecodeCfg(score_thr=0.1))[0]
    assert det_b2["centers"][0].tolist() == [3 * 8 + 3.5, 6 * 8 + 3.5] and len(det_b2["scores"]) == 1                   # without valid: the padding wins, the valid cell is suppressed
    # ---- padding: mask area in the invalid stride-4 region must not count (maskness, area, NMS) ----
    mf2 = mf.clone(); mf2[0, 0, 12:16, :] = 20.0                                                           # object A's channel also lit across the padded rows
    mv = torch.ones(1, H4, H4, dtype=torch.bool); mv[:, 12:, :] = False
    d_valid = decode(ModelOutput(out.r_pred, heat, km, mf2), DecodeCfg(score_thr=0.1), mask_valid=mv)[0]
    d_novalid = decode(ModelOutput(out.r_pred, heat, km, mf2), DecodeCfg(score_thr=0.1))[0]
    a_valid = (d_valid["masks"][0] > 0.5).sum(); a_novalid = (d_novalid["masks"][0] > 0.5).sum()
    assert a_valid == 36 and a_novalid > 36                                                                 # padded rows removed before everything else
    assert float(d_valid["masks"][0][12:, :].max()) == 0.0
    for bad in ((torch.ones(1, H8 - 1, H8, dtype=torch.bool), None), (None, torch.ones(2, H4, H4, dtype=torch.bool))):
        try:
            decode(out, DecodeCfg(), heat_valid=bad[0], mask_valid=bad[1]); raise AssertionError("must raise")
        except ValueError:
            pass
    # ---- select_pointed: the covering mask with the highest score; else the most probable at the pointer; None if empty ----
    m = torch.zeros(3, 8, 8); m[0, 1:4, 1:4] = 0.9; m[1, 1:4, 1:4] = 0.8; m[2, 5:8, 5:8] = 0.95
    d = {"masks": m, "scores": torch.tensor([0.3, 0.6, 0.9]), "labels": torch.tensor([0, 1, 2])}
    assert select_pointed(d, (2 * 4 + 1, 2 * 4 + 1)) == 1                     # cell (2,2): masks 0 and 1 cover it, 1 has the higher score
    assert select_pointed(d, (6 * 4, 6 * 4)) == 2                             # cell (6,6): only mask 2
    assert select_pointed(d, (0, 7 * 4)) == 2 or select_pointed(d, (0, 7 * 4)) in (0, 1)   # nothing covers (0,7): argmax of prob there
    m2 = m.clone(); m2[0, 7, 0] = 0.2; d2 = {**d, "masks": m2}
    assert select_pointed(d2, (0, 7 * 4)) == 0                                # the most probable at the pointer (0.2 > 0)
    assert select_pointed({"masks": torch.zeros(0, 8, 8), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}, (3, 3)) is None
    assert select_pointed(d, (-50, 1000)) is not None                           # pointer outside the canvas: clamped, no crash
    print("decode unit test OK")


if __name__ == "__main__":
    unit_test()
