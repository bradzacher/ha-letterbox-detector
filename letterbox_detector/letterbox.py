"""Deterministic letterbox-lid detector for the Front Gate package camera.

Each frame is compared to a closed-lid reference after illumination
normalization. A lid-open decision requires:

  1. A large darker-than-reference residual in the right-hand region
     (the raised lid / its interior).
  2. A sharp, dark-on-the-right vertical Sobel peak in that region —
     the lid's straight vertical bar, which shadows and a person's
     arm or legs do not produce.
  3. The residual must stay local to the lid. A global or near-global
     change (headlights, exposure swings, a dark flash) can light up
     the whole right-hand region and is rejected.
  4. Sharpness must stay in a realistic band. A near-black or
     crushed-exposure frame can blow up the peak/median ratio.

The published state goes on only after hold_frames consecutive
positives (3 frames at 3 fps is about 1 second).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

ROI_X0, ROI_X1 = 0.50, 1.00
ROI_Y0, ROI_Y1 = 0.12, 0.88
# Control region: left of the lid, same vertical band. A real open
# barely changes this area; headlights and exposure swings often do.
CTRL_X0, CTRL_X1 = 0.00, 0.45
ILLUM_SIGMA = 61.0
DIFF_SCALE = 120.0
DIFF_THRESH = 35
MORPH_OPEN = 9
MORPH_CLOSE = 21
# Tuned on the 16 Nov 2026 17:07:10–35 ACDT clip, 17:31 live stills,
# the 20:38 / 21:15 ACDT headlight false positives (full-ROI residual
# ~0.98, area ~720k, sharpness 7 and 667), and the 17 Nov 2026 13:45
# delivery miss: a real mid-swing lid reached roi_frac 0.50–0.65 with
# sharpness ~4.0–4.5 (one crisp bar at 28.56). Headlights stay above
# 0.95, so the localization cap can sit at 0.75. A slightly soft
# sharpness (>=3.8) is accepted only when the residual is large and local.
MIN_AREA = 50_000
MAX_ROI_FRAC = 0.75
SHARP_MIN = 5.0
SHARP_SOFT = 3.8
MIN_AREA_SOFT = 150_000
SHARP_MAX = 40.0
HOLD_FRAMES = 3
REF_EMA_SLOW = 0.004
# 19 Nov 2026 13:27 ACDT miss: a courier held a brown parcel across the
# raised lid. The bright parcel hid the dark-residual and the lid bar
# (area 40k, sharp 2.1), so the start-of-watch refresh took three
# lid-open frames as the new "closed" reference. Learning now also
# needs the two-sided (brighter or darker) change against the current
# reference to be small; closed frames sit under ~2k, a person or a
# parcel is 48k-180k.
LEARN_MAX_ABS = 25_000
# Edge-on lid: a tall, narrow darker residual that reaches the bottom
# of the frame (the lid rises from the box at the bottom edge). Covers
# a parcel held across the lid, where the Sobel bar score stays soft.
TALL_MIN_AREA = 35_000
TALL_MIN_H = 0.45
TALL_MIN_BOTTOM = 0.80
TALL_MAX_W = 0.30
TALL_SHARP_MIN = 2.0
SOBEL_X0, SOBEL_X1 = 0.48, 0.98
SOBEL_Y0, SOBEL_Y1 = 0.15, 0.85
SOBEL_SMOOTH = 11


@dataclass
class FrameScore:
    open_frame: bool
    area: int = 0
    bbox: Optional[tuple[int, int, int, int]] = None
    fill: float = 0.0
    sharp: float = 0.0
    peak: float = 0.0
    roi_frac: float = 0.0
    left_area: int = 0
    abs_area: int = 0
    tall: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def global_illum(self) -> bool:
        return self.roi_frac > MAX_ROI_FRAC or self.left_area >= MIN_AREA


def _illum_norm(gray: np.ndarray, sigma: float = ILLUM_SIGMA) -> np.ndarray:
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return g / (blur + 8.0)


def _vertical_bar_sharpness(gray: np.ndarray) -> tuple[float, float]:
    """Return (sharpness, peak) for a dark-on-the-right vertical bar.

    Sharpness is the smoothed column energy of negative Sobel-X (right
    side darker than left) divided by the column-energy median. A
    raised lid makes a thin, high-contrast peak; shadows and a person
    spread energy out and stay near 1–3.5 on the tuning set. Extreme
    ratios come from crushed or empty frames, not the lid.
    """
    h, w = gray.shape
    y0, y1 = int(h * SOBEL_Y0), int(h * SOBEL_Y1)
    x0, x1 = int(w * SOBEL_X0), int(w * SOBEL_X1)
    roi = gray[y0:y1, x0:x1]
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    sx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    col = np.mean(np.clip(-sx, 0, None), axis=0)
    k = min(SOBEL_SMOOTH, max(1, (len(col) // 2) * 2 + 1))
    col = np.convolve(col, np.ones(k) / k, mode="same")
    peak = float(col.max()) if col.size else 0.0
    med = float(np.median(col) + 1e-3) if col.size else 1e-3
    return peak / med, peak


def _residual_mask(darker: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    vis = np.clip(darker * DIFF_SCALE, 0, 255).astype(np.uint8)
    _, th = cv2.threshold(vis, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    h, w = th.shape
    mask = np.zeros_like(th)
    mask[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)] = 255
    th = cv2.bitwise_and(th, mask)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_OPEN, MORPH_OPEN))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_CLOSE, MORPH_CLOSE))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)
    return th


def _largest_blob(th: np.ndarray) -> tuple[int, Optional[tuple[int, int, int, int]], float]:
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = int(cv2.contourArea(c))
    x, y, cw, ch = cv2.boundingRect(c)
    fill = float(area) / float(cw * ch + 1e-6)
    return area, (x, y, cw, ch), fill


class LetterboxDetector:
    def __init__(
        self,
        hold_frames: int = HOLD_FRAMES,
        min_area: int = MIN_AREA,
        sharp_min: float = SHARP_MIN,
        max_roi_frac: float = MAX_ROI_FRAC,
        sharp_max: float = SHARP_MAX,
        sharp_soft: float = SHARP_SOFT,
        min_area_soft: int = MIN_AREA_SOFT,
    ):
        self.hold_frames = int(hold_frames)
        self.min_area = int(min_area)
        self.sharp_min = float(sharp_min)
        self.max_roi_frac = float(max_roi_frac)
        self.sharp_max = float(sharp_max)
        self.sharp_soft = float(sharp_soft)
        self.min_area_soft = int(min_area_soft)
        self.ref_bgr: Optional[np.ndarray] = None
        self.ref_n: Optional[np.ndarray] = None
        self._positives = 0
        self.state = False
        self.last_score: Optional[FrameScore] = None

    def set_reference(self, bgr: np.ndarray) -> None:
        if bgr is None or bgr.size == 0:
            raise ValueError("empty reference")
        self.ref_bgr = bgr.copy()
        gray = cv2.cvtColor(self.ref_bgr, cv2.COLOR_BGR2GRAY)
        self.ref_n = _illum_norm(gray)

    def has_reference(self) -> bool:
        return self.ref_n is not None

    def _adapt_reference(self, bgr: np.ndarray, rate: float) -> None:
        if self.ref_bgr is None:
            self.set_reference(bgr)
            return
        if self.ref_bgr.shape != bgr.shape:
            bgr = cv2.resize(bgr, (self.ref_bgr.shape[1], self.ref_bgr.shape[0]))
        blended = cv2.addWeighted(self.ref_bgr, 1.0 - rate, bgr, rate, 0)
        self.set_reference(blended)

    def should_learn_reference(self, score: FrameScore) -> bool:
        """True when the frame looks closed enough to update the reference."""
        if score.open_frame or score.sharp >= self.sharp_min * 0.7:
            return False
        if score.global_illum:
            return True
        # Never learn a frame that still differs from the reference in
        # either direction (person, parcel held over the lid).
        if score.abs_area >= LEARN_MAX_ABS:
            return False
        # A parcel left in the box after a real open leaves a modest
        # residual (often 25–40k) that is still clearly not a lid.
        # Absorb that as the new closed scene, slowly, so the next
        # delivery is compared against the box as it actually looks.
        if score.area < self.min_area:
            return True
        # Global lighting change with no lid-like edge: snap/adapt to
        # the new illumination rather than holding a stale reference.
        return score.global_illum

    def score_frame(self, bgr: np.ndarray) -> FrameScore:
        if self.ref_n is None:
            return FrameScore(False, reasons=["no_reference"])
        if bgr is None or bgr.size == 0:
            return FrameScore(False, reasons=["empty_frame"])
        if bgr.shape[:2] != self.ref_n.shape[:2]:
            bgr = cv2.resize(bgr, (self.ref_n.shape[1], self.ref_n.shape[0]))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        n = _illum_norm(gray)
        darker = np.clip(self.ref_n - n, 0, None)
        h, w = gray.shape
        roi_th = _residual_mask(darker, ROI_X0, ROI_X1, ROI_Y0, ROI_Y1)
        ctrl_th = _residual_mask(darker, CTRL_X0, CTRL_X1, ROI_Y0, ROI_Y1)
        area, bbox, fill = _largest_blob(roi_th)
        abs_th = _residual_mask(np.abs(self.ref_n - n), 0.0, 1.0, ROI_Y0, ROI_Y1)
        abs_area, _, _ = _largest_blob(abs_th)
        tall = False
        if bbox is not None:
            bx, by, bw, bh = bbox
            tall = (
                area >= TALL_MIN_AREA
                and bh >= TALL_MIN_H * h
                and (by + bh) >= TALL_MIN_BOTTOM * h
                and bw <= TALL_MAX_W * w
            )
        left_area, _, _ = _largest_blob(ctrl_th)
        roi_area = max(1, int((ROI_Y1 - ROI_Y0) * h) * int((ROI_X1 - ROI_X0) * w))
        roi_frac = float(area) / float(roi_area)
        sharp, peak = _vertical_bar_sharpness(gray)
        reasons = []
        if area < self.min_area:
            reasons.append(f"area {area}<{self.min_area}")
        if roi_frac > self.max_roi_frac:
            reasons.append(f"roi_frac {roi_frac:.2f}>{self.max_roi_frac}")
        if left_area >= self.min_area:
            reasons.append(f"left_area {left_area}>={self.min_area}")
        sharp_hard = self.sharp_min <= sharp <= self.sharp_max
        sharp_soft = (
            self.sharp_soft <= sharp <= self.sharp_max
            and area >= self.min_area_soft
        )
        if not (sharp_hard or sharp_soft):
            if sharp < self.sharp_min:
                reasons.append(f"sharp {sharp:.2f}<{self.sharp_min}")
            if sharp > self.sharp_max:
                reasons.append(f"sharp {sharp:.2f}>{self.sharp_max}")
        open_frame = (
            area >= self.min_area
            and roi_frac <= self.max_roi_frac
            and left_area < self.min_area
            and (sharp_hard or sharp_soft)
        )
        tall_open = (
            tall
            and TALL_SHARP_MIN <= sharp <= self.sharp_max
            and roi_frac <= self.max_roi_frac
            and left_area < self.min_area
        )
        if tall_open and not open_frame:
            open_frame = True
            reasons = ["tall_lid"]
        return FrameScore(
            open_frame=open_frame,
            area=area,
            bbox=bbox,
            fill=fill,
            sharp=sharp,
            peak=peak,
            roi_frac=roi_frac,
            left_area=left_area,
            abs_area=abs_area,
            tall=tall,
            reasons=reasons,
        )

    def feed(self, bgr: np.ndarray) -> dict:
        score = self.score_frame(bgr)
        self.last_score = score
        if score.open_frame:
            self._positives += 1
        else:
            self._positives = 0
            if self.should_learn_reference(score):
                if score.global_illum:
                    # Lighting changed broadly and the lid is not open.
                    # Replace the reference so a day ref cannot linger
                    # into night or a post-headlight scene.
                    self.set_reference(bgr)
                else:
                    self._adapt_reference(bgr, REF_EMA_SLOW)
        held = self._positives >= self.hold_frames
        self.state = held
        return {
            "open_frame": score.open_frame,
            "state": held,
            "positives": self._positives,
            "area": score.area,
            "bbox": score.bbox,
            "fill": score.fill,
            "sharp": score.sharp,
            "peak": score.peak,
            "roi_frac": score.roi_frac,
            "left_area": score.left_area,
            "reasons": score.reasons,
        }

    def bootstrap_reference(self, bgr: np.ndarray) -> bool:
        """Accept a startup frame as the closed reference if it looks closed."""
        if bgr is None or bgr.size == 0:
            return False
        # Sharpness does not need a reference. A large residual is not
        # available yet, so only reject a clearly lid-like vertical bar.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sharp, _ = _vertical_bar_sharpness(gray)
        if sharp >= self.sharp_min:
            return False
        self.set_reference(bgr)
        return True
