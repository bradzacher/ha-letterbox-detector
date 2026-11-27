"""Offline check that v1.4.2 would have held on the 27 Nov 13:11 delivery."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "letterbox_detector"))
from letterbox import FrameScore, LetterboxDetector  # noqa: E402


# Logged scores from local_letterbox_detector 1.4.1 at 13:11:10–13:11:13 ACDT
# on 27 Nov 2026. Only these frames cleared the add-on's info near-miss gate.
FRAMES = [
    dict(i=9, open_frame=False, area=25382, roi_frac=0.035, left_area=10869, sharp=1.97),
    dict(i=10, open_frame=False, area=25382, roi_frac=0.035, left_area=10869, sharp=1.97),
    dict(i=11, open_frame=True, area=183925, roi_frac=0.252, left_area=16512, sharp=5.09),
    dict(i=12, open_frame=False, area=33810, roi_frac=0.046, left_area=10831, sharp=4.08),
    dict(i=13, open_frame=False, area=36312, roi_frac=0.050, left_area=11051, sharp=2.47),
    dict(i=14, open_frame=False, area=36261, roi_frac=0.050, left_area=15865, sharp=1.90),
    dict(i=15, open_frame=False, area=15090, roi_frac=0.021, left_area=14548, sharp=4.19),
]


def score_from(row: dict) -> FrameScore:
    return FrameScore(
        open_frame=row["open_frame"],
        area=row["area"],
        roi_frac=row["roi_frac"],
        left_area=row["left_area"],
        sharp=row["sharp"],
    )


def replay(hold_like_v141: bool = False) -> list[dict]:
    det = LetterboxDetector(hold_frames=3)
    rows = []
    for row in FRAMES:
        if hold_like_v141:
            # v1.4.1: only open_frame increments the streak.
            s = score_from(row)
            det.last_score = s
            if s.open_frame:
                det._positives += 1
            else:
                det._positives = 0
            held = det._positives >= det.hold_frames
            det.state = held
            rec = {
                "i": row["i"],
                "open_frame": s.open_frame,
                "near_open": False,
                "near_open_count": 0,
                "positives": det._positives,
                "state": held,
            }
        else:
            rec = det.consider_score(score_from(row))
            rec = {
                "i": row["i"],
                "open_frame": rec["open_frame"],
                "near_open": rec["near_open"],
                "near_open_count": rec["near_open_count"],
                "positives": rec["positives"],
                "state": rec["state"],
            }
        rows.append(rec)
        print(
            "  frame {i} open={open_frame} near_open={near_open} near_count={near_open_count} "
            "positives={positives} held={state}".format(**rec)
        )
    return rows


def main() -> int:
    print("=== v1.4.1 hold logic (open_frame only) ===")
    old = replay(hold_like_v141=True)
    print("=== v1.4.2 hold logic (open then near-open) ===")
    new = replay(hold_like_v141=False)

    assert not any(r["state"] for r in old), "v1.4.1 should not have held on this burst"
    held_at = next((r["i"] for r in new if r["state"]), None)
    assert held_at == 13, f"v1.4.2 should first hold at frame 13, got {held_at}"
    assert new[2]["positives"] == 1 and new[2]["open_frame"]
    assert new[3]["near_open"] and new[3]["positives"] == 2
    assert new[4]["near_open"] and new[4]["positives"] == 3 and new[4]["state"]
    # frame 14 sharp 1.90 < 2.0, frame 15 area 15090 < 30000: streak resets
    assert not new[5]["near_open"] and new[5]["positives"] == 0 and not new[5]["state"]
    assert not new[6]["near_open"] and new[6]["positives"] == 0
    print("PASS: v1.4.2 reaches hold at frame 13; v1.4.1 never holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
