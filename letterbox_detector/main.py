#!/usr/bin/env python3
"""Front Gate package-camera letterbox detector add-on.

Idles with no RTSP decode until a Front Gate motion or person sensor
turns on, then watches the package camera for a limited window and
publishes binary_sensor.letterbox_opened. Stream URLs, API keys and
tokens are never written to logs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from letterbox import LetterboxDetector

OPTIONS_PATH = Path("/data/options.json")
REF_PATH = Path("/data/closed_reference.jpg")
ENTITY_ID = "binary_sensor.letterbox_opened"
WATCHING_ID = "binary_sensor.letterbox_detector_watching"
TEST_SENSOR_ID = "sensor.letterbox_detector_last_test"
HA_BASE_DEFAULT = "http://supervisor/core"
WS_URLS = (
    "ws://supervisor/core/websocket",
    "ws://supervisor/core/api/websocket",
)
DEFAULT_TRIGGERS = (
    "binary_sensor.front_gate_motion",
    "binary_sensor.front_gate_person_detected",
    "event.front_gate_smart_detection",
    "event.front_gate_lock_access",
    "event.front_gate_package",
    "binary_sensor.front_gate_is_dark",
)
# Always listen for these even if an older options.json omits them.
EXTRA_TRIGGERS = DEFAULT_TRIGGERS
# binary_sensors whose on-state means "activity still happening".
# is_dark is inverted (on = night) and must not keep a window open.
LEVEL_TRIGGERS = (
    "binary_sensor.front_gate_motion",
    "binary_sensor.front_gate_person_detected",
    "binary_sensor.front_gate_speaking_detected",
)
# event.front_gate_smart_detection types that should start a watch.
SMART_EVENT_TYPES = {"person", "face", "package"}
TEST_EVENT = "letterbox_detector_test"
DEFAULT_CLIP = "/share/letterbox/clip-ch3-170710-170735.mp4"
WATCH_LOG_ROOT = Path("/data/watches")
WATCH_LOG_KEEP = 20
WATCH_LOG_MAX_BYTES = 80 * 1024 * 1024
SHARE_LETTERBOX = Path("/share/letterbox")
SHARE_WATCH_ROOT = SHARE_LETTERBOX / "watches"
WATCH_NEAR_KEEP = 20
REF_PATH_NIGHT = Path("/data/closed_reference_night.jpg")
NIGHT_MEAN = 50.0
NIGHT_SEED = Path("/share/letterbox/closed-night.jpg")

log = logging.getLogger("letterbox")


def _redact_url(url: str) -> str:
    try:
        p = urlparse(url)
        path_len = len(p.path or "") + len(f"?{p.query}" if p.query else "")
        netloc = p.hostname or "unknown"
        if p.port:
            netloc = f"{netloc}:{p.port}"
        return f"{p.scheme}://{netloc}/<redacted len={path_len}>"
    except Exception:
        return f"<url len={len(url)}>"


def _load_options() -> dict:
    if OPTIONS_PATH.exists():
        return json.loads(OPTIONS_PATH.read_text())
    return {}


def _supervisor_token() -> str:
    for key in ("SUPERVISOR_TOKEN", "HASSIO_TOKEN"):
        val = os.environ.get(key) or ""
        if val:
            return val
    for path in (
        Path("/run/s6/container_environment/SUPERVISOR_TOKEN"),
        Path("/run/s6/container_environment/HASSIO_TOKEN"),
    ):
        if path.exists():
            val = path.read_text().strip()
            if val:
                return val
    return ""


def _ha_headers() -> dict:
    token = _supervisor_token()
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _ha_base() -> str:
    if os.environ.get("HA_BASE"):
        return os.environ["HA_BASE"].rstrip("/")
    return HA_BASE_DEFAULT


def ha_request(method: str, path: str, body=None, timeout: int = 20, raw: bool = False):
    url = _ha_base() + path
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=_ha_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            if raw:
                return resp.status, payload, resp.headers
            if not payload:
                return resp.status, None
            return resp.status, json.loads(payload)
    except urllib.error.HTTPError as e:
        raw_body = e.read()
        if raw:
            return e.code, raw_body, e.headers
        try:
            parsed = json.loads(raw_body) if raw_body else None
        except Exception:
            parsed = {"error": raw_body.decode(errors="replace")[:200]}
        return e.code, parsed


def publish_entity(entity_id: str, state: str, attrs: dict, timeout: int = 15) -> bool:
    payload = {"state": state, "attributes": attrs}
    status, _ = ha_request("POST", f"/api/states/{entity_id}", payload, timeout=timeout)
    if status not in (200, 201):
        log.warning("failed to publish %s HTTP %s", entity_id, status)
        return False
    return True


def snapshot_frame(entity: str, timeout: int = 8) -> Optional[np.ndarray]:
    status, body, _ = ha_request(
        "GET", f"/api/camera_proxy/{entity}", timeout=timeout, raw=True
    )
    if status != 200 or not body:
        return None
    arr = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def resolve_rtsp_url(opts: dict) -> Optional[str]:
    explicit = (opts.get("rtsp_url") or "").strip()
    if explicit:
        return explicit
    host = (opts.get("protect_host") or "").strip()
    alias = (opts.get("rtsp_alias") or "").strip()
    if host and alias:
        return f"rtsps://{host}:7441/{alias}?enableSrtp"
    return None


def rtsp_variants(url: str) -> list[str]:
    """Prefer plain RTSP (no TLS) for a faster cold start, then RTSPS."""
    out: list[str] = []
    try:
        p = urlparse(url)
        host = p.hostname
        alias = (p.path or "/").lstrip("/").split("?")[0]
        if host and alias:
            for cand in (
                f"rtsp://{host}:7447/{alias}",
                f"rtsps://{host}:7441/{alias}?enableSrtp",
                f"rtsps://{host}:7441/{alias}",
            ):
                if cand not in out:
                    out.append(cand)
    except Exception:
        pass
    if url not in out:
        out.append(url)
    return out


def open_capture(url: str) -> Optional[cv2.VideoCapture]:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|stimeout;3000000|max_delay;500000|"
        "fflags;nobuffer+discardcorrupt|flags;low_delay"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if cap is not None and cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if cap is not None:
        cap.release()
    return None


def grab_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        return None
    return frame


def _frame_mean(bgr) -> float:
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return 0.0
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def _is_night_frame(bgr) -> bool:
    return _frame_mean(bgr) < NIGHT_MEAN


def save_reference(det: LetterboxDetector) -> None:
    if det.ref_bgr is None:
        return
    path = REF_PATH_NIGHT if _is_night_frame(det.ref_bgr) else REF_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), det.ref_bgr)
    except Exception as exc:
        log.warning("could not persist reference: %s", type(exc).__name__)


def export_watches() -> None:
    """Copy watch logs and closed references onto /share for diagnosis."""
    try:
        SHARE_LETTERBOX.mkdir(parents=True, exist_ok=True)
        SHARE_WATCH_ROOT.mkdir(parents=True, exist_ok=True)
        for src, name in (
            (REF_PATH, "closed_reference.jpg"),
            (REF_PATH_NIGHT, "closed_reference_night.jpg"),
        ):
            if src.exists():
                shutil.copy2(src, SHARE_LETTERBOX / name)
        if not WATCH_LOG_ROOT.exists():
            log.info("no watch logs under /data/watches to export")
            return
        copied = 0
        for d in sorted(WATCH_LOG_ROOT.iterdir()):
            if not d.is_dir():
                continue
            target = SHARE_WATCH_ROOT / d.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(d, target)
            copied += 1
        log.info("exported %s watch log(s) to /share/letterbox/watches", copied)
    except Exception as exc:
        log.warning("export watches failed: %s", type(exc).__name__)


def _read_image(path: Path):
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img if img is not None and img.size else None


def _classified_refs() -> tuple[Optional[object], Optional[object]]:
    """Load stored and seeded stills, then bucket them by brightness."""
    night, day = None, None
    candidates = []
    for path, hinted in (
        (REF_PATH_NIGHT, "night"),
        (NIGHT_SEED, "night"),
        (REF_PATH, "day"),
        (Path("/share/letterbox/closed-day.jpg"), "day"),
    ):
        img = _read_image(path)
        if img is None:
            continue
        label = "night" if _is_night_frame(img) else "day"
        candidates.append((label, img, path.name))
    for label, img, name in candidates:
        if label == "night" and night is None:
            night = img
            log.info("night reference from %s mean=%.1f", name, _frame_mean(img))
        elif label == "day" and day is None:
            day = img
            log.info("day reference from %s mean=%.1f", name, _frame_mean(img))
    return day, night


def maybe_load_reference(det: LetterboxDetector, prefer_night: Optional[bool] = None) -> bool:
    day, night = _classified_refs()
    if prefer_night is True and night is not None:
        img, label = night, "night"
    elif prefer_night is False and day is not None:
        img, label = day, "day"
    elif prefer_night is True and day is not None:
        img, label = day, "day"
    elif day is not None:
        img, label = day, "day"
    elif night is not None:
        img, label = night, "night"
    else:
        return False
    det.set_reference(img)
    log.info("using %s closed reference %sx%s mean=%.1f", label, img.shape[1], img.shape[0], _frame_mean(img))
    return True


def select_reference_for_frame(det: LetterboxDetector, frame) -> None:
    """Swap in the night or day closed reference to match this frame's light."""
    day, night = _classified_refs()
    if _is_night_frame(frame) and night is not None:
        det.set_reference(night)
    elif (not _is_night_frame(frame)) and day is not None:
        det.set_reference(day)
    elif night is not None:
        det.set_reference(night)
    elif day is not None:
        det.set_reference(day)


class WatchController:
    """Thread-safe watch window: 60s from last trigger, extended while on."""

    def __init__(self, watch_seconds: int):
        self.watch_seconds = watch_seconds
        self._lock = threading.Lock()
        self.deadline = 0.0
        self.last_trigger = ""
        self.last_trigger_at = 0.0
        self.wake = threading.Event()
        # Test flags
        self.replay_path: Optional[str] = None
        self.replay_requested = False
        self.suppress_open_publish = False

    def extend(self, source: str, seconds: Optional[int] = None) -> None:
        dur = self.watch_seconds if seconds is None else seconds
        with self._lock:
            self.deadline = max(self.deadline, time.time() + dur)
            self.last_trigger = source
            self.last_trigger_at = time.time()
        self.wake.set()
        log.info("watch extended by %ss via %s", dur, source)

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.deadline - time.time())

    def active(self) -> bool:
        return self.remaining() > 0 or self.replay_requested

    def request_replay(self, path: str) -> None:
        with self._lock:
            self.replay_path = path
            self.replay_requested = True
            self.suppress_open_publish = True
        self.wake.set()
        log.info("replay requested")

    def take_replay(self) -> Optional[str]:
        with self._lock:
            if not self.replay_requested:
                return None
            self.replay_requested = False
            return self.replay_path


class WatchLog:
    """Save per-frame scores and a few JPEGs for each watch window."""

    def __init__(self):
        self.dir: Optional[Path] = None
        self._fh = None
        self._last_frame = None
        self._saved_open = 0
        self._saved_near = 0
        self._n = 0
        self._saved = {}
        self._best_area = -1
        self._best_area_frame = None
        self._best_sharp = -1.0
        self._best_sharp_frame = None

    def start(self, trigger: str) -> None:
        self.close({})
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = WATCH_LOG_ROOT / stamp
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            meta = {"started": stamp, "trigger": trigger}
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
            self._fh = open(self.dir / "scores.jsonl", "a")
        except Exception as exc:
            log.warning("watch log start failed: %s", type(exc).__name__)
            self.dir = None
            self._fh = None
        self._last_frame = None
        self._saved_open = 0
        self._saved_near = 0
        self._n = 0
        self._saved = {}
        self._best_area = -1
        self._best_area_frame = None
        self._best_sharp = -1.0
        self._best_sharp_frame = None
        self._prune()

    def record(self, frame, result: dict, source: str, t_ms: int) -> None:
        if self.dir is None:
            return
        self._n += 1
        if frame is not None:
            self._last_frame = frame
        rec = {
            "i": self._n,
            "t_ms": t_ms,
            "source": source,
            "open_frame": bool(result.get("open_frame")),
            "held": bool(result.get("state")),
            "area": int(result.get("area") or 0),
            "roi_frac": round(float(result.get("roi_frac") or 0.0), 3),
            "left_area": int(result.get("left_area") or 0),
            "sharp": round(float(result.get("sharp") or 0.0), 2),
            "peak": round(float(result.get("peak") or 0.0), 2),
            "positives": int(result.get("positives") or 0),
            "reasons": result.get("reasons") or [],
            "mean": round(_frame_mean(frame), 1) if frame is not None else None,
        }
        try:
            if self._fh:
                self._fh.write(json.dumps(rec) + chr(10))
                if self._n % 5 == 0:
                    self._fh.flush()
        except Exception:
            pass
        if frame is None:
            return
        if rec["area"] > self._best_area:
            self._best_area = rec["area"]
            self._best_area_frame = frame
        if rec["sharp"] > self._best_sharp:
            self._best_sharp = rec["sharp"]
            self._best_sharp_frame = frame
        tag = None
        if self._n == 1:
            tag = "first"
        elif rec["open_frame"] and self._saved_open < 8:
            self._saved_open += 1
            tag = f"open-{self._saved_open}"
        elif (
            (rec["area"] >= 25000 or rec["sharp"] >= 4.0)
            and not rec["open_frame"]
            and self._saved_near < WATCH_NEAR_KEEP
        ):
            self._saved_near += 1
            tag = f"near-{self._saved_near}"
        elif self._n % 15 == 0:
            tag = f"p-{self._n:04d}"
        if tag:
            self._write_jpg(f"{tag}.jpg", frame)
            self._write_jpg(f"f-{self._n:04d}.jpg", frame)
            self._saved[tag] = self._n

    def close(self, summary: dict) -> None:
        if self.dir is None:
            return
        if self._last_frame is not None:
            self._write_jpg("last.jpg", self._last_frame)
            self._saved.setdefault("last", self._n)
        if self._best_area_frame is not None:
            self._write_jpg("best-area.jpg", self._best_area_frame)
        if self._best_sharp_frame is not None:
            self._write_jpg("best-sharp.jpg", self._best_sharp_frame)
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass
        self._fh = None
        try:
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            (self.dir / "saved.json").write_text(json.dumps(self._saved, indent=2))
        except Exception:
            pass
        log.info("watch log saved %s frames=%s", self.dir.name, self._n)
        export_watches()
        self.dir = None
        self._last_frame = None
        self._best_area_frame = None
        self._best_sharp_frame = None

    def _write_jpg(self, name: str, frame) -> None:
        if self.dir is None or frame is None:
            return
        try:
            cv2.imwrite(str(self.dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        except Exception:
            pass

    def save_ref(self, frame, name: str = "closed-reference.jpg") -> None:
        self._write_jpg(name, frame)

    def _prune(self) -> None:
        try:
            if not WATCH_LOG_ROOT.exists():
                return
            dirs = sorted([d for d in WATCH_LOG_ROOT.iterdir() if d.is_dir()])
            def _size():
                return sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
            while len(dirs) > WATCH_LOG_KEEP or _size() > WATCH_LOG_MAX_BYTES:
                if not dirs:
                    break
                oldest = dirs.pop(0)
                for f in oldest.rglob("*"):
                    if f.is_file():
                        f.unlink()
                oldest.rmdir()
                log.info("pruned watch log %s", oldest.name)
        except Exception as exc:
            log.warning("watch log prune failed: %s", type(exc).__name__)


def event_should_trigger(eid: str, new_state: Optional[dict], old_state: Optional[dict]) -> Optional[str]:
    """Return a trigger label if this state change should start/extend a watch."""
    if not new_state:
        return None
    new = new_state.get("state")
    old = (old_state or {}).get("state")
    if new in (None, "unknown", "unavailable"):
        return None
    if eid == "binary_sensor.front_gate_is_dark":
        # on = the scene is dark. A falling edge means something just
        # lit the driveway (a person, a torch, or headlights).
        if old == "on" and new == "off":
            return eid + " (became light)"
        return None
    if eid.startswith("event."):
        if new == old:
            return None
        attrs = new_state.get("attributes") or {}
        et = attrs.get("event_type") or ""
        types = attrs.get("smart_detect_types") or []
        if not isinstance(types, list):
            types = [types]
        if eid.endswith("smart_detection"):
            if et in SMART_EVENT_TYPES or any(t in SMART_EVENT_TYPES for t in types):
                return f"{eid} ({et or ','.join(str(x) for x in types) or 'smart'})"
            return None
        # lock_access, package, motion_detection: any new event
        label = et or "event"
        return f"{eid} ({label})"
    # Level binary sensors (motion, person, ...)
    if new == "on":
        return eid
    return None


def _ws_url() -> str:
    return os.environ.get("HA_WS", "") or WS_URLS[0]


def run_websocket(triggers: set[str], watch: WatchController) -> None:
    """Subscribe to HA state changes and the test event. Reconnects on failure."""
    try:
        import asyncio
        import websockets
    except ImportError:
        log.error("python websockets is not available; triggers will not be received")
        return

    async def session(uri: str) -> None:
        token = _supervisor_token()
        async with websockets.connect(
            uri, open_timeout=20, ping_interval=None, max_size=10_000_000
        ) as ws:
            hello = json.loads(await ws.recv())
            if hello.get("type") != "auth_required":
                raise RuntimeError(f"unexpected hello {hello.get('type')}")
            await ws.send(json.dumps({"type": "auth", "access_token": token}))
            auth = json.loads(await ws.recv())
            if auth.get("type") != "auth_ok":
                raise RuntimeError(f"ws auth failed: {auth}")
            mid = 0

            async def call(typ, **payload):
                nonlocal mid
                mid += 1
                await ws.send(json.dumps({"id": mid, "type": typ, **payload}))
                while True:
                    ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    if ev.get("id") == mid:
                        return ev
                    _handle(ev)

            ev = await call("subscribe_events", event_type="state_changed")
            if not ev.get("success"):
                raise RuntimeError(f"subscribe state_changed failed: {ev}")
            ev = await call("subscribe_events", event_type=TEST_EVENT)
            if not ev.get("success"):
                raise RuntimeError(f"subscribe {TEST_EVENT} failed: {ev}")
            # If a level trigger is already on, start a window immediately.
            ev = await call("get_states")
            already = []
            for st in ev.get("result") or []:
                eid = st.get("entity_id")
                if eid in triggers and eid in LEVEL_TRIGGERS and st.get("state") == "on":
                    already.append(eid)
            if already:
                watch.extend("+".join(already) + " (already on)")
            log.info("websocket subscribed triggers=%s", sorted(triggers))
            while True:
                raw = await ws.recv()
                _handle(json.loads(raw))

    def _handle(msg: dict) -> None:
        if msg.get("type") != "event":
            return
        event = msg.get("event") or {}
        etype = event.get("event_type")
        data = event.get("data") or {}
        if etype == "state_changed":
            eid = data.get("entity_id")
            if eid not in triggers:
                return
            label = event_should_trigger(eid, data.get("new_state"), data.get("old_state"))
            if label:
                watch.extend(label)
        elif etype == TEST_EVENT:
            evd = data if "action" in data else (data.get("action") and data) or data
            # HA bus fires with the POST body as event data.
            action = evd.get("action") or evd.get("type") or "watch"
            if action == "replay":
                path = evd.get("path") or DEFAULT_CLIP
                watch.request_replay(path)
            else:
                # Safe simulated trigger: starts a live watch window but the
                # lid is expected closed, so letterbox_opened stays off.
                watch.extend("test:" + action)

    urls = [_ws_url(), *[u for u in WS_URLS if u != _ws_url()]]
    while True:
        last_err = None
        for uri in urls:
            try:
                log.info("connecting HA websocket")
                asyncio.run(session(uri))
            except Exception as exc:
                last_err = exc
                log.warning("websocket %s failed: %s", uri.split("/")[2] if "://" in uri else uri, type(exc).__name__)
                continue
        log.warning("websocket loop retry in 5s: %s", type(last_err).__name__ if last_err else "unknown")
        time.sleep(5)


def _load_replay_frames(path: str) -> tuple[list, float, Optional[str]]:
    """Load a video or a folder of JPEGs. Returns (frames, fps, error)."""
    target = Path(path)
    if target.is_dir():
        files = sorted(target.glob("f-*.jpg"))
        if not files:
            files = sorted(
                [f for f in target.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"} and f.is_file()],
                key=lambda f: f.name,
            )
        frames = []
        for f in files:
            img = cv2.imread(str(f))
            if img is not None and img.size:
                frames.append(img)
        if not frames:
            return [], 3.0, "could_not_open_clip"
        return frames, 3.0, None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 3.0, "could_not_open_clip"
    fps = cap.get(cv2.CAP_PROP_FPS) or 3.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return [], float(fps) or 3.0, "could_not_open_clip"
    return frames, float(fps) or 3.0, None


def replay_clip(path: str, det: LetterboxDetector, hold_frames: int) -> dict:
    """Run the detector on a local clip or JPEG folder. Never publishes opened=on."""
    clip_frames, fps, err = _load_replay_frames(path)
    if err:
        return {"ok": False, "error": err}
    # Fresh hold counter, keep the existing closed reference.
    worker = LetterboxDetector(
        hold_frames=hold_frames,
        min_area=det.min_area,
        sharp_min=det.sharp_min,
        max_roi_frac=det.max_roi_frac,
        sharp_max=det.sharp_max,
        sharp_soft=det.sharp_soft,
        min_area_soft=det.min_area_soft,
    )
    if det.has_reference() and det.ref_bgr is not None:
        worker.set_reference(det.ref_bgr)
    opened = []
    held = []
    first_open = None
    first_held = None
    i = 0
    t0 = time.time()
    boot = 0
    for frame in clip_frames[:15]:
        if worker.bootstrap_reference(frame):
            boot += 1
            if boot >= 3:
                break
    if boot:
        log.info("replay bootstrapped closed reference from first %s closed frames", boot)
    elif worker.has_reference():
        log.info("replay using live closed reference")
    else:
        log.warning("replay has no closed reference")
    for frame in clip_frames:
        i += 1
        result = worker.feed(frame)
        sec = (i - 1) / fps if fps else i
        if result["open_frame"]:
            opened.append(round(sec, 2))
            if first_open is None:
                first_open = sec
        if result["state"]:
            held.append(round(sec, 2))
            if first_held is None:
                first_held = sec
    elapsed = time.time() - t0
    summary = {
        "ok": True,
        "clip": Path(path).name,
        "frames": i,
        "fps": round(float(fps), 2),
        "open_frames": len(opened),
        "held_frames": len(held),
        "first_open_s": None if first_open is None else round(first_open, 2),
        "first_held_s": None if first_held is None else round(first_held, 2),
        "last_open_s": None if not opened else opened[-1],
        "last_held_s": None if not held else held[-1],
        "elapsed_s": round(elapsed, 2),
    }
    log.info("replay result %s", json.dumps(summary))
    return summary


def run() -> None:
    opts = _load_options()
    level_name = str(opts.get("log_level") or "info").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    camera_entity = opts.get("camera_entity") or "camera.front_gate_package_camera"
    hold_frames = int(opts.get("hold_frames") or 3)
    min_area = int(opts.get("min_area") or 50000)
    sharp_min = float(opts.get("sharp_min") or 5.0)
    watch_seconds = int(opts.get("watch_seconds") or 90)
    triggers = opts.get("trigger_entities") or list(DEFAULT_TRIGGERS)
    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    # Keep built-in event triggers even if an older options.json omits them.
    for extra in EXTRA_TRIGGERS:
        if extra not in triggers:
            triggers.append(extra)
    triggers_set = set(triggers)
    token = _supervisor_token()
    log.info(
        "starting detector camera=%s triggers=%s watch_seconds=%s hold_frames=%s "
        "min_area=%s sharp_min=%s ha_base=%s token_set=%s token_len=%s",
        camera_entity,
        triggers,
        watch_seconds,
        hold_frames,
        min_area,
        sharp_min,
        _ha_base(),
        bool(token),
        len(token),
    )

    det = LetterboxDetector(
        hold_frames=hold_frames, min_area=min_area, sharp_min=sharp_min
    )
    maybe_load_reference(det)
    export_watches()
    watch = WatchController(watch_seconds)
    watch_log = WatchLog()

    def common_attrs(extra: dict) -> dict:
        return {
            "device_class": "opening",
            "friendly_name": "Letterbox opened",
            "icon": "mdi:mailbox-up" if extra.get("state_on") else "mdi:mailbox",
            "watching": watch.active(),
            "watch_remaining": int(round(watch.remaining())),
            "last_trigger": watch.last_trigger or None,
            "triggers": triggers,
            **{k: v for k, v in extra.items() if k != "state_on"},
        }

    def publish_opened(state: str, extra: dict) -> bool:
        attrs = common_attrs({**extra, "state_on": state == "on"})
        return publish_entity(ENTITY_ID, state, attrs)

    def publish_watching(is_on: bool, extra: dict | None = None) -> None:
        payload = {
            "device_class": "running",
            "friendly_name": "Letterbox detector watching",
            "icon": "mdi:eye" if is_on else "mdi:eye-off",
            "last_trigger": watch.last_trigger or None,
            "watch_remaining": int(round(watch.remaining())),
            "triggers": triggers,
            **(extra or {}),
        }
        publish_entity(WATCHING_ID, "on" if is_on else "off", payload)

    publish_opened(
        "off",
        {"area": 0, "sharp": 0, "positives": 0, "source": "idle", "available": True, "frames": 0},
    )
    publish_watching(False)

    url = resolve_rtsp_url(opts)
    if url:
        log.info("RTSP source configured %s", _redact_url(url))
    else:
        log.warning("no RTSP URL in options; watch windows will use HA snapshots")

    ws_thread = threading.Thread(
        target=run_websocket, args=(triggers_set, watch), name="ha-ws", daemon=True
    )
    ws_thread.start()

    cap = None
    source = "idle"
    last_publish = 0.0
    last_state = "off"
    last_watch_pub = 0.0
    frames_seen = 0
    was_watching = False
    first_frame_ms: Optional[int] = None
    snapshot_ms: Optional[int] = None
    rtsp_ms: Optional[int] = None
    watch_started_at = 0.0
    ref_refresh_count = 0
    ref_refresh_target = 3
    last_trigger_poll = 0.0
    rtsp_thread = None
    rtsp_result: dict = {}

    def close_stream() -> None:
        nonlocal cap, source
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        source = "idle"

    def connect_rtsp_blocking() -> Optional[cv2.VideoCapture]:
        url_now = resolve_rtsp_url(opts)
        if not url_now:
            return None
        for cand in rtsp_variants(url_now):
            log.info("opening stream %s", _redact_url(cand))
            opened = open_capture(cand)
            if opened is not None:
                log.info("stream opened %s", _redact_url(cand))
                return opened
            log.warning("could not open %s", _redact_url(cand))
        return None

    def begin_watch() -> None:
        nonlocal was_watching, frames_seen, first_frame_ms, snapshot_ms, rtsp_ms
        nonlocal watch_started_at, ref_refresh_count, cap, source, last_state
        nonlocal rtsp_thread, rtsp_result
        was_watching = True
        frames_seen = 0
        first_frame_ms = None
        snapshot_ms = None
        rtsp_ms = None
        watch_started_at = time.time()
        ref_refresh_count = 0
        last_state = "off"
        det._positives = 0
        det.state = False
        close_stream()
        source = "starting"
        rtsp_result = {}
        watch_log.start(watch.last_trigger or "unknown")

        def _rtsp_worker():
            t0 = time.time()
            opened = connect_rtsp_blocking()
            rtsp_result["cap"] = opened
            rtsp_result["ms"] = int((time.time() - t0) * 1000)
            rtsp_result["done"] = True

        rtsp_thread = threading.Thread(target=_rtsp_worker, name="rtsp-open", daemon=True)
        rtsp_thread.start()
        publish_watching(True, {"phase": "connecting"})
        log.info("watch window started")

    def end_watch() -> None:
        nonlocal was_watching, cap, source
        close_stream()
        was_watching = False
        source = "idle"
        # Persist the reference if we refreshed it this window.
        if det.has_reference():
            save_reference(det)
        publish_opened(
            last_state if last_state == "on" else "off",
            {
                "area": 0,
                "sharp": 0,
                "positives": 0,
                "source": "idle",
                "available": True,
                "frames": frames_seen,
                "first_frame_ms": first_frame_ms,
                "snapshot_ms": snapshot_ms,
                "rtsp_ms": rtsp_ms,
            },
        )
        publish_watching(False, {"first_frame_ms": first_frame_ms, "snapshot_ms": snapshot_ms, "rtsp_ms": rtsp_ms})
        watch_log.close({
            "frames": frames_seen,
            "first_frame_ms": first_frame_ms,
            "snapshot_ms": snapshot_ms,
            "rtsp_ms": rtsp_ms,
            "trigger": watch.last_trigger,
            "last_state": last_state,
        })
        log.info(
            "watch window ended frames=%s first_frame_ms=%s snapshot_ms=%s rtsp_ms=%s",
            frames_seen,
            first_frame_ms,
            snapshot_ms,
            rtsp_ms,
        )

    def maybe_adopt_rtsp() -> None:
        nonlocal cap, source, rtsp_ms
        if cap is not None:
            return
        if rtsp_result.get("done") and rtsp_result.get("cap") is not None:
            cap = rtsp_result.pop("cap")
            rtsp_ms = rtsp_result.get("ms")
            source = "rtsp"
            log.info("using RTSP after %s ms", rtsp_ms)

    while True:
        # Handle a replay test without publishing opened=on.
        replay = watch.take_replay()
        if replay:
            if not Path(replay).exists():
                summary = {"ok": False, "error": "clip_not_found", "path_name": Path(replay).name}
                log.warning("replay clip missing")
            else:
                summary = replay_clip(replay, det, hold_frames)
            publish_entity(
                TEST_SENSOR_ID,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                {
                    "friendly_name": "Letterbox detector last test",
                    "icon": "mdi:test-tube",
                    **{k: v for k, v in summary.items() if k != "ok"},
                    "ok": summary.get("ok"),
                },
            )
            watch.suppress_open_publish = False
            continue

        if watch.active():
            if not was_watching:
                begin_watch()
            maybe_adopt_rtsp()

            t0 = time.time()
            frame = None
            frame_source = None
            # Prefer live RTSP once it is up; otherwise HA snapshots so a
            # lid opening ~3s after the person appears is not waiting on
            # the RTSP keyframe.
            if cap is not None:
                frame = grab_frame(cap)
                if frame is not None:
                    frame_source = "rtsp"
                    source = "rtsp"
            if frame is None:
                snap_t0 = time.time()
                frame = snapshot_frame(camera_entity)
                if frame is not None:
                    frame_source = "snapshot"
                    source = "snapshot"
                    if snapshot_ms is None:
                        snapshot_ms = int((time.time() - watch_started_at) * 1000)
                        log.info("first snapshot in %s ms", snapshot_ms)
                # Don't burn CPU if both fail.
                if frame is None:
                    time.sleep(0.15)

            if frame is not None:
                if first_frame_ms is None:
                    first_frame_ms = int((time.time() - watch_started_at) * 1000)
                    select_reference_for_frame(det, frame)
                    log.info(
                        "first frame in %s ms via %s (snapshot_ms=%s rtsp_ms=%s night=%s mean=%.1f)",
                        first_frame_ms,
                        frame_source,
                        snapshot_ms,
                        rtsp_ms,
                        _is_night_frame(frame),
                        _frame_mean(frame),
                    )
                    publish_watching(
                        True,
                        {
                            "phase": "watching",
                            "first_frame_ms": first_frame_ms,
                            "snapshot_ms": snapshot_ms,
                            "rtsp_ms": rtsp_ms,
                            "source": frame_source,
                        },
                    )
                frames_seen += 1

                # Refresh the closed reference from the first clearly
                # closed frames of this window, without poisoning it if
                # the lid is already open when we start.
                if ref_refresh_count < ref_refresh_target and det.has_reference():
                    score = det.score_frame(frame)
                    if det.should_learn_reference(score):
                        det.set_reference(frame)
                        ref_refresh_count += 1
                        if ref_refresh_count >= ref_refresh_target:
                            save_reference(det)
                            watch_log.save_ref(frame)
                            log.info("closed reference refreshed at start of watch")
                elif ref_refresh_count < ref_refresh_target and not det.has_reference():
                    if det.bootstrap_reference(frame):
                        ref_refresh_count += 1
                        if ref_refresh_count >= ref_refresh_target:
                            save_reference(det)
                            watch_log.save_ref(frame)
                            log.info("closed reference established at start of watch")

                result = det.feed(frame)
                watch_log.record(
                    frame,
                    result,
                    source,
                    int((time.time() - watch_started_at) * 1000),
                )
                # Log near-miss / open decisions so a future miss is diagnosable
                # even if JPEG saving failed.
                if result.get("open_frame") or result.get("state") or (
                    (result.get("area") or 0) >= 25000 or (result.get("sharp") or 0) >= 4.0
                ):
                    log.info(
                        "frame %s open=%s held=%s area=%s roi_frac=%s left=%s sharp=%.2f reasons=%s",
                        frames_seen,
                        result.get("open_frame"),
                        result.get("state"),
                        result.get("area"),
                        round(float(result.get("roi_frac") or 0.0), 3),
                        result.get("left_area"),
                        float(result.get("sharp") or 0.0),
                        result.get("reasons"),
                    )
                state = "on" if result["state"] else "off"
                if watch.suppress_open_publish:
                    state = "off"
                now = time.time()
                changed = state != last_state
                if changed or now - last_publish >= 15:
                    extra = {
                        "area": int(result.get("area") or 0),
                        "roi_frac": round(float(result.get("roi_frac") or 0.0), 3),
                        "left_area": int(result.get("left_area") or 0),
                        "sharp": round(float(result.get("sharp") or 0.0), 2),
                        "positives": int(result.get("positives") or 0),
                        "bbox": list(result["bbox"]) if result.get("bbox") else None,
                        "source": source,
                        "available": True,
                        "frames": frames_seen,
                        "first_frame_ms": first_frame_ms,
                        "snapshot_ms": snapshot_ms,
                        "rtsp_ms": rtsp_ms,
                    }
                    ok = publish_opened(state, extra)
                    last_publish = now
                    if ok:
                        if changed:
                            log.info(
                                "letterbox %s area=%s sharp=%.2f positives=%s",
                                state,
                                result["area"],
                                result["sharp"],
                                result["positives"],
                            )
                        last_state = state
                if now - last_watch_pub >= 15:
                    publish_watching(
                        True,
                        {
                            "phase": "watching",
                            "first_frame_ms": first_frame_ms,
                            "snapshot_ms": snapshot_ms,
                            "rtsp_ms": rtsp_ms,
                            "source": source,
                            "frames": frames_seen,
                        },
                    )
                    last_watch_pub = now

            # If a trigger is still held on, extend the window even when
            # HA does not emit further state_changed events.
            now_poll = time.time()
            if now_poll - last_trigger_poll >= 5:
                last_trigger_poll = now_poll
                still_on = []
                for eid in triggers:
                    if eid not in LEVEL_TRIGGERS:
                        continue
                    st, body = ha_request("GET", f"/api/states/{eid}", timeout=5)
                    if st == 200 and isinstance(body, dict) and body.get("state") == "on":
                        still_on.append(eid)
                if still_on:
                    watch.extend("+".join(still_on) + " (still on)")
            if source == "snapshot" or cap is None:
                # Pace snapshot pulls near 3 fps once we have a frame.
                elapsed = time.time() - t0
                time.sleep(max(0.0, (1 / 3) - elapsed))
        else:
            if was_watching:
                end_watch()
            now = time.time()
            if now - last_publish >= 30:
                publish_opened(
                    "off",
                    {
                        "area": 0,
                        "sharp": 0,
                        "positives": 0,
                        "source": "idle",
                        "available": True,
                        "frames": 0,
                        "first_frame_ms": first_frame_ms,
                        "snapshot_ms": snapshot_ms,
                        "rtsp_ms": rtsp_ms,
                    },
                )
                publish_watching(False)
                last_publish = now
            # Idle: block until a trigger or a short timeout for heartbeats.
            watch.wake.wait(timeout=5)
            watch.wake.clear()


if __name__ == "__main__":
    try:
        run()
    except Exception:
        logging.getLogger("letterbox").exception("detector crashed")
        raise
