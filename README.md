# Letterbox Detector

A Home Assistant add-on that watches a UniFi Protect package camera for
the letterbox lid opening. It stays idle until a person, face, package,
lock, or motion event, then scores frames from the package camera and
turns on `binary_sensor.letterbox_opened` when the lid is held open for
about a second (three frames at 3 fps).

Detection is deterministic: a closed-lid reference image plus the lid's
dark vertical residual, with guards so headlights and global lighting
washes are rejected. It is not an AI model.

It was written for a G6 Pro Entry package camera (channel 3, 1600x1200
at 3 fps). `binary_sensor.letterbox_detector_watching` shows whether a
watch window is active. Each window writes scores and a few frames under
`/data/watches/` and copies them to `/share/letterbox/watches/` so misses
can be diagnosed later.

## Install

This repository is a Home Assistant add-on repository.

1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**.
2. Open the three-dot menu and choose **Repositories**.
3. Add `https://github.com/bradzacher/ha-letterbox-detector`.
4. Find **Letterbox Detector** in the store, install it, configure the
   options below, and start the add-on.

The add-on uses the Home Assistant API. For a live stream, set `rtsp_url`
(or a UniFi Protect host that can be turned into a stream URL). Without
an RTSP URL it falls back to Home Assistant camera snapshots.

## Options

Defaults match the original Front Gate install.

- `camera_entity` — package camera used for snapshots (default
  `camera.front_gate_package_camera`)
- `protect_host` — UniFi Protect host, used when building a stream URL
- `rtsp_url` — explicit RTSP or RTSPS URL
- `hold_frames` — consecutive open frames required before the sensor
  turns on (default 3)
- `min_area`, `sharp_min` — detection thresholds
- `watch_seconds` — watch window length (default 90), extended while a
  level trigger stays on
- `trigger_entities` — entities that start a watch (person, motion,
  smart detection, lock access, package, and `is_dark`)
- `log_level`

A test event `letterbox_detector_test` with `action: watch` or
`action: replay` can start a window or replay a clip or JPEGs under
`/share/letterbox/`. Replay never turns `binary_sensor.letterbox_opened`
on.

## Changelog

### 1.4.1 (19 Nov 2026)

Fixes a delivery miss where a parcel held over the lid was learned as
the closed scene. The closed reference is only learned when the frame
barely differs from it in either direction. A tall, narrow dark residual
reaching the bottom of the frame (the lid seen edge-on behind a parcel)
counts as open.
