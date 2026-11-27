# Letterbox Detector

Local Home Assistant add-on that sits idle until a Front Gate activity
signal, then watches the G6 Pro Entry package camera (channel 3,
1600x1200 at 3 fps) and sets `binary_sensor.letterbox_opened` when the
letterbox lid is held open for at least ~1 second.

Default triggers (Supervisor websocket, no extra HA YAML):

- `binary_sensor.front_gate_person_detected`
- `binary_sensor.front_gate_motion` (the Protect motion *enable*
  switch is currently off, so this sensor may never fire)
- `event.front_gate_smart_detection` of type person, face, or package
- `event.front_gate_lock_access`
- `event.front_gate_package`
- `binary_sensor.front_gate_is_dark` turning off (the scene just got
  brighter — a fallback when a face or person is not flagged)

There is no package-camera-specific motion entity. A watch window lasts
`watch_seconds` (default 90) and is extended while a level trigger
stays on.

`binary_sensor.letterbox_detector_watching` shows whether a watch
window is active.

Each watch window writes scores and a few frames under `/data/watches/`
(rotated, size-capped) and copies them to `/share/letterbox/watches/`
so misses can be diagnosed from File Editor or SSH.

Detection is deterministic (closed-reference residual plus the lid's
straight dark vertical bar, with a cap so global lighting / headlights
are rejected). It is not an AI model.

Night / infrared openings should now start a watch when a face, lock
access, or is_dark flicker is seen. A real night opening has not yet
been confirmed end-to-end.

A test event `letterbox_detector_test` with `action: watch` or
`action: replay` can start a window or replay a clip or a folder of
JPEGs under `/share/letterbox/` without being a live notify test of
the lid itself. Replay never turns `binary_sensor.letterbox_opened` on.

v1.4.0 raises the residual localization cap to 75% of the lid region
and accepts a slightly soft vertical-bar score when the residual is
large and still local (a mid-swing lid). Headlight washes that fill
the whole region are still rejected. A parcel left in the box is
allowed to become the new closed reference.

v1.4.1 (19 Nov 2026) fixes the 13:27 ACDT delivery miss. The closed
reference is only learned when the frame barely differs from it in
either direction (a parcel held over the lid no longer becomes the
"closed" scene), and a tall, narrow dark residual reaching the bottom
of the frame (the lid seen edge-on behind a parcel) counts as open.
