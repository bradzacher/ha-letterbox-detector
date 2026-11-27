#!/usr/bin/with-contenv bash
set -euo pipefail
cd /usr/src/letterbox
exec python3 -u main.py
