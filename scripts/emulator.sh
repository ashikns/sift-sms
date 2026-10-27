#!/usr/bin/env bash
# Boots the sift_api37 AVD headless (KVM) and waits for boot to complete. Prints the serial.
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=${SIFT_AVD:-sift_api37}
PORT=${SIFT_EMU_PORT:-5554}
SERIAL="emulator-$PORT"
"$(dirname "$0")/avd.sh" >/dev/null
if ! adb devices | grep -q "^$SERIAL"; then
  mkdir -p "$(dirname "$0")/out"
  nohup emulator -avd "$NAME" -port "$PORT" -no-window -no-audio -no-boot-anim -no-snapshot-save \
    -gpu swiftshader_indirect >"$(dirname "$0")/out/emulator.log" 2>&1 &
fi
adb -s "$SERIAL" wait-for-device
for _ in $(seq 1 180); do
  [[ "$(adb -s "$SERIAL" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]] && break
  sleep 2
done
adb -s "$SERIAL" shell settings put global window_animation_scale 0 >/dev/null 2>&1 || true
adb -s "$SERIAL" shell settings put global transition_animation_scale 0 >/dev/null 2>&1 || true
adb -s "$SERIAL" shell settings put global animator_duration_scale 0 >/dev/null 2>&1 || true
echo "$SERIAL"
