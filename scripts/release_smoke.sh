#!/usr/bin/env bash
# Smoke test of the minified release build on the emulator. e2e.sh drives the debug build, which R8 never touches, so it can't see
# release-only breakage: R8 once deleted the field that keeps the memory-mapped encoder alive, and ONNX Runtime then read freed memory
# (SIGSEGV) after the next GC. This launches the release app, runs Settings → Rescan messages, moves the app to the background (ART
# compacts its heap) and delivers a burst of SMS that all need the encoder. Fails on any crash. Emulator only.
#
#   scripts/release_smoke.sh
#   SKIP_INSTALL=1 scripts/release_smoke.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

SERIAL=${ANDROID_SERIAL:-emulator-5554}
case "$SERIAL" in
  emulator-*) ;;
  *) echo "refusing to run against $SERIAL: release_smoke.sh only drives emulators" >&2; exit 2 ;;
esac
A="adb -s $SERIAL"
PKG=dev.ashik.sift
PASSED=0
FAILED=0
RUN=$(printf '%03d' $((RANDOM % 1000)))

pass() { echo "  PASS  $*"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL  $*"; FAILED=$((FAILED + 1)); }
step() { echo; echo "== $*"; }

pid() { $A shell pidof "$PKG" | tr -d '\r'; }
same_process() { local p; p=$(pid); [[ -n "$p" && "$p" == "$PID" ]]; }
sms_count() { $A shell content query --uri content://sms --projection _id | grep -c "^Row:"; }
# Capture first, then match: `grep -q` exiting early would SIGPIPE adb and fail the pipeline under pipefail.
notified() { local d; d=$($A shell dumpsys notification --noredact 2>/dev/null | tr -d '\r' | grep -A60 "pkg=$PKG"); [[ "$d" == *"$1"* ]]; }
# tap <label>: taps the center of the first node whose text or content-desc is exactly <label>.
tap() {
  local b
  $A shell uiautomator dump /sdcard/sift-ui.xml >/dev/null 2>&1
  b=$($A shell cat /sdcard/sift-ui.xml | grep -oE "<node [^>]*(text|content-desc)=\"$1\"[^>]*>" | head -1 \
    | grep -oE 'bounds="\[[0-9]+,[0-9]+\]\[[0-9]+,[0-9]+\]"' | grep -oE '[0-9]+' | tr '\n' ' ')
  [[ -n "$b" ]] || return 1
  set -- $b
  $A shell input tap $(( ($1 + $3) / 2 )) $(( ($2 + $4) / 2 ))
}

# wait_until <seconds> <command...>: polls until the command succeeds.
wait_until() {
  local secs=$1; shift
  for _ in $(seq 1 $((secs * 4))); do
    if "$@"; then return 0; fi
    sleep 0.25
  done
  return 1
}

step "Setup ($SERIAL, run $RUN)"
if [[ -z "${SKIP_INSTALL:-}" ]]; then
  ANDROID_SERIAL=$SERIAL ./gradlew --console=plain -q :app:installRelease || { echo "install failed"; exit 1; }
fi
FLAGS=$($A shell dumpsys package "$PKG" | grep -m1 -E '^\s*pkgFlags=')
[[ -n "$FLAGS" && "$FLAGS" != *DEBUGGABLE* ]] && pass "release build installed" \
  || fail "the installed build isn't the release build; drop SKIP_INSTALL"
$A shell cmd role add-role-holder android.app.role.SMS "$PKG" >/dev/null
[[ "$($A shell cmd role get-role-holders android.app.role.SMS | tr -d '\r')" == "$PKG" ]] && pass "Sift holds the SMS role" || fail "role not held"
# A fresh process, so startup creates the encoder again. Force-stopping also cancels Sift's notifications, keeping it under
# Android's 50-per-app limit across runs.
$A shell am force-stop "$PKG"
$A logcat -b all -c

step "Startup (reconcile + backfill)"
$A shell am start -W -n "$PKG/.ui.MainActivity" >/dev/null
sleep 10
PID=$(pid)
[[ -n "$PID" ]] && pass "running 10 s after launch" || { fail "not running 10 s after launch"; $A logcat -b crash -d | head -40; exit 1; }

step "Rescan"
$A logcat -b main -c  # just main: the crash and system buffers must keep the whole run
tap "Settings" && sleep 1 && tap "Rescan messages" && pass "tapped Settings → Rescan messages" || fail "couldn't reach Rescan messages"
rescanned() { $A logcat -d -s WM-WorkerWrapper:I | grep -F "Worker result SUCCESS" | grep -F BackfillClassifyWorker >/dev/null; }
wait_until 60 rescanned && pass "rescan finished" || fail "no successful BackfillClassifyWorker run"
same_process && pass "survived the rescan" || fail "process died during the rescan"

step "Background + SMS burst"
$A shell input keyevent KEYCODE_HOME
sleep 20  # ART compacts the heap of an app that went to the background
BEFORE=$(sms_count)
HAM=("Your Chase verification code is 48213. Do not share it with anyone."
     "CVS Pharmacy: your prescription is ready for pickup at 123 Main St."
     "Delta: flight DL1234 to SEA now departs 6:45 PM from gate B12.")
SPAM=("USPS: your parcel is held due to an incomplete address. Update within 24h: https://usps-redeliver.top/t"
      "Toll Services: unpaid toll balance of \$6.99. Avoid a late fee: https://ezpass-billing.click/pay"
      "Congrats! You won a \$500 Walmart gift card. Claim now: https://bit.ly/3xYz9")
N=24
tag() { printf '#%s-%02d' "$RUN" "$1"; }  # fixed width, so no tag is a prefix of another
for i in $(seq 0 $((N - 1))); do
  if (( i % 2 )); then text=${HAM[$((i / 2 % 3))]}; else text=${SPAM[$((i / 2 % 3))]}; fi
  $A emu sms send "650${RUN}$(printf '%04d' "$i")" "$text $(tag "$i")" >/dev/null
  sleep 0.7
done
stored() { [[ $(sms_count) -ge $((BEFORE + N)) ]]; }
wait_until 30 stored && pass "all $N messages stored" || fail "stored $(( $(sms_count) - BEFORE )) of $N messages"
sleep 5
same_process && pass "survived the burst" || fail "process died during the burst"
missing=0; leaked=0
for i in $(seq 0 $((N - 1))); do
  if (( i % 2 )); then notified "$(tag "$i")" || missing=$((missing + 1)); else notified "$(tag "$i")" && leaked=$((leaked + 1)); fi
done
[[ $missing -eq 0 ]] && pass "every ham message notified" || fail "$missing ham messages didn't notify"
[[ $leaked -eq 0 ]] && pass "no spam message notified" || fail "$leaked spam messages notified"

step "Logs"
CRASH=$($A logcat -b crash -d | tr -d '\r')
if [[ "$CRASH" == *"$PKG"* ]]; then
  fail "crash in $PKG"
  echo "$CRASH" | head -40
else
  pass "no crashes"
fi
[[ "$($A logcat -d | tr -d '\r')" == *"ANR in $PKG"* ]] && fail "ANR in $PKG" || pass "no ANRs"

echo
echo "release smoke: $PASSED passed, $FAILED failed"
[[ $FAILED -eq 0 ]]
