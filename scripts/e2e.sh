#!/usr/bin/env bash
# End-to-end checks on the emulator: real SMS delivery (adb emu sms send), spam placement, notifications,
# multipart, learning from feedback, Undo, role loss + reconcile, Doze. Emulator only.
#
#   scripts/emulator.sh          # boot the AVD (once)
#   scripts/e2e.sh               # build + install the debug app, run every scenario
#   SKIP_INSTALL=1 scripts/e2e.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

SERIAL=${ANDROID_SERIAL:-emulator-5554}
case "$SERIAL" in
  emulator-*) ;;
  *) echo "refusing to run against $SERIAL: e2e.sh only drives emulators" >&2; exit 2 ;;
esac
A="adb -s $SERIAL"
PKG=dev.ashik.sift
DBG=content://dev.ashik.sift.debug
OUT=scripts/out
mkdir -p "$OUT"
PASSED=0
FAILED=0
RUN=$(printf '%03d' $((RANDOM % 1000)))
num() { echo "650${RUN}$(printf '%04d' "$1")"; }

pass() { echo "  PASS  $*"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL  $*"; FAILED=$((FAILED + 1)); }
step() { echo; echo "== $*"; }

send_sms() { $A emu sms send "$1" "$2" >/dev/null; }
idle() { $A shell content call --uri "$DBG" --method awaitIdle >/dev/null 2>&1; }
conversations() { $A shell content query --uri "$DBG/conversations" 2>/dev/null | tr -d '\r'; }
messages() { $A shell content query --uri "$DBG/messages" 2>/dev/null | tr -d '\r'; }
field() { grep -oE "(^|[ ,])$1=[^,]*" | head -1 | sed -E "s/^[ ,]?$1=//"; }
conv_field() { conversations | grep -F "address=$1," | head -1 | field "$2"; }
msg_field() { messages | grep -F "address=$1," | head -1 | field "$2"; }
# Capture first, then match: `grep -q` exiting early would SIGPIPE adb and fail the pipeline under pipefail.
notified() { local d; d=$($A shell dumpsys notification --noredact 2>/dev/null | tr -d '\r' | grep -A60 "pkg=$PKG"); [[ "$d" == *"$1"* ]]; }
call() { $A shell content call --uri "$DBG" --method "$1" ${2:+--arg "$2"} 2>/dev/null | tr -d '\r'; }

# wait_until <seconds> <command...>: polls until the command succeeds.
wait_until() {
  local secs=$1; shift
  for _ in $(seq 1 $((secs * 4))); do
    if "$@"; then return 0; fi
    sleep 0.25
  done
  return 1
}
folder_is() { [[ "$(conv_field "$1" folder)" == "$2" ]]; }
has_message() { local m; m=$(messages); [[ "$m" == *"$1"* ]]; }

step "Setup ($SERIAL, run $RUN)"
if [[ -z "${SKIP_INSTALL:-}" ]]; then
  ANDROID_SERIAL=$SERIAL ./gradlew --console=plain -q :app:installDebug || { echo "install failed"; exit 1; }
fi
$A shell cmd role add-role-holder android.app.role.SMS "$PKG" >/dev/null
[[ "$($A shell cmd role get-role-holders android.app.role.SMS | tr -d '\r')" == "$PKG" ]] && pass "Sift holds the SMS role" || fail "role not held"
$A logcat -c
$A shell am start -W -n "$PKG/.ui.MainActivity" >/dev/null
sleep 2
$A shell input keyevent KEYCODE_HOME
idle
# Learning persists across runs; start each run from the base model so "the model misses it" holds.
call resetLearning >/dev/null

step "Incoming ham and spam"
HAM="Dinner at 7? I booked the usual place $RUN"
SPAM="WINNER! Claim your 1000 USD gift card now at http://bit.ly/x9z $RUN"
send_sms "$(num 1)" "$HAM"
send_sms "$(num 2)" "$SPAM"
wait_until 15 folder_is "$(num 1)" 0 && pass "ham landed in Inbox" || fail "ham folder=$(conv_field "$(num 1)" folder)"
wait_until 15 folder_is "$(num 2)" 1 && pass "spam landed in Spam" || fail "spam folder=$(conv_field "$(num 2)" folder)"
wait_until 5 notified "$HAM" && pass "ham notified" || fail "no notification for ham"
notified "$SPAM" && fail "spam was notified" || pass "spam not notified"

step "Multipart"
LONG=$(printf 'Long message %s part ' "$RUN"; for i in $(seq 1 40); do printf 'word%02d ' "$i"; done)
LONG=${LONG:0:400}
send_sms "$(num 3)" "$LONG"
if wait_until 15 has_message "Long message $RUN"; then
  BODY=$($A shell content query --uri content://sms --projection body --where "address='$(num 3)'" | tr -d '\r')
  COUNT=$(echo "$BODY" | grep -c '^Row:')
  [[ "$COUNT" == "1" && "$BODY" == *"${LONG: -20}"* ]] && pass "400-char message stored as one message" || fail "multipart stored as $COUNT rows"
else
  fail "multipart message never arrived"
fi

step "Learning from feedback"
# A "pig butchering" opener: no link or ask yet, so the base model misses it even from a first-contact number
# (p < 0.02) and it stays in the Inbox. Variants are near-duplicates (n-gram cosine 0.94 / 0.86).
FAM_A="Good morning! It was nice meeting you at the conference in Denver. Are you free for coffee next week? $RUN"
FAM_B="Good morning! It was so nice meeting you at the conference in Denver. Are you free for coffee next week? $RUN"
FAM_C="Good morning, it was nice meeting you at the conference in Denver! Are you free for coffee this week? $RUN"
send_sms "$(num 4)" "$FAM_A"
send_sms "$(num 5)" "$FAM_B"
wait_until 15 folder_is "$(num 5)" 0 && pass "new scam family starts in Inbox (model misses it)" || fail "family B folder=$(conv_field "$(num 5)" folder)"
idle
T4=$(conv_field "$(num 4)" threadId)
RES=$(call markSpam "thread:$T4")
MOVED=$(echo "$RES" | grep -oE 'moved=[0-9]+' | cut -d= -f2)
[[ "${MOVED:-0}" -ge 1 ]] && pass "marking spam moved $MOVED similar message(s)" || fail "markSpam result: $RES"
folder_is "$(num 4)" 1 && pass "marked conversation moved to Spam" || fail "marked folder=$(conv_field "$(num 4)" folder)"
folder_is "$(num 5)" 1 && [[ "$(msg_field "$(num 5)" reason)" == "SIMILAR_TO_SPAM" ]] \
  && pass "near-duplicate moved with SIMILAR_TO_SPAM" || fail "family B reason=$(msg_field "$(num 5)" reason)"
send_sms "$(num 6)" "$FAM_C"
wait_until 15 folder_is "$(num 6)" 1 && pass "new variant from a new sender lands in Spam" || fail "family C folder=$(conv_field "$(num 6)" folder)"
[[ "$(msg_field "$(num 6)" reason)" == "SIMILAR_TO_SPAM" ]] && pass "reason SIMILAR_TO_SPAM" || fail "family C reason=$(msg_field "$(num 6)" reason)"
notified "$FAM_C" && fail "learned spam was notified" || pass "learned spam not notified"
T6=$(conv_field "$(num 6)" threadId)
call markNotSpam "thread:$T6" >/dev/null
folder_is "$(num 6)" 0 && pass "not spam moves it back to Inbox" || fail "after markNotSpam folder=$(conv_field "$(num 6)" folder)"

step "Undo"
T1=$(conv_field "$(num 1)" threadId)
BATCH=$(call markSpam "thread:$T1" | grep -oE 'batchId=[0-9]+' | cut -d= -f2)
folder_is "$(num 1)" 1 && pass "mark spam" || fail "mark spam folder=$(conv_field "$(num 1)" folder)"
call undo "$BATCH" >/dev/null
folder_is "$(num 1)" 0 && pass "undo restores Inbox" || fail "after undo folder=$(conv_field "$(num 1)" folder)"

step "Role loss and reconcile"
GM=com.google.android.apps.messaging
if $A shell pm path "$GM" >/dev/null 2>&1; then
  $A shell cmd role add-role-holder android.app.role.SMS "$GM" >/dev/null
  # Google Messages only stores SMS once it has been launched; bring it up, then deliver.
  $A shell monkey -p "$GM" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
  sleep 5
  OUTSIDE="Sent while Google Messages was default $RUN"
  send_sms "$(num 7)" "$OUTSIDE"
  sleep 10
  $A shell input keyevent KEYCODE_HOME
  $A shell cmd role add-role-holder android.app.role.SMS "$PKG" >/dev/null
  sleep 2
  $A shell am start -W -n "$PKG/.ui.MainActivity" >/dev/null
  if wait_until 30 has_message "$OUTSIDE"; then pass "reconcile imported a message received while another app was default"; else
    if [[ "$($A shell content query --uri content://sms --projection body | tr -d '\r')" == *"$OUTSIDE"* ]]; then
      fail "message is in the provider but wasn't reconciled"
    else
      echo "  SKIP  Google Messages didn't store the message (first-run state); reconcile not exercised"
    fi
  fi
  $A shell input keyevent KEYCODE_HOME
else
  echo "  SKIP  Google Messages not installed"
fi
[[ "$($A shell cmd role get-role-holders android.app.role.SMS | tr -d '\r')" == "$PKG" ]] && pass "role restored" || fail "role not restored"

step "Doze"
$A shell dumpsys deviceidle force-idle >/dev/null
DOZE="Doze check from Alex $RUN"
send_sms "$(num 8)" "$DOZE"
wait_until 5 notified "$DOZE" && pass "notification within 5 s while in Doze" || fail "no notification in Doze"
$A shell dumpsys deviceidle unforce >/dev/null

step "Screenshots"
$A shell am start -W -n "$PKG/.ui.MainActivity" >/dev/null
sleep 2
$A exec-out screencap -p >"$OUT/e2e_inbox.png" && echo "  saved $OUT/e2e_inbox.png"
$A shell input keyevent KEYCODE_HOME

step "Crash / ANR scan"
LOG=$($A logcat -d 2>/dev/null)
FATALS=$(echo "$LOG" | grep -A2 "FATAL EXCEPTION")
if [[ "$FATALS" == *"Process: $PKG"* ]]; then
  fail "FATAL EXCEPTION in $PKG"
  echo "$LOG" | grep -A15 "FATAL EXCEPTION" | head -30
else
  pass "no crashes"
fi
[[ "$LOG" == *"ANR in $PKG"* ]] && fail "ANR in $PKG" || pass "no ANRs"

echo
echo "e2e: $PASSED passed, $FAILED failed"
[[ $FAILED -eq 0 ]]
