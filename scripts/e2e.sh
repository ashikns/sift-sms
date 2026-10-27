#!/usr/bin/env bash
# End-to-end checks on the emulator: real SMS delivery (adb emu sms send), spam placement, notifications,
# multipart, learning from feedback, Undo, the learning entries page, a rescan of all history, the re-check after a
# retrain, role loss + reconcile, Doze. Emulator only.
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
engine_field() { $A shell content query --uri "$DBG/engine" 2>/dev/null | tr -d '\r' | field "$1"; }

# Screen automation: dump the UI with uiautomator, then find nodes by text or content description.
ui_dump() { $A shell uiautomator dump /sdcard/sift-e2e-ui.xml >/dev/null 2>&1; $A shell cat /sdcard/sift-e2e-ui.xml >"$OUT/ui.xml"; }
# ui_find <label> [<anchor>]: center of the node whose text or content-desc is <label>; with <anchor>, the one
# vertically nearest the node containing <anchor> (e.g. the delete button in that list row).
ui_find() {
  python3 -I - "$OUT/ui.xml" "$1" "${2:-}" <<'PY'
import re, sys, xml.etree.ElementTree as ET
nodes = list(ET.parse(sys.argv[1]).getroot().iter("node"))
label = lambda n: (n.get("text") or "") + "\n" + (n.get("content-desc") or "")
def center(n):
    x1, y1, x2, y2 = map(int, re.findall(r"\d+", n.get("bounds")))
    return (x1 + x2) // 2, (y1 + y2) // 2
targets = [n for n in nodes if sys.argv[2] in (n.get("text"), n.get("content-desc"))]
if sys.argv[3]:
    anchor = next((n for n in nodes if sys.argv[3] in label(n)), None)
    if anchor is None: sys.exit(1)
    targets.sort(key=lambda n: abs(center(n)[1] - center(anchor)[1]))
if not targets: sys.exit(1)
print(*center(targets[0]))
PY
}
ui_tap() { local xy; ui_dump; xy=$(ui_find "$@") || return 1; $A shell input tap $xy; }
ui_has() { ui_dump; python3 -I -c 'import sys, xml.etree.ElementTree as ET; sys.exit(not any(sys.argv[2] in (n.get("text") or "") + (n.get("content-desc") or "") for n in ET.parse(sys.argv[1]).getroot().iter("node")))' "$OUT/ui.xml" "$1"; }

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
# Learning persists across runs; forget it so each run starts from the base model and "the model misses it" holds.
call forgetLearning >/dev/null

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
RES=$(call markSpam "$(msg_field "$(num 4)" id)")
MOVED=$(echo "$RES" | grep -oE 'moved=[0-9]+' | cut -d= -f2)
[[ "${MOVED:-0}" -ge 1 ]] && pass "marking spam moved $MOVED similar message(s)" || fail "markSpam result: $RES"
folder_is "$(num 4)" 1 && pass "the marked message's conversation moved to Spam" || fail "marked folder=$(conv_field "$(num 4)" folder)"
folder_is "$(num 5)" 1 && [[ "$(msg_field "$(num 5)" reason)" == "SIMILAR_TO_SPAM" ]] \
  && pass "near-duplicate moved with SIMILAR_TO_SPAM" || fail "family B reason=$(msg_field "$(num 5)" reason)"
send_sms "$(num 6)" "$FAM_C"
wait_until 15 folder_is "$(num 6)" 1 && pass "new variant from a new sender lands in Spam" || fail "family C folder=$(conv_field "$(num 6)" folder)"
[[ "$(msg_field "$(num 6)" reason)" == "SIMILAR_TO_SPAM" ]] && pass "reason SIMILAR_TO_SPAM" || fail "family C reason=$(msg_field "$(num 6)" reason)"
notified "$FAM_C" && fail "learned spam was notified" || pass "learned spam not notified"
call markNotSpam "$(msg_field "$(num 6)" id)" >/dev/null
folder_is "$(num 6)" 0 && pass "not spam moves it back to Inbox" || fail "after markNotSpam folder=$(conv_field "$(num 6)" folder)"

step "Undo"
BATCH=$(call markSpam "$(msg_field "$(num 1)" id)" | grep -oE 'batchId=[0-9]+' | cut -d= -f2)
folder_is "$(num 1)" 1 && pass "mark spam" || fail "mark spam folder=$(conv_field "$(num 1)" folder)"
call undo "$BATCH" >/dev/null
folder_is "$(num 1)" 0 && pass "undo restores Inbox" || fail "after undo folder=$(conv_field "$(num 1)" folder)"

step "Learning entries"
# Deleting an entry only takes effect, with one retrain, once the page is left.
idle
EX0=$(engine_field examples)
V0=$(engine_field modelVersion)
$A shell am start -W -f 0x10008000 -n "$PKG/.ui.MainActivity" >/dev/null # new, cleared task: start at the Inbox
sleep 1
ui_tap "Settings" && sleep 1 && ui_tap "Learning entries" && sleep 1
ui_has "$FAM_A" && pass "Settings → Learning entries lists the marks ($EX0)" || fail "the marked scam isn't listed"
ui_tap "Delete entry" "$FAM_A" && sleep 1
ui_has "$FAM_A" && fail "deleted entry still listed" || pass "deleted entry leaves the list"
sleep 4 # longer than the retrain debounce
[[ "$(engine_field examples)" == "$EX0" && "$(engine_field modelVersion)" == "$V0" ]] \
  && pass "nothing forgotten or retrained while the page is open" \
  || fail "engine changed while on the page: examples=$(engine_field examples) version=$(engine_field modelVersion)"
$A shell input keyevent KEYCODE_BACK
retrained() { [[ "$(engine_field examples)" == "$((EX0 - 1))" && "$(engine_field modelVersion)" == "$((V0 + 1))" ]]; }
wait_until 20 retrained && pass "leaving the page forgets the entry and retrains once" \
  || fail "after leaving: examples=$(engine_field examples) version=$(engine_field modelVersion) (was $EX0, $V0)"
wait_until 20 folder_is "$(num 5)" 0 && pass "the near-duplicate it had pulled into Spam returns to Inbox" \
  || fail "near-duplicate folder=$(conv_field "$(num 5)" folder) reason=$(msg_field "$(num 5)" reason)"
folder_is "$(num 4)" 1 && pass "the marked message keeps its Spam label" || fail "marked message folder=$(conv_field "$(num 4)" folder)"
$A shell input keyevent KEYCODE_BACK
$A shell input keyevent KEYCODE_HOME

step "Rescan all history"
# History older than the automatic backfill's 90 days: only a rescan classifies it.
seed_stat() { call seedStats | grep -oE "$1=[0-9]+" | cut -d= -f2; }
SEEDED=$(call seedHistory 800 | grep -oE 'seeded=[0-9]+' | cut -d= -f2)
[[ "${SEEDED:-0}" -eq 800 && "$(seed_stat classified)" -eq 0 ]] \
  && pass "seeded 800 unclassified messages from 100-400 days ago" || fail "seedHistory=$SEEDED classified=$(seed_stat classified)"
scan_dump() { $A shell dumpsys notification --noredact 2>/dev/null | tr -d '\r' | grep -A40 "pkg=$PKG"; }
scan_progress() { local d; d=$(scan_dump); [[ "$d" == *"Rescanning messages"* && "$d" == *android.progressMax* ]]; }
no_scan_progress() { ! scan_progress; }
scan_foreground() {
  local d; d=$($A shell dumpsys activity services "$PKG" 2>/dev/null | tr -d '\r')
  [[ "$d" == *SystemForegroundService* && "$d" == *"isForeground=true"* ]]
}
scan_done() { notified "Rescan complete"; }
# Started with the UI showing, as from Settings: Android only lets a visible app start a foreground service.
$A shell am start -W -n "$PKG/.ui.MainActivity" >/dev/null
call rescan >/dev/null
wait_until 15 scan_progress && pass "progress notification while rescanning" || fail "no rescan progress notification"
scan_foreground && pass "rescan runs as a foreground service" || fail "rescan isn't running in the foreground"
wait_until 180 scan_done && pass "summary notification when done" || fail "no rescan summary notification"
notified "moved to Spam" && pass "summary counts conversations moved to Spam" \
  || fail "summary: $(scan_dump | grep -m1 -oE 'android.bigText=[^)]*')"
wait_until 10 no_scan_progress && pass "progress notification removed" || fail "progress notification left behind"
[[ "$(seed_stat classified)" -eq 800 ]] && pass "rescan classified all 800 old messages" || fail "classified $(seed_stat classified) of 800"
SCAMS=$(seed_stat scams)
FLAGGED=$(seed_stat scamsFlagged)
WRONG=$(seed_stat ordinaryFlagged)
(( FLAGGED * 10 >= SCAMS * 9 )) && pass "old scams now in Spam ($FLAGGED of $SCAMS)" || fail "old scams in Spam: $FLAGGED of $SCAMS"
(( WRONG * 10 <= 800 - SCAMS )) && pass "old ordinary texts mostly stay out of Spam ($WRONG flagged)" || fail "ordinary texts in Spam: $WRONG"
$A shell input keyevent KEYCODE_HOME

step "Re-check after a retrain"
# Messages this old are only re-scored by the re-check that follows every retrain. Marking one old seeded parcel scam
# "not spam" makes its copies from other senders near-duplicates of a not-spam message.
rechecks() { local d; d=$($A logcat -d -s WM-WorkerWrapper:I 2>/dev/null); echo "$d" | grep -F RecheckWorker | grep -cF SUCCESS; }
recheck_progress() { local d; d=$(scan_dump); [[ "$d" == *"Re-checking messages"* ]]; }
R0=$(rechecks)
F0=$(seed_stat scamsFlagged)
# (Matched here: the device shell strips quotes inside --where, and a "+" number then compares as a number.)
SEED0=$($A shell content query --uri content://sms --projection _id:address 2>/dev/null | tr -d '\r' | grep -E "address=\+15550190000$" | field _id)
call markNotSpam "$SEED0" >/dev/null
wait_until 30 recheck_progress && pass "re-check progress notification" || fail "no re-check progress notification"
recheck_ran() { [[ "$(rechecks)" -gt "$R0" ]]; }
wait_until 120 recheck_ran && pass "a re-check of all history ran after the retrain" || fail "no re-check after the mark"
F1=$(seed_stat scamsFlagged)
(( F1 <= F0 - 100 )) && pass "the re-check moved old copies out of Spam ($F0 -> $F1 flagged)" || fail "old copies still flagged: $F0 -> $F1"
$A shell input keyevent KEYCODE_HOME

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
