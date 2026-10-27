# Sift — offline SMS with on-device AI spam filtering

Sift is a text-only SMS app for Android (built for the Pixel 8a, Android 14+). It becomes your default SMS app,
filters spam with a small transformer model that runs entirely on the phone, and learns from you: every time you
mark a message **Spam** or **Not spam**, the model retrains on the device. The app has no INTERNET permission, so
your messages and what it learns never leave the phone.

Sift only receives: it can't send or reply (it doesn't even hold the SEND_SMS permission). Links in messages are
plain text, never clickable; long-press a message to select and copy all or part of it.

## How the spam filter works

1. **Normalize.** Obfuscation tricks are undone before anything else: invisible characters are removed, and
   homoglyphs (Cyrillic "а" for "a"), fullwidth and "math bold" letters, and accents are folded to plain ASCII. The
   tricks themselves become signals.
2. **Understand.** A fine-tuned MiniLM transformer (22.7M parameters, int8, ~23 MB) turns the text into a 384-number
   embedding in about 10 ms, using ONNX Runtime on the CPU.
3. **Score.** A linear model combines the embedding with hashed word/character n-grams and 27 message features (link
   to a shortener, a risky domain ending, a brand look-alike or the brand's official site; phone numbers, money
   amounts, "reply STOP", OTP codes, capital letters, sender type…). Carrier-vetted short codes count as strong
   evidence of a legitimate sender, and international numbers count against.
   It also weighs the sender paired with each of the message's words and link domains, so a sender counts only together with what its message says: a bank's card-alert wording can read as legitimate from that bank while the bank's upsells and other senders' look-alikes don't. Every route of an Indian business sender is one sender (`AX-ICICIT-S`, `JD-ICICIT-T` → `ICICIT`), and its category suffix is ignored.
4. **Decide**, always per message and from that message's own content. Nothing is decided per sender: the same
   sender can send a promo you call spam and a passcode you need, and even a contact's message is judged on what it
   says. In order:
   1. a message nearly identical to one you've labeled gets your label ("not spam" matches also need the same sender, on any of its routes, or no link domains the labeled message didn't have);
   2. otherwise the model decides, at the sensitivity you chose (Conservative / Balanced / Aggressive).

   There are three outcomes: **Not spam**, **Possible spam** (stays in the inbox with a badge), and **Spam** (moved
   to the Spam tab, no notification). OTP-looking messages without links are never fully filtered. On any error the
   message is treated as not spam, so a message is never lost.
5. **Learn.** After installation, your marks are the only thing Sift learns from: no reading habits, no other rules.
   A mark applies to that one message. Each mark is stored on the phone as a training example (the text, its
   features and your label), so learning survives deleting the message.
   - Near-duplicates of the marked message follow your label immediately.
   - A few seconds later the model retrains on all your marks, anchored to the base model: lightly for word and
     character n-grams, strongly for each marked message's sender paired with its words and link domains. A sender
     only ever counts together with a message's wording, so "Not spam" on a bank's card alerts clears that bank's
     other card alerts (under any route prefix: `AX-ICICIT-S` and `JD-ICICIT-T` are one sender), not its upsells or
     another sender's similar text.
   - Then the open conversation is re-checked with the new model first ("Re-checking now…" shows until it's done),
     then every other message, old ones included; the progress shows in the notification shade.
   - Deleting an entry under Settings → Learning entries removes that mark: Sift forgets it, retrains, and judges the
     message again on its own.
   - The embedding weights, message-feature weights and sender-type signals stay fixed. That way a mark on one
     short-code promo doesn't shift every short-code message.
   - After an app update ships a new model, your marks are re-applied to it automatically.

## How well it works

Measured on data the model never trained on (`ml/scripts/s09_evaluate.py`). False-positive rate (FPR) means
legitimate messages wrongly moved to Spam.

**Your own inbox.** The oldest half (2,337 messages, May 2025 – Feb 2026) trained the model; the newest half
(2,337 messages, Feb – Oct 2026: 2,116 legit, 221 spam) is held out:

| Setting | Spam caught | FPR |
|---|---|---|
| Conservative | 35% | 0.09% |
| **Balanced (default)** | **50%** | **0.80%** |
| Aggressive | 87% | 2.6% |

That's before any marks. Replayed in arrival order through the phone's engine from a fresh install, by a user who
corrects every mistake, the held-out half takes 57 marks in 8½ months: 9 legitimate messages put in Spam and 48
promotions left in the inbox the first time a template appears. Errors in the later half drop from 87 without
learning to 34.

**Public data** (test split: 1,387 legit / 3,809 spam from UCI, Mishra, NCSU, IMC'25, the templates and Indian
datasets):

| Data | Setting | Spam caught | FPR |
|---|---|---|---|
| Test split | Conservative | 69.9% | 0% |
| | **Balanced** | **84.9%** | **0.07%** |
| | Aggressive | 94.9% | 1.87% |
| Scam campaigns that first appeared after the training cutoff | Balanced | 82% | — |
| 200 hand-written modern US messages | **Balanced** | **79%** | **0%** |

Balanced is calibrated for 0.5% FPR on validation messages that include your own, and your hardest legitimate
notices (dealer service reminders, invoices, insurance) push it up: on public data it lands far under that budget,
with lower recall. At exactly 0.5% FPR the test split's recall is 92.8% (new campaigns 93%, the modern set 88%).
The Indian SMS corpus with model-assisted labels (`cube_sms77k`) trains but stays out of thresholds and gates;
Balanced catches 81% of its spam at 0.09% FPR.

- No one-time-passcode (OTP) message was ever filtered.
- Undoing obfuscation (invisible characters, look-alike letters, fullwidth and "math" letters) gives identical model
  input, 100% of the time. Visible tricks (spaced letters, leetspeak, typos) cost 2.5 points of recall.

**Learning from feedback** (`s08_simulate.py`: 8 simulated streams, learning only from marks):
- **Scams the model missed.** After you mark one, 74% of its variants (new names, amounts, links, new numbers) are
  caught, vs 9% without learning.
- **Legitimate notices it flagged.** After "Not spam", all of that sender's similar notices stay in the inbox, vs
  25%.
- Errors on repeat campaigns halve. The base model already handles most of these streams, so the total cost
  (5 × false positives + misses) stays within 2% of not learning, and a user who only ever marks spam sees no extra
  false positives on held-out legit messages.

The settings are the `hash` and `snd` (sender × wording) priors that `ml/scripts/s08_simulate.py` writes to
`data/work/personalization.json` and `s11_package.py` ships in `model.json`: word and character n-grams retrain
lightly (`hash` 4), sender × wording strongly (`snd` 0.25).

**On the phone.** The bundle's JVM and on-device parity tests against the Python reference pass.

## Requirements

- Android Studio (its bundled JDK is used for command-line builds) with SDK Platform 37, Build-Tools, Platform-Tools,
  and the Emulator.
- For retraining the model (optional, the trained model is checked in): Python via [uv](https://docs.astral.sh/uv/)
  and an NVIDIA GPU (CPU works too, slowly).

## Build, test and install

```sh
source scripts/env.sh                     # JAVA_HOME = Studio's JBR, SDK adb first on PATH
./gradlew :spamcore:test                  # ML engine: parity with the Python reference, trainer, engine
./gradlew :app:testDebugUnitTest          # app logic
scripts/e2e.sh                            # end to end on the emulator, debug build (boot it with scripts/emulator.sh)
scripts/release_smoke.sh                  # the minified release build on the emulator: startup, rescan, SMS burst
./gradlew :app:installRelease             # install on the connected phone (USB debugging on)
```

Install over adb as above, not by tapping an APK file. Android 15+ restricts sideloaded files from becoming the
default SMS app.

### Before switching to Sift

- **Turn off RCS chats in Google Messages** (Settings → RCS chats), so people's messages fall back to SMS.
- Sift is SMS-only. **MMS** (group chats, pictures, and some long texts that carriers convert to MMS) isn't
  supported; you'll get a notification that one arrived.
- Open Sift and accept the "default SMS app" prompt.

### Switching back

Settings → Apps → Default apps → SMS app → Google Messages. Every message Sift received is in the system's
message store, so it shows up there too.

## Repository layout

| Path | What |
|---|---|
| `app/` | Android app (Compose Material 3, Room 3, WorkManager). The model bundle is in `src/main/assets/spam_model/`. |
| `spamcore/` | Pure Kotlin ML engine: normalizer, tokenizer, features, ONNX encoder, head, on-device trainer, memory, decision rules. Tested on the desktop JVM with the real model. |
| `ml/` | Python training and export pipeline. `ml/SPEC.md` is the exact Python ↔ Kotlin parity contract. |
| `scripts/` | `env.sh`, the emulator end-to-end test (`e2e.sh`) and the release-build smoke test (`release_smoke.sh`). |

## Retraining the model

```sh
cd ml
uv sync
uv run python -I scripts/s00_fetch.py        # public datasets → data/raw (licenses in MANIFEST.json)
uv run python -I scripts/s01_corpus.py       # normalize, dedupe, near-duplicate clusters (needs the inbox export)
uv run python -I scripts/s01b_rider_pool.py  # unlabeled pool for pseudo-labeling (not part of the corpus)
uv run python -I scripts/s02_split.py        # inbox time split, grouped splits, "new campaign" + inbox_eval holdouts
uv run python -I scripts/s03_finetune.py     # fine-tune MiniLM (GPU)
uv run python -I scripts/s04_export.py       # ONNX → fused → int8 per-channel → ORT format
uv run python -I scripts/s05_embed.py        # int8 embeddings, batch size 1 (as on the phone)
uv run python -I scripts/s06_heads.py        # base head
uv run python -I scripts/s07_calibrate.py    # thresholds by false-positive rate
uv run python -I scripts/s07b_sender_prior.py  # sender-type weights (short code, international…), modern dev split
uv run python -I scripts/s08_simulate.py     # simulated users → personalization defaults
uv run python -I scripts/s08_simulate.py --confirm   # re-check the chosen defaults on fresh streams
uv run python -I scripts/s09_evaluate.py     # acceptance gates (non-zero exit on failure)
uv run python -I scripts/s11_package.py      # → app/src/main/assets/spam_model/
uv run python -I scripts/s10_golden.py --prelim && uv run python -I scripts/s10_golden.py   # parity fixtures
uv run pytest
```

Training data:
- UCI SMS Spam Collection, CC BY 4.0
- Mishra & Soni 2022 SMS phishing, CC BY 4.0
- NCSU `wspr-ncsu/sms-phishing`, MIT
- UCL `reportsmishing/Smishing-Dataset-IMC25`, CC BY 4.0
- About 160 hand-written modern legitimate-message templates
- Indian corpora (`ml/data/raw/india/`): junioralive India Spam SMS, MIT; Ridham115 synthetic Indian scams, CC BY 4.0
  (training only); and, with no license stated, the `cunique.csv` corpus (~77k messages with senders), IIIT-Delhi
  SMSAssassin, and craftogrammer's transaction SMS

Plus the user's own phone inbox, labeled by the user: `ml/data/raw/inbox/inbox_v1.jsonl`, private, never committed
and not fetched by s00 (s01 fails without it). Its newest half is the `inbox_eval` split, used for evaluation only. Of
its oldest half, the newest 20% validates and the rest trains.

The inboxes of 93 Indian delivery riders (Kaggle, CC0) are kept outside the corpus as an unlabeled pool for
pseudo-labeling.

A separate set of 400 hand-written modern US messages is used for evaluation only.

## Known limitations

- Scam "openers" with no link or ask ("Hi, is this Sarah?") read exactly like real first messages. The base model
  lets them through; your marks and the first-contact signal handle them.
- A new promotion from a sender whose alerts you also get is judged on its wording alone the first time: Balanced
  leaves about half of first-seen promotions in the inbox. One Spam mark per template teaches it.
- Not in v1: MMS, search, blocking.
- Learned data is deliberately excluded from cloud backup. Clearing the app's data resets what it learned.
