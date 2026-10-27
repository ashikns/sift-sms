# Sift — offline SMS with on-device AI spam filtering

Sift is a text-only SMS app for Android (built for the Pixel 8a, Android 14+). It becomes your default SMS app,
filters spam with a small transformer model that runs entirely on the phone, and learns from you: every time you
mark a message **Spam** or **Not spam**, the model retrains on the device. The app has no INTERNET permission, so
your messages and what it learns never leave the phone.

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
4. **Decide**, always per message and from that message's own content. Nothing is decided per sender: the same
   sender can send a promo you call spam and a passcode you need, and even a contact's message is judged on what it
   says. In order:
   1. a message nearly identical to one you've labeled gets your label ("not spam" matches also need the same
      sender, or no link domains the labeled message didn't have);
   2. otherwise the model decides, at the sensitivity you chose (Conservative / Balanced / Aggressive).

   There are three outcomes: **Not spam**, **Possible spam** (stays in the inbox with a badge), and **Spam** (moved
   to the Spam tab, no notification). OTP-looking messages without links are never fully filtered. On any error the
   message is treated as not spam, so a message is never lost.
5. **Learn.** After installation, your marks are the only thing Sift learns from: no replies, no reading habits, no
   other rules. A mark applies to that one message; marking a conversation means its latest incoming message. Each
   mark is stored on the phone as a training example (the text, its features and your label), so learning survives
   deleting the message.
   - Near-duplicates of the marked message follow your label immediately.
   - A few seconds later the model's word and character n-gram weights retrain lightly on all your marks, anchored
     to the base model. Only messages that share the marked messages' wording are affected.
   - The embedding weights, message-feature weights and sender-type signals stay fixed. That way a mark on one
     short-code promo doesn't shift every short-code message.
   - After an app update ships a new model, your marks are re-applied to it automatically.

## How well it works

Measured on held-out data the model never trained on (`ml/scripts/s09_evaluate.py`). False-positive rate (FPR) means
legitimate messages wrongly moved to Spam.

| Data | Setting | Spam caught | FPR |
|---|---|---|---|
| Public-dataset test split (3,535 spam / 968 legit) | Conservative | 92.0% | 0.10% |
| | **Balanced (default)** | **95.4%** | **1.34%** |
| | Aggressive | 96.2% | 1.96% |
| Scam campaigns that first appeared after the training cutoff | Balanced | 94.9% | — |
| 200 hand-written modern US messages | Conservative | 78% | 0% |
| | **Balanced** | **89%** | **3%** |
| Same set, transactional legit only (no personal chat or first contacts): 100 spam / 76 legit | **Balanced** | **89%** | **1.3%** (1 of 76) |

- No one-time-passcode (OTP) message was ever filtered.
- Undoing obfuscation (invisible characters, look-alike letters, fullwidth and "math" letters) gives identical model
  input, 100% of the time.
- Visible tricks (spaced letters, leetspeak, typos) cost only 1.3 points of recall.

Two of the 13 acceptance gates are not met:
- **Test-split FPR (target ≤ 0.5%).** 10 of the 13 false positives are one template family: legitimate recruiter
  follow-ups ("Hi Jose, this is Sophia from Acme Corp HR…"), which read like job scams. Excluding that family, the FPR
  is 0.31%.
- **Modern-set FPR (target ≤ 2%).** The 3 misses are a card-payment receipt, a Facebook Marketplace inquiry and a
  doctor's-office reschedule. On transactional messages alone (passcodes, bank, delivery, appointments, promos)
  it's 1 of 76: the card-payment receipt.

The underlying gap is that public data has few modern legitimate texts. One "Not spam" tap fixes that message and
its look-alikes, and the Conservative setting gives 0% FPR on both sets.

**Learning from feedback** (`s08_simulate.py`: 8 simulated streams of transactional and spam messages, learning only
from marks). Corrections carry over to the message coming back:
- **Scams the model missed.** After you mark one, 65% of its variants (new names, amounts, links, new numbers) are
  caught, vs 10% without learning.
- **Legitimate notices it flagged.** After "Not spam", 75% of that sender's similar notices stay in the inbox, vs
  12%.

Overall (about 33 marks per 800 messages):
- a user who corrects mistakes both ways ends at 95% of the no-learning cost (5 × false positives + misses), with
  36% fewer errors on repeat campaigns;
- a user who marks wrongly 5% of the time ends at 97%.

The trade-off is deliberate ("light retraining"). Someone who only ever marks spam ends at 98%, but about 1.6% more
legitimate transactional messages get flagged, because delivery and bank scams share wording with real notices.
- Memory-only learning would avoid that drift, but generalizes "Not spam" less well (62% kept).
- Stronger retraining keeps 88% but drifts 2.8%.

The setting is the `hash` prior in `ml/config/pipeline.yaml`.

**On the phone (emulator measurement).** The encoder takes 13 ms (median) / 29 ms (95th percentile) per message.
Results match the Python reference with zero changed decisions over 733 test cases. On the desktop JVM the match is
exact (1e-14).

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

Settings → Apps → Default apps → SMS app → Google Messages. Every message Sift sent or received is in the
system's message store, so it shows up there too.

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
uv run python -I scripts/s01_corpus.py       # normalize, dedupe, near-duplicate clusters
uv run python -I scripts/s02_split.py        # group-stratified splits + "new campaign" holdout
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

Training data (all permissively licensed):
- UCI SMS Spam Collection, CC BY 4.0
- Mishra & Soni 2022 SMS phishing, CC BY 4.0
- NCSU `wspr-ncsu/sms-phishing`, MIT
- UCL `reportsmishing/Smishing-Dataset-IMC25`, CC BY 4.0
- About 160 hand-written modern legitimate-message templates

A separate set of 400 hand-written modern US messages is used for evaluation only.

## Known limitations

- Scam "openers" with no link or ask ("Hi, is this Sarah?") read exactly like real first messages. The base model
  lets them through; your marks and the first-contact signal handle them.
- Not in v1: MMS, search, blocking, choosing a SIM per message.
- Learned data is deliberately excluded from cloud backup. Clearing the app's data resets what it learned.
