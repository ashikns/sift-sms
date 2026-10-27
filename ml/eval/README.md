# modern_v1 — hand-written modern US SMS evaluation set (frozen)

`modern_v1.jsonl` holds 400 messages written by hand for Sift (2026-10-27), independently of the training templates
in `templates/ham_v1.yaml`: 200 spam and 200 ham in a 2024–2026 US style. **Never train on it.** It measures what
the public corpora can't: modern smishing (tolls, USPS redelivery, wrong-number openers, task scams, ...) against
modern legitimate traffic (OTP codes, carrier/bank alerts, opted-in promos, first-contact texts).

Fields: `id`, `text`, `label` (1 = spam), `category`, `sender` (fictional: 555-01xx numbers, made-up short codes and
email handles), `sender_type` (PHONE | SHORT_CODE | ALPHANUMERIC | EMAIL), `split` (`dev` / `test`).

- Spam categories: toll, delivery, bank_fraud, account, wrong_number, job, crypto, loan, prize, government, utility,
  tech_support, misc, and obfuscated (homoglyphs, spaced/dotted letters, zero-width and soft-hyphen characters,
  fullwidth and math-bold letters, small capitals, combining underlines, leetspeak).
- Ham categories: otp, delivery, appointment, pharmacy, bank, alert, travel, promo, account, personal, first_contact.
- Split: within each (label, category) group the items are shuffled with a fixed seed and halved; odd groups alternate
  the extra item between dev and test, giving exactly 100 messages per label per split.
- Wrong-number openers are labeled spam (they open "pig butchering" scams) even though they read like legitimate
  first-contact texts. Expect them to be the hardest category.
- All malicious-looking domains are invented. Brand names appear only to make impersonation realistic.
