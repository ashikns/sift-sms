"""Deterministic parity corpus: hand-picked edge cases plus seeded random Unicode strings, each with a sender key.

Everything here is synthetic or hand-written; nothing comes from a user's own messages.
"""

from __future__ import annotations

import random

EDGE_CASES = [
    "",
    " ",
    "Hello, World!  See you at 7?",
    "URGENT: Your E-ZPass account has an unpaid toll of $4.15. Pay now at https://ezpass-toll.top/pay?id=9 to avoid a $50 fee.",
    "USPS: Your package is on hold due to an incomplete address. Update: usps-redeliver.info/track",
    "Your verification code is 482913. Don't share it with anyone.",
    "Your 2FA code: 4829",
    "WINNER!! You've won a £1000 prize. Txt CLAIM to 81010 (18+ only) T&Cs apply",
    "Hi, this is Sam from the soccer team \U0001F44B are you coming Saturday?",
    "Аpple ID locked. Verify at https://appIe-id.support.com@secure-login.top/x",
    "ＦＲＥＥ \U0001D407\U0001D400\U0001D40C\U0001D401\U0001D414\U0001D411\U0001D406\U0001D404\U0001D411",
    "fr​ee gi­ft c‍ard ⁠now",
    "café naıve résumé über Z̶a̶l̶g̶o̶",
    "It’s “fine” — really • ok … 你好 مرحبا",
    "Call (555) 010-4477 or +1 555 010 9921 now",
    "Text 55501 for info, email jane.doe@example.com",
    "Pay €20 or ₹500 or ¥300 or 5 dollars or USD 10",
    "Reply STOP to opt out. Msg&data rates may apply.",
    "See you soon.Love you",
    "bit.ly/3xYz tinyurl.com/abc t.co/q https://192.168.1.10/login",
    "https://www.paypal.com@evil.top/login www.chase.com.secure-verify.xyz",
    "https://xn--pple-43d.com/verify and http://evil.github.io/x and shop.example.co.uk",
    "\U0001F468‍\U0001F469‍\U0001F467 family ❤️ \U0001F1FA\U0001F1F8",
    "a\ud800b \udc00 lone surrogates",
    "x" * 2100,
    "word " * 300,
    "supercalifragilisticexpialidocious" * 4,
    "lol ok see u l8r :) ^_^ ~~~ `backticks` {braces} [brackets] |pipes| \\slash",
    "\t\nTabs\r\nand\x00controls\x7f\x85here",
    "ALL CAPS MESSAGE!!!!!! CLICK NOW!!!",
    "Your Amazon order #112-5551234-0099 has shipped. Track at https://www.amazon.com/gp/your-account/orders",
    "Ονε ΙRS іnvoice ѕecurity",
    "ⅠⅡⅢ ①② ½ ™ ﬁx",
]

# Sender keys as the app normalizes them (SenderInfo.key), for the sid, DLT category and SND cases: DLT headers in every
# category, without one, with a letter that isn't one, and near misses; phone numbers, short codes, an email,
# alphanumeric IDs, and no sender at all.
SENDERS = [
    "AX-ICICIT-S", "JD-ICICIT-T", "VM-HDFCBK-G", "AD-HDFCBK-P", "BZ-AIRTEL", "ICICIT", "AX-ICI-S", "AX-ICICIBNK-S",
    "AX-123456-S", "+915550100123", "+15550100199", "57575", "jane.doe@example.com", "AMAZON ALERTS", "unknown", "",
    None, "AX-ICICIBANK-S", "AX-IC-S", "A1-ICICIT-S", "AXX-ICICIT-S", "AX-ICICIT-SS", "AX-ICICIT-1", "AX-ICICIT-",
    "AX--S", "ax-icicit-s", "AX-ICICIT-S-T", "-ICICIT-S", "AX-ICİCIT-S", "AX-ICICIT-S ", "AX-ICIC|T-S", "AX-ICICIT-X",
    "AMAZ\ud800N \U0001F4E6",  # an unpaired surrogate (a malformed address) and a real supplementary code point
]

# Hand-written Indian messages with the kind of sender each arrives from: one sender's alerts and upsell on two routes,
# another bank's passcode and debit alert, a government weather alert, a phished KYC notice from a phone number, a
# short code and an operator.
DLT_CASES = [
    ("AX-ICICIT-S", "INR 2,500.00 spent using ICICI Bank Card XX1234 on 28-Oct-26 at AMAZON. Avl Limit: INR 1,20,000.00. "
                    "If not you, call 18002662."),
    ("JD-ICICIT-S", "INR 640.00 spent using ICICI Bank Card XX1234 on 29-Oct-26 at SWIGGY. Avl Limit: INR 1,19,360.00. "
                    "If not you, call 18002662."),
    ("VM-ICICIT-P", "Dear Customer, your ICICI Bank Credit Card limit can be increased to Rs 3,00,000. Apply now at "
                    "https://icici.co/lmt T&C apply"),
    ("VM-HDFCBK-S", "482913 is the OTP for your HDFC Bank card ending 5678 for INR 999.00 at FLIPKART. Valid 10 mins. "
                    "Do not share."),
    ("AD-HDFCBK-T", "Update: INR 1,250.00 debited from HDFC Bank A/c XX9012 on 28-10-26 to VPA zomato@hdfcbank. "
                    "Not you? Call 18002586161"),
    ("JM-NDMAEW-G", "Orange alert: heavy rainfall likely in Mumbai and Thane in the next 24 hours. Avoid waterlogged "
                    "roads and stay indoors. -NDMA"),
    ("+915550100123", "Dear user your KYC is pending. Update now at http://kyc-update-sbi.top or your account will be "
                      "blocked today."),
    ("57575", "Your Amazon order #405-1234567-8901234 is out for delivery today. Track at https://amzn.in/d/abc"),
    ("BZ-AIRTEL", "Recharge with Rs 299 for unlimited calls and 2GB/day. Offer valid till 31 Oct. Visit airtel.in/prepaid"),
]
DLT_IDS = ["ICICIT", "HDFCBK", "SBIINB", "AXISBK", "AMAZON", "AIRTEL", "JIOINF", "PAYTMB", "ZOMATO", "SWIGGY"]

SEGMENTS = [
    lambda r: r.choice(["free", "WIN", "prize", "code", "verify", "account", "toll", "USPS", "hi", "lol", "ok"]),
    lambda r: str(r.randrange(10 ** r.randrange(1, 9))),
    lambda r: r.choice(["$", "£", "€", "₹", "¥", "¢", "%", "&", "#", "@", "!", "?", "...", "-", "/"]),
    lambda r: r.choice(["https://", "http://", "www.", ""]) + r.choice(["ezpass", "usps", "pay-now", "a1b2", "x"])
    + r.choice([".com", ".top", ".xyz", ".so", ".love", ".co.uk", ".github.io", ".ly"]) + r.choice(["", "/p?q=1", "/"]),
    lambda r: "".join(chr(r.choice([0x0430, 0x0435, 0x043E, 0x0440, 0x0441, 0x0391, 0x0392, 0x0399, 0x03BF, 0x0456]))
                      for _ in range(r.randrange(1, 4))),
    lambda r: "".join(chr(0xFF21 + r.randrange(26)) for _ in range(r.randrange(1, 5))),
    lambda r: "".join(chr(0x1D400 + r.randrange(52)) for _ in range(r.randrange(1, 5))),
    lambda r: chr(r.choice([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0xFE0F, 0x202E])),
    lambda r: chr(r.randrange(0x1F300, 0x1FA70)),
    lambda r: "e" + chr(r.randrange(0x0300, 0x0370)),
    lambda r: chr(r.randrange(0x4E00, 0x9FFF)),
    lambda r: chr(r.randrange(0x00A0, 0x0250)),
    lambda r: r.choice([" ", "  ", "\n", "\t", " ", " ", "　"]),
    lambda r: "(555) 010-" + str(r.randrange(1000, 9999)),
    lambda r: chr(r.randrange(0x20, 0x7F)),
    lambda r: chr(r.randrange(0x80, 0x3000)),
    lambda r: chr(r.randrange(0x10000, 0x2FFFF)),
]


def edge_senders() -> list[str | None]:
    """A sender key for each of EDGE_CASES, cycling through SENDERS."""
    return [SENDERS[i % len(SENDERS)] for i in range(len(EDGE_CASES))]


def random_senders(n: int, seed: int = 4321) -> list[str | None]:
    """Sender keys for random cases: DLT headers on random routes (half of them), phone numbers, short codes, the
    hand-picked keys, and no sender."""
    r = random.Random(seed)
    out: list[str | None] = []
    for _ in range(n):
        k = r.randrange(8)
        if k < 4:
            route = chr(65 + r.randrange(26)) + chr(65 + r.randrange(26))
            out.append(f"{route}-{r.choice(DLT_IDS)}" + (f"-{r.choice('SPTG')}" if r.random() < 0.8 else ""))
        elif k == 4:
            out.append(r.choice(["+91555", "+1555"]) + str(r.randrange(10 ** 6, 10 ** 7)))
        elif k == 5:
            out.append(str(r.randrange(100, 1000000)))
        elif k == 6:
            out.append(r.choice(SENDERS))
        else:
            out.append(None)
    return out


def random_strings(n: int, seed: int = 1234) -> list[str]:
    r = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(r.randrange(1, 25)):
            parts.append(r.choice(SEGMENTS)(r))
            if r.random() < 0.6:
                parts.append(" ")
        out.append("".join(parts))
    return out
