"""Deterministic parity corpus: hand-picked edge cases plus seeded random Unicode strings."""

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
