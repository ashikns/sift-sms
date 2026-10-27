"""Deterministic expansion of the hand-written modern-ham templates (templates/ham_v1.yaml).

Placeholder grammar (see the YAML header): `{slot}`, `{a|b|c}` (options may nest placeholders), generators
(`{d6}`, `{time}`, `{n:2-6}`, ...), and `{name@var}` which draws once per message and reuses the value for every
`@var` reference (e.g. a WebOTP code that must appear twice).

Each template gets its own RNG seeded from (seed, template id), so editing one template never changes the
expansions of another. Expansions are deduplicated within a template.
"""

from __future__ import annotations

import hashlib
import random
import string
from pathlib import Path

import pandas as pd
import yaml

from .schema import finalize

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AREA_CODES = ["206", "214", "303", "305", "312", "404", "407", "412", "503", "512", "602", "614", "615", "617",
              "704", "801", "813", "816", "916", "919"]


def _rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _generate(name: str, rng: random.Random) -> str | None:
    """Built-in generators; returns None when `name` isn't one."""
    if len(name) > 1 and name[0] == "d" and name[1:].isdigit():
        return "".join(rng.choice(string.digits) for _ in range(int(name[1:])))
    if name.startswith("n:"):
        lo, hi = (int(x) for x in name[2:].split("-", 1))
        return str(rng.randint(lo, hi))
    if name == "code6_dash":
        return f"{rng.randint(0, 999):03d}-{rng.randint(0, 999):03d}"
    if name == "amount":
        return _money(round(rng.uniform(1.5, 480.0), 2))
    if name == "amount_big":
        return _money(round(rng.uniform(100.0, 6000.0), 2))
    if name == "amount_round":
        return f"${rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150, 200])}"
    if name == "time":
        hour, minute = rng.randint(7, 21), rng.choice(range(0, 60, 5))
        suffix = "AM" if hour < 12 else "PM"
        return f"{(hour - 1) % 12 + 1}:{minute:02d} {suffix if rng.random() < 0.8 else suffix.lower()}"
    if name == "date_md":
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        return f"{month:02d}/{day:02d}" if rng.random() < 0.6 else f"{month}/{day}"
    if name == "date_long":
        return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}"
    if name == "weekday":
        return rng.choice(WEEKDAYS)
    if name == "last4":
        return f"{rng.randint(0, 9999):04d}"
    if name == "order":
        return "#" + "".join(rng.choice(string.digits) for _ in range(rng.randint(7, 9)))
    if name == "order_amz":
        return f"11{rng.randint(1, 4)}-{rng.randint(0, 9999999):07d}-{rng.randint(0, 9999999):07d}"
    if name == "track_ups":
        return "1Z" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6)) + \
            "".join(rng.choice(string.digits) for _ in range(10))
    if name == "track_usps":
        return "9400" + "".join(rng.choice(string.digits) for _ in range(18))
    if name == "track_fedex":
        return "".join(rng.choice(string.digits) for _ in range(12))
    if name == "flightno":
        return str(rng.randint(100, 2999))
    if name == "gate":
        return f"{rng.choice('ABCDE')}{rng.randint(1, 45)}"
    if name == "plate":
        letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
        return f"{letters}{rng.randint(1000, 9999)}" if rng.random() < 0.6 else f"{rng.randint(1, 9)}{letters}{rng.randint(100, 999)}"
    if name == "minutes":
        return str(rng.choice([5, 10, 15, 30]))
    if name == "zip":
        return f"{rng.randint(10000, 99950):05d}"
    if name == "pct":
        return f"{rng.choice(range(15, 65, 5))}%"
    if name == "phone":
        area, tail = rng.choice(AREA_CODES), f"01{rng.randint(0, 99):02d}"
        return rng.choice([f"({area}) 555-{tail}", f"{area}-555-{tail}", f"{area}.555.{tail}"])
    if name == "tollfree":
        return f"1-800-555-01{rng.randint(0, 99):02d}"
    return None


def _split_options(body: str) -> list[str]:
    """Split on top-level `|` (ignoring pipes inside nested braces)."""
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


class Expander:
    def __init__(self, global_slots: dict[str, list[str]], local_slots: dict[str, list[str]] | None,
                 rng: random.Random):
        self.slots = {**global_slots, **(local_slots or {})}
        self.rng = rng
        self.bound: dict[str, str] = {}

    def expand(self, text: str, depth: int = 0) -> str:
        if depth > 8:
            raise ValueError(f"placeholder nesting too deep in {text!r}")
        out, i = [], 0
        while i < len(text):
            if text[i] != "{":
                if text[i] == "}":
                    raise ValueError(f"unbalanced '}}' in {text!r}")
                out.append(text[i])
                i += 1
                continue
            j, level = i + 1, 1
            while j < len(text) and level:
                level += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            if level:
                raise ValueError(f"unbalanced '{{' in {text!r}")
            out.append(self._resolve(text[i + 1:j - 1], depth))
            i = j
        return "".join(out)

    def _resolve(self, body: str, depth: int) -> str:
        options = _split_options(body)
        if len(options) > 1:
            return self.expand(self.rng.choice(options), depth + 1)
        name, _, var = body.partition("@")
        name = name.strip()
        if var and var in self.bound:
            return self.bound[var]
        value = _generate(name, self.rng)
        if value is None:
            if name not in self.slots:
                raise KeyError(f"unknown slot or generator {name!r}")
            value = self.expand(str(self.rng.choice(self.slots[name])), depth + 1)
        if var:
            self.bound[var] = value
        return value


def load_templates(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    ids = [t["id"] for t in doc["templates"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate template ids: {sorted(dupes)}")
    return doc


def expand_templates(doc: dict, *, seed: int, target_total: int, max_per_template: int) -> pd.DataFrame:
    templates = doc["templates"]
    total_weight = sum(float(t.get("weight", 1)) for t in templates)
    rows = []
    for t in templates:
        want = max(1, min(max_per_template, round(target_total * float(t.get("weight", 1)) / total_weight)))
        rng = _rng(seed, t["id"])
        seen: set[str] = set()
        for _ in range(want * 25):
            if len(seen) >= want:
                break
            text = Expander(doc.get("slots", {}), t.get("slots"), rng).expand(t["text"]).strip()
            if text in seen:
                continue
            seen.add(text)
            rows.append({
                "text": text,
                "label": 0,
                "source": "templates",
                "source_id": f"{t['id']}#{len(seen)}",
                "category": t["category"],
                "template_id": t["id"],
                "lang": "en",
            })
    return finalize(pd.DataFrame(rows))


def load(templates_path: Path, *, seed: int, target_total: int, max_per_template: int) -> pd.DataFrame:
    return expand_templates(load_templates(templates_path), seed=seed, target_total=target_total,
                            max_per_template=max_per_template)
