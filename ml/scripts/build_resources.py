"""Builds the parity-critical resource files used by norm_v1 (see SPEC.md).

Usage:
    uv run python -I scripts/build_resources.py --confusables /path/confusables.txt --tlds /path/tlds-alpha-by-domain.txt

Inputs are Unicode 15.0.0 `confusables.txt` (matches Python 3.12's unicodedata) and IANA's TLD list. The outputs
are committed; Kotlin reads the committed files, so this script only needs to run when the lists change.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "resources"

# GSM 03.38 default alphabet plus the extension table (escape-prefixed characters).
GSM7_BASIC = (
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
GSM7_EXTENSION = "\f^{}\\[~]|€"

# Multi-label public suffixes worth treating as one suffix: common second-level ccTLD registries plus free-hosting
# platforms that phishing kits abuse (so "evil.github.io" yields the registrable domain "evil.github.io").
PSL_LITE = """
co.uk org.uk me.uk ltd.uk plc.uk net.uk ac.uk gov.uk
com.au net.au org.au edu.au gov.au co.nz org.nz net.nz
co.in net.in org.in gov.in ac.in co.za org.za gov.za
com.br net.br com.mx com.ar com.co com.pe com.ve com.ec
com.cn net.cn org.cn gov.cn com.hk com.sg com.my com.ph com.tw com.vn com.pk com.bd com.ng co.ke
co.jp ne.jp or.jp co.kr co.id co.th com.tr com.eg com.sa co.il com.ua net.ua com.ru
github.io gitlab.io web.app firebaseapp.com pages.dev workers.dev vercel.app netlify.app herokuapp.com
blogspot.com wixsite.com weebly.com glitch.me ngrok.io ngrok-free.app azurewebsites.net appspot.com
r2.dev onrender.com fly.dev surge.sh 000webhostapp.com wordpress.com myshopify.com square.site
godaddysites.com carrd.co webflow.io framer.website notion.site sites.google.com
""".split()

# TLDs that are also everyday English words or abbreviations. A bare "word.word" join (a missing space after a
# period) must not become a URL, so these TLDs need extra evidence (a path, 3+ labels, or a hyphen/digit).
TLD_STOPLIST = """
am as at be by do id in is it me my no so to tv us
ai al ad ma pa pm st ms mr er et eu ie eg ph pe
app art bar best bet bid bike bio black blog blue book box buy buzz cab cafe call cam camp car care cars cash
casa cat center chat city click cloud club codes cool date day deal deals dog domains earth eat email fail
fan fans farm fish fit fly foo food free fun fund game games gift gifts gold golf green group guide guru hair
health help here home hot house how ink jobs joy kids land law life like link live living loan loans lol love
ltd map market media men menu mom money movie music network new news now one online page party pay pet phone
photo pics pink pizza place play plus pro promo red rent rest review rich rip rocks run sale save school shop
show site ski sky soy spa space sport spot star store study style sucks support surf talk tax taxi team tech
tips today tools top tours town toys trade vet video vip vote watch win wine work works world wow yoga you
zone
""".split()


def gsm7_lines() -> list[str]:
    cps = sorted({ord(c) for c in GSM7_BASIC + GSM7_EXTENSION})
    out = ["# GSM 03.38 default alphabet + extension table, one hex code point per line (SPEC norm_v1 step 2)."]
    for cp in cps:
        out.append(f"{cp:04X}  # {unicodedata.name(chr(cp), 'CONTROL')}")
    return out


def is_ascii_alnum(s: str) -> bool:
    return bool(s) and all(c.isascii() and c.isalnum() for c in s)


def confusable_lines(path: Path) -> tuple[list[str], list[str]]:
    rows: dict[int, str] = {}
    skipped: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(";")]
        src = [int(h, 16) for h in fields[0].split()]
        dst = "".join(chr(int(h, 16)) for h in fields[1].split())
        if len(src) != 1 or src[0] < 0x80:
            continue
        cp = src[0]
        ch = chr(cp)
        if unicodedata.normalize("NFKC", ch) != ch:
            continue  # NFKC already folds it (step 5 runs before step 6)
        if unicodedata.category(ch) in ("Mn", "Me", "Mc"):
            continue  # marks are stripped in step 7
        if not is_ascii_alnum(dst) or len(dst) > 2:
            continue
        words = unicodedata.name(ch, "").split()
        # confusables.txt uses lowercase "l" as the prototype for every I-like glyph; restore the intended letter
        # so e.g. Greek "ΙRS" folds to "IRS" rather than "lRS".
        if dst == "l" and ("I" in words or "IOTA" in words or "DOTLESS" in words):
            dst = "I" if "CAPITAL" in words else "i"
        if cp in rows and rows[cp] != dst:
            skipped.append(f"{cp:04X} conflicting {rows[cp]!r} vs {dst!r}")
            continue
        rows[cp] = dst
    out = ["# Single non-ASCII code point -> ASCII (1-2 chars), from Unicode 15.0.0 confusables.txt (SPEC norm_v1 step 6).",
           "# Only NFKC-stable sources; I-like glyphs keep their letter (I/i) instead of the 'l' prototype."]
    for cp in sorted(rows):
        out.append(f"{cp:04X}\t{rows[cp]}  # {unicodedata.name(chr(cp), '?')}")
    return out, skipped


def tld_lines(path: Path) -> list[str]:
    tlds = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        t = raw.strip().lower()
        if not t or t.startswith("#"):
            continue
        if t.isascii() and t.isalpha():
            tlds.append(t)
    header = path.read_text(encoding="utf-8").splitlines()[0].strip()
    return [f"# All-letter IANA TLDs ({header.lstrip('# ')}) (SPEC URL scanner)."] + sorted(set(tlds))


def write(name: str, lines: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"--- {name}: {len(lines)} lines")
        return
    (RES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RES / name} ({len(lines)} lines)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusables", type=Path, required=True)
    ap.add_argument("--tlds", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="print N sample confusable rows")
    args = ap.parse_args()

    conf, skipped = confusable_lines(args.confusables)
    if args.show:
        step = max(1, (len(conf) - 2) // args.show)
        for row in conf[2::step][: args.show]:
            print("  ", row)
        for s in skipped[:20]:
            print("  skipped:", s)
    stop = sorted(set(TLD_STOPLIST))
    tlds = tld_lines(args.tlds)
    missing = [t for t in stop if t not in set(tlds[1:])]
    if missing:
        print("stoplist entries that are not IANA TLDs (dropped):", " ".join(missing))
        stop = [t for t in stop if t not in missing]

    write("gsm7_v1.txt", gsm7_lines(), args.dry_run)
    write("confusables_v1.tsv", conf, args.dry_run)
    write("tlds_v1.txt", tlds, args.dry_run)
    write("tld_stoplist_v1.txt", ["# TLDs that are everyday words: bare domains need a path, 3+ labels, or a hyphen/digit (SPEC URL scanner)."] + stop, args.dry_run)
    write("psl_lite_v1.txt", ["# Multi-label public suffixes (SPEC URL scanner: registrable domain)."] + sorted(set(PSL_LITE)), args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
