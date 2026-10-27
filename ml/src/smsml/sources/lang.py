"""Small, documented language heuristics for sources without a language field.

is_english (NCSU): a message is English when
  1. at least `min_ascii_ratio` of its letter-bearing words contain an ASCII letter (mixed-script words such
     as homoglyph-obfuscated "Ηello" count as Latin, so obfuscated English spam isn't dropped), and
  2. it has at least `min_stopword_hits` hits from an English list (function words plus common SMS-scam content
     words such as "login", "balance", "package") and more of them than function words from any of the other
     languages that dominate the smishing corpora (es, fr, de, nl, it, pt, id, ro, nordic, fi, pl).

It is deliberately conservative toward short messages: a message with no function word in any language is
treated as English only if it is mostly ASCII and has at most 4 words (e.g. "Verify now: bit.ly/x").

script_language (inbox): the language of the script most of a message's letters are in. ASCII letters are English,
an Indian script gives the language it most often writes (SCRIPTS), any other script None. A message without
letters (an amount, a number) is English.
"""

from __future__ import annotations

from collections import Counter

ENGLISH = frozenset("""
a about after all am an and any are as at be been but by can could did do does dont for from get got had has
have he her here hi him his how i if im in is it its just know let like me my no not now of ok on or our out
please so that the their them there they this to today up us was we were what when where which who will with
would you your youre yours thank thanks hello dear sir madam
account address alert app balance bank bonus call card cash claim click code confirm customer delivery deposit
free gift id link login new notice now offer order package password payment pending prize receive reward secure
security service shipping support total update urgent username verify visit win winner withdraw
""".split())

OTHER = {
    "es": frozenset("el la los las de del que y en un una por para con su sus es se al lo como mas pero tu usted "
                    "hola estimado cliente cuenta haga aqui".split()),
    "fr": frozenset("le la les de des du et en un une pour avec est sur vous votre vos pas que qui au aux ce "
                    "bonjour cher compte ici".split()),
    "de": frozenset("der die das und ist nicht ein eine mit von zu den dem des sie ihr ihre ihren wir bitte "
                    "hallo konto hier jetzt".split()),
    "nl": frozenset("de het een en van op te dat die voor met niet zijn uw je jouw u wij hier nu".split()),
    "it": frozenset("il lo la gli le di che e un una per con non del della sono tuo tua suo sua gentile "
                    "cliente conto qui".split()),
    "pt": frozenset("o os a as de do da dos das que e em um uma para com nao seu sua voce ola caro cliente "
                    "aqui".split()),
    "id": frozenset("yang dan di ke dari untuk dengan ini itu anda kami tidak ada akan bisa".split()),
    "ro": frozenset("este si sau la pe cu pentru tau ta dvs va de in din urmatorul acum".split()),
    "nordic": frozenset("og i til med din det er har fra hos av som att och eller ditt dig deg kan ikke inte "
                        "vid idag kr".split()),
    "fi": frozenset("ja on kun jopa asti sinun olet tai tama".split()),
    "pl": frozenset("i w na z do jest nie sie twoj twoja prosze".split()),
}

# Unicode blocks of the Indian scripts → the language each most often writes.
SCRIPTS = (
    (0x0900, 0x097F, "hi"),  # Devanagari
    (0x0980, 0x09FF, "bn"),  # Bengali
    (0x0A00, 0x0A7F, "pa"),  # Gurmukhi
    (0x0A80, 0x0AFF, "gu"),  # Gujarati
    (0x0B00, 0x0B7F, "or"),  # Oriya
    (0x0B80, 0x0BFF, "ta"),  # Tamil
    (0x0C00, 0x0C7F, "te"),  # Telugu
    (0x0C80, 0x0CFF, "kn"),  # Kannada
    (0x0D00, 0x0D7F, "ml"),  # Malayalam
)


def _words(text: str) -> list[str]:
    out, cur = [], []
    for ch in text:
        if ch.isalpha() or ch == "'":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return [w.strip("'") for w in out if w.strip("'")]


def is_english(text: str, min_ascii_ratio: float = 0.9, min_stopword_hits: int = 1) -> bool:
    words = _words(text)
    if not words:
        return False
    latin = sum(1 for w in words if any("a" <= c.lower() <= "z" for c in w))
    if latin / len(words) < min_ascii_ratio:
        return False
    ascii_words = [w.lower().replace("'", "") for w in words if w.isascii()]
    en_hits = sum(1 for w in ascii_words if w in ENGLISH)
    other_hits = max(sum(1 for w in ascii_words if w in vocab) for vocab in OTHER.values())
    if en_hits == 0 and other_hits == 0:
        return len(words) <= 4
    return en_hits >= min_stopword_hits and en_hits > other_hits


def script_language(text: str) -> str | None:
    counts: Counter[str | None] = Counter()
    for ch in text:
        if ch.isascii():
            if ch.isalpha():
                counts["en"] += 1
            continue
        cp = ord(ch)
        code = next((lang for lo, hi, lang in SCRIPTS if lo <= cp <= hi), None)
        if code or ch.isalpha():  # an Indian block counts whole (its vowel signs aren't letters), elsewhere letters
            counts[code] += 1
    return counts.most_common(1)[0][0] if counts else "en"
