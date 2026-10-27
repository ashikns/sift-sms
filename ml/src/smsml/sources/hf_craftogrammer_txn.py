"""craftogrammer/sms-tool-calling-sft on Hugging Face (no license declared): real Indian SMS with their DLT senders,
inside tool-calling conversations that train a transaction extractor.

Files: `india/hf_craftogrammer_txn/sft_tool_{train,eval}.jsonl`, one conversation per line. Its user turn is
"SMS sender: <sender>\nSMS message: <text>" and its final tool call, submit_extraction, holds the result. Only
completed transactions (skip false, direction in / out) are kept, as ham; the other messages (OTPs, promos,
reminders, failed or balance-only notices) carry no spam label, so they're skipped. Deduplicated on (sender, text);
source_id = `train:<line>` / `eval:<line>`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .lang import script_language
from .schema import finalize, sender_type

DIR = Path("india/hf_craftogrammer_txn")
FILES = {"train": "sft_tool_train.jsonl", "eval": "sft_tool_eval.jsonl"}
SENDER, MESSAGE = "SMS sender: ", "\nSMS message: "


def transaction(conversation: dict) -> tuple[str, str] | None:
    """(sender, text) when the conversation's final extraction is a completed transaction, else None."""
    calls = [c["function"] for m in conversation["messages"] if m["role"] == "assistant"
             for c in m.get("tool_calls") or []]
    if not calls or calls[-1]["name"] != "submit_extraction":
        return None
    result = calls[-1]["arguments"]
    if isinstance(result, str):
        result = json.loads(result)
    if result.get("skip") or result.get("direction") not in ("in", "out"):
        return None
    user = next(m["content"] for m in conversation["messages"] if m["role"] == "user")
    head, sep, text = user.partition(MESSAGE)
    if not sep or not head.startswith(SENDER):
        raise ValueError(f"unexpected user turn {user[:60]!r}")
    return head[len(SENDER):].strip(), text.strip()


def load(raw_dir: Path) -> pd.DataFrame:
    rows, seen = [], set()
    for part, name in FILES.items():
        with (Path(raw_dir) / DIR / name).open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                found = transaction(json.loads(line))
                if found is None or not found[1] or found in seen:
                    continue
                seen.add(found)
                sender, text = found
                kind = sender_type(sender)
                rows.append({
                    "text": text,
                    "label": 0,
                    "source": "hf_craftogrammer_txn",
                    "source_id": f"{part}:{lineno}",
                    "lang": script_language(text),
                    "sender": sender if kind else None,
                    "sender_type": kind,
                })
    return finalize(pd.DataFrame(rows))
