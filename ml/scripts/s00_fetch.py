"""Download the public SMS datasets into ml/data/raw/<source>/ and record them in MANIFEST.json.

Each source gets its own directory (the Indian corpora under india/); downloads resume from `.part` files, retry
with backoff, and are verified against published SHA-256 hashes where the host provides one, else against the hash
pinned when the file was first fetched (files at pinned commits/revisions, so their bytes can't change). Archives
marked `extract` are extracted (with path-traversal checks) into `<source>/extracted/`. Nothing downloaded is ever
executed. The user's labeled inbox export is local and private, so it isn't fetched.

Usage: uv run python -I scripts/s00_fetch.py [--only SOURCE ...] [--force]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

ML_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ML_DIR / "data" / "raw"
MANIFEST = RAW_DIR / "MANIFEST.json"

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120
RETRIES = 6
CHUNK = 1 << 16
USER_AGENT = "sift-smsml-fetch/1.0 (+research dataset download)"

MENDELEY_FILES_API = "https://data.mendeley.com/public-api/datasets/f45bkkt8pr/files?folder_id=root&version=1"
GH_NCSU = "https://raw.githubusercontent.com/wspr-ncsu/sms-phishing/main/"
GH_IMC25 = "https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/main/"
GH_IIITD = ("https://raw.githubusercontent.com/princebari/-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-"
            "Collection-of-Hindi-and-English-Messages/2faeb9a823bdc1d6eee036152bd79f81759d05c3/")
GH_CUBE = "https://raw.githubusercontent.com/saranshbansal/spam-detection-analytics-tool/32f57a55b523689722ebd18f14939a08fabfa784/"
GH_JUNIORALIVE = "https://raw.githubusercontent.com/junioralive/india-spam-sms-classification/09137cc677a7d8d24caef0817de41f55a6b3ed44/"
HF_RIDHAM = "https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited/resolve/45e92c38ba20e973ef3ff5d1893a0a710f622018/"
HF_CRAFTOGRAMMER = "https://huggingface.co/datasets/craftogrammer/sms-tool-calling-sft/resolve/6764a2ae68a010261f52f75101e9524f78050e32/"

SOURCES: dict[str, dict] = {
    "uci": {
        "title": "UCI SMS Spam Collection (Almeida & Gómez Hidalgo, 2011)",
        "homepage": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "files": [
            {"url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
             "name": "sms+spam+collection.zip", "extract": True},
        ],
    },
    "mishra": {
        "title": "SMS Phishing Dataset for Machine Learning and Pattern Recognition (Mishra & Soni, 2022)",
        "homepage": "https://data.mendeley.com/datasets/f45bkkt8pr/1",
        "license": "CC BY 4.0",
        "license_url": "http://creativecommons.org/licenses/by/4.0",
        # Resolved at run time from the Mendeley files API, which also publishes SHA-256 hashes.
        "mendeley_files": ["Dataset_5971.zip"],
        "files": [],
    },
    "ncsu": {
        "title": "On SMS Phishing Tactics and Infrastructure (Nahapetyan et al., IEEE S&P 2024)",
        "homepage": "https://github.com/wspr-ncsu/sms-phishing",
        "license": "MIT",
        "license_url": "https://github.com/wspr-ncsu/sms-phishing/blob/main/LICENSE",
        "files": [
            {"url": GH_NCSU + "phishing_messages.csv", "name": "phishing_messages.csv"},
            {"url": GH_NCSU + "phishing_campaigns.csv", "name": "phishing_campaigns.csv"},
            {"url": GH_NCSU + "LICENSE", "name": "LICENSE"},
            {"url": GH_NCSU + "README.md", "name": "README.md"},
        ],
    },
    "imc25": {
        "title": "Fishing for Smishing: Mining Public User Reports (Agarwal et al., IMC 2025)",
        "homepage": "https://github.com/reportsmishing/Smishing-Dataset-IMC25",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "files": [
            {"url": GH_IMC25 + "dataset/final_dataset_output.csv", "name": "final_dataset_output.csv"},
            {"url": GH_IMC25 + "dataset/time_day.csv", "name": "time_day.csv"},
            {"url": GH_IMC25 + "LICENSE.txt", "name": "LICENSE.txt"},
            {"url": GH_IMC25 + "README.md", "name": "README.md"},
        ],
    },
    "iiitd_smsassassin": {
        "title": "IIIT-Delhi SMS spam dataset (SMSAssassin; Yadav, Kumaraguru, Goyal, Gupta, Naik 2011), GitHub copy",
        "homepage": "https://github.com/princebari/-SMS-Spam-Classification-on-Indian-Dataset-A-Crowdsourced-"
                    "Collection-of-Hindi-and-English-Messages",
        "license": "none stated (a copy of an IIIT-D research dataset)",
        "license_url": None,
        "dir": "india/iiitd_smsassassin",
        "files": [
            {"url": GH_IIITD + "indian_spam.csv", "name": "indian_spam.csv",
             "pinned_sha256": "76625e8b1802a9a578b522ffb711dafe695a219600e6e2e180f1cea13c0791c3"},
            {"url": GH_IIITD + "README.md", "name": "README.md",
             "pinned_sha256": "44e5bcc19006c66f3ca3c48364011b49ed8e6525d8691376cab8e11364d4ba28"},
        ],
    },
    "cube_sms77k": {
        "title": "Indian SMS corpus cunique.csv (~77k unique, sender + truth label; 'Github1' in Salman et al. 2022)",
        "homepage": "https://github.com/saranshbansal/spam-detection-analytics-tool",
        "license": "none stated",
        "license_url": None,
        "dir": "india/cube_sms77k",
        "files": [
            {"url": GH_CUBE + "cunique.csv", "name": "cunique.csv",
             "pinned_sha256": "5bb81a15c1a5beadef82d7ee9bb5f81126394fe09efbf6f0b910934b904dc495"},
        ],
    },
    "junioralive_spam_ham": {
        "title": "India Spam SMS Classification (junioralive)",
        "homepage": "https://github.com/junioralive/india-spam-sms-classification",
        "license": "MIT",
        "license_url": GH_JUNIORALIVE + "LICENSE",
        "dir": "india/junioralive_spam_ham",
        "files": [
            {"url": GH_JUNIORALIVE + "dataset/spam_ham_india.csv", "name": "spam_ham_india.csv",
             "pinned_sha256": "e6a28126d1c4ec9d805a10cd89f2de2bb63650532102e9f35ab05c909c3a79f3"},
            {"url": GH_JUNIORALIVE + "LICENSE", "name": "LICENSE",
             "pinned_sha256": "203ac8eb9e2c16d5cd517c24732f87003f0b9be3e3bfd3ec3253d2cb6b61cb8d"},
            {"url": GH_JUNIORALIVE + "README.md", "name": "README.md",
             "pinned_sha256": "41f6988adea27e3d6896710a7b10c9e4e079ff1404c1d1f974fc5b623229ee16"},
        ],
    },
    "hf_ridham_synth_scam": {
        "title": "Indian Scam SMS, synthetic and audited (Ridham115/indian-scam-sms-synthetic-audited)",
        "homepage": "https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dir": "india/hf_ridham_synth_scam",
        "files": [
            {"url": HF_RIDHAM + "data/synthetic_audited.csv", "name": "synthetic_audited.csv",
             "pinned_sha256": "45bde5f156a1a42698bab209e32bcd41b3639013d772f644cc3208ed39fb61e2"},
            {"url": HF_RIDHAM + "README.md", "name": "README.md",
             "pinned_sha256": "bc661d6cbf9318a00786f34062f640eb24333644f2941b03375c22f27c755c7c"},
            {"url": HF_RIDHAM + "label_rubric.md", "name": "label_rubric.md",
             "pinned_sha256": "e44b9eda3e6540851a51570fe0866a102d1504e4825f591bd5d8074e834946fe"},
        ],
    },
    "hf_craftogrammer_txn": {
        "title": "craftogrammer/sms-tool-calling-sft (real Indian SMS with DLT senders in tool-calling conversations)",
        "homepage": "https://huggingface.co/datasets/craftogrammer/sms-tool-calling-sft",
        "license": "none declared",
        "license_url": None,
        "dir": "india/hf_craftogrammer_txn",
        "files": [
            {"url": HF_CRAFTOGRAMMER + "sft_tool_train.jsonl", "name": "sft_tool_train.jsonl",
             "pinned_sha256": "43e98ac3726ffb5e28e98a7d884cf436e046b90ea3b8783f1ad1deb1f55f2aed"},
            {"url": HF_CRAFTOGRAMMER + "sft_tool_eval.jsonl", "name": "sft_tool_eval.jsonl",
             "pinned_sha256": "191bfa3614034e9ba0a21fc21c30b35ea7edcc7f54e4f4c724fde2fb9b9bcfb5"},
        ],
    },
    # Unlabeled (the pseudo-labeling pool, scripts/s01b_rider_pool.py), read straight from the archive. Kaggle may
    # want API credentials for the download.
    "kaggle_rider_inbox": {
        "title": "Inbox dumps of 93 Indian delivery riders, 2022 (Kaggle ramniwashkumar/sms-data)",
        "homepage": "https://www.kaggle.com/datasets/ramniwashkumar/sms-data",
        "license": "CC0: Public Domain (per the Kaggle listing)",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "dir": "india/kaggle_rider_inbox",
        "files": [
            {"url": "https://www.kaggle.com/api/v1/datasets/download/ramniwashkumar/sms-data", "name": "sms-data.zip",
             "pinned_sha256": "c2ae0cac33e472bfedee341a19e18c8968d3f4b9c23a789027a5915996e563b8"},
        ],
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def get_json(sess: requests.Session, url: str):
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = sess.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            wait = min(60, 2 ** attempt)
            print(f"  ! {url}: {e} (retry in {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}: {last}")


def download(sess: requests.Session, url: str, dest: Path, expected_sha256: str | None) -> None:
    """Download `url` to `dest`, resuming from `dest.part`; verifies `expected_sha256` if given."""
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(RETRIES):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with sess.get(url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), headers=headers) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with part.open(mode) as f:
                    for block in r.iter_content(CHUNK):
                        f.write(block)
            break
        except requests.RequestException as e:
            last = e
            wait = min(60, 2 ** attempt)
            print(f"  ! {url}: {e} (resume in {wait}s)", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"giving up on {url}: {last}")

    digest = sha256_file(part)
    if expected_sha256 and digest != expected_sha256:
        part.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {url}: got {digest}, expected {expected_sha256}")
    part.replace(dest)


def safe_extract(archive: Path, out_dir: Path) -> list[Path]:
    """Extract a zip into a new, empty `out_dir`, refusing absolute paths and `..` components."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    root = out_dir.resolve()
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe path in {archive.name}: {name!r}")
            target = (out_dir / name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe path in {archive.name}: {name!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(target)
    # Nested archives (the UCI zip has none today, Mendeley's might) are extracted next to themselves.
    for inner in list(written):
        if inner.suffix.lower() == ".zip":
            written.extend(safe_extract(inner, inner.with_suffix("")))
    return written


def resolve_mendeley(sess: requests.Session, wanted: list[str]) -> list[dict]:
    listing = get_json(sess, MENDELEY_FILES_API)
    files = []
    for entry in listing:
        if entry.get("filename") in wanted:
            details = entry["content_details"]
            files.append({
                "url": details["download_url"],
                "name": entry["filename"],
                "sha256": details.get("sha256_hash"),
                "extract": entry["filename"].lower().endswith(".zip"),
            })
    missing = set(wanted) - {f["name"] for f in files}
    if missing:
        raise RuntimeError(f"Mendeley listing lacks {sorted(missing)}")
    return files


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"sources": {}}


def fetch_source(sess: requests.Session, name: str, spec: dict, previous: dict | None, force: bool) -> dict:
    src_dir = RAW_DIR / spec.get("dir", name)
    record = {
        "title": spec["title"],
        "homepage": spec["homepage"],
        "license": spec["license"],
        "license_url": spec["license_url"],
        "status": "ok",
        "files": [],
        "extracted": [],
    }
    files = list(spec["files"])
    if spec.get("mendeley_files"):
        files += resolve_mendeley(sess, spec["mendeley_files"])

    prev_files = {f["path"]: f for f in (previous or {}).get("files", [])}
    if force and src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = src_dir / f["name"]
        rel = str(dest.relative_to(RAW_DIR))
        prev = prev_files.get(rel)
        if dest.exists() and prev and sha256_file(dest) == prev["sha256"]:
            print(f"  = {rel} (cached)")
            entry = prev
        else:
            print(f"  ↓ {f['url']}")
            download(sess, f["url"], dest, f.get("sha256") or f.get("pinned_sha256"))
            entry = {
                "url": f["url"],
                "path": rel,
                "sha256": sha256_file(dest),
                "bytes": dest.stat().st_size,
                "retrieved_utc": utc_now(),
                "sha256_verified_against_host": bool(f.get("sha256")),
                "sha256_pinned": bool(f.get("pinned_sha256")),
            }
        record["files"].append(entry)
        if f.get("extract"):
            for out in safe_extract(dest, src_dir / "extracted" / dest.stem):
                if out.is_file():
                    record["extracted"].append({
                        "path": str(out.relative_to(RAW_DIR)),
                        "sha256": sha256_file(out),
                        "bytes": out.stat().st_size,
                    })
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), help="fetch only these sources")
    ap.add_argument("--force", action="store_true", help="delete and re-download the selected sources")
    args = ap.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    manifest.setdefault("sources", {})
    sess = session()
    failures = 0
    for name in args.only or SOURCES:
        print(f"[{name}]")
        try:
            manifest["sources"][name] = fetch_source(
                sess, name, SOURCES[name], manifest["sources"].get(name), args.force)
        except Exception as e:  # record and keep going: one dead mirror shouldn't block the rest
            failures += 1
            print(f"  ✗ {name} unavailable: {e}", file=sys.stderr)
            manifest["sources"][name] = {
                "title": SOURCES[name]["title"],
                "homepage": SOURCES[name]["homepage"],
                "license": SOURCES[name]["license"],
                "status": "unavailable",
                "error": str(e),
                "checked_utc": utc_now(),
            }
        manifest["generated_utc"] = utc_now()
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ML_DIR)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
