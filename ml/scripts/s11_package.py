"""s11: package the model bundle into the app (app/src/main/assets/spam_model/).

    uv run python -I scripts/s11_package.py

Copies encoder.int8.ort, head_base.bin, vocab.txt and resources/*, and writes model.json with SHA-256s for every
file. The engine (spamcore ModelBundle.load) verifies them at load time.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys

from smsml.features import META_NAMES
from smsml.hashing import BITS, SEED
from smsml.pipeline import APP_BUNDLE, BUNDLE_WORK, ML_DIR, config, sha256_file, snapshot_dir, work, write_json
from smsml.resources import FILES

ORT_VERSION = "1.27.0"


def main() -> int:
    cfg = config()
    cal = json.loads(work("calibration.json").read_text())
    pers = json.loads(work("personalization.json").read_text())
    metrics_path = work("metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    if APP_BUNDLE.exists():
        shutil.rmtree(APP_BUNDLE)
    (APP_BUNDLE / "resources").mkdir(parents=True)
    shutil.copyfile(BUNDLE_WORK / "encoder.int8.ort", APP_BUNDLE / "encoder.int8.ort")
    shutil.copyfile(BUNDLE_WORK / "head_base.bin", APP_BUNDLE / "head_base.bin")
    shutil.copyfile(snapshot_dir(cfg) / "vocab.txt", APP_BUNDLE / "vocab.txt")
    for name in FILES:
        shutil.copyfile(ML_DIR / "resources" / name, APP_BUNDLE / "resources" / name)

    best = pers["best"]["params"]
    pc = cfg["personalization"]
    head_sha = sha256_file(APP_BUNDLE / "head_base.bin")
    now = dt.datetime.now(dt.timezone.utc)
    personalization = {
        "priors": {k: best[k] for k in ("embed", "meta", "hash")},
        "weights": {**pc["weights"], "ham": best["ham"]},
        "debounce_ms": pc["debounce_ms"],
    }
    knn = {"tau_spam": best["tau_spam"], "tau_ham": cal["knn"]["tau_ham"]}
    # The version covers everything that shapes learned weights, so the phone retrains them when any of it changes.
    shaping = json.dumps({"head": head_sha, "personalization": personalization, "knn": knn,
                          "thresholds": cal["thresholds"]}, sort_keys=True)
    version_hash = hashlib.sha256(shaping.encode("utf-8")).hexdigest()[:8]
    model = {
        "schema": 1,
        "bundle_version": f"{now:%Y.%m.%d}-{version_hash}",
        "created_utc": now.isoformat(timespec="seconds"),
        "encoder": {
            "file": "encoder.int8.ort", "sha256": sha256_file(APP_BUNDLE / "encoder.int8.ort"), "dim": 384,
            "max_len": int(cfg["model"]["max_len"]), "ort_version": ORT_VERSION,
            "inputs": ["input_ids", "attention_mask"], "output": "embedding",
            "base_model": f"{cfg['model']['id']}@{cfg['model']['revision']}", "quant": "dynamic-qint8-per-channel",
        },
        "tokenizer": {"file": "vocab.txt", "sha256": sha256_file(APP_BUNDLE / "vocab.txt")},
        "normalizer": {"version": "norm_v1",
                       "resources": {name: sha256_file(APP_BUNDLE / "resources" / name) for name in FILES}},
        "features": {"version": "feat_v1", "hash_bits": BITS, "hash_seed": SEED, "meta_names": list(META_NAMES)},
        "head": {"file": "head_base.bin", "sha256": head_sha},
        "thresholds": cal["thresholds"],
        "personalization": personalization,
        "knn": knn,
        "metrics": {k: metrics[k] for k in ("test", "future", "int8", "learning", "gates") if k in metrics},
    }
    write_json(APP_BUNDLE / "model.json", model)
    size = sum(p.stat().st_size for p in APP_BUNDLE.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {APP_BUNDLE} ({size:.1f} MB): bundle {model['bundle_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
