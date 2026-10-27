"""s01b: the unlabeled pool for pseudo-labeling, from the riders' inboxes (sources/kaggle_rider_inbox.py).

    uv run python -I scripts/s01b_rider_pool.py

Writes data/work/rider_pool.parquet: one row per distinct (sender, text) with sender_type, n (messages) and users
(distinct riders). It isn't part of the corpus: s01 never reads it.
"""

from __future__ import annotations

import sys

from smsml.pipeline import work
from smsml.sources import RAW_DIR, kaggle_rider_inbox


def main() -> int:
    try:
        df = kaggle_rider_inbox.pool(RAW_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}; run scripts/s00_fetch.py --only kaggle_rider_inbox", file=sys.stderr)
        return 1
    out = work("rider_pool.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} distinct (sender, text) from {int(df['n'].sum())} messages;",
          df["sender_type"].value_counts(dropna=False).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
