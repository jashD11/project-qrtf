"""
Phase 2 data acquisition — bulk-download the NSE 15-minute OHLCV CSVs from the
shared Google Drive folder.

The Drive folder ("15m_dataset") holds one CSV per (symbol, exchange), named
``<SYMBOL>.<EXCHANGE>_15minute_ohlcv.csv`` (~176 files: every symbol dual-listed
on .NSE and .BSE). This script enumerates the folder, keeps only the **.NSE**
files, and downloads them into ``data/raw_15min/`` (gitignored).

Design notes
    - Enumeration uses gdown's ``download_folder(skip_download=True)`` so the
      whole listing is fetched in one call without pulling any bytes; the
      historical 50-file folder cap does not apply.
    - Idempotent: an existing non-empty target file is skipped, so re-running
      resumes an interrupted pull rather than re-downloading.
    - Per-file failures (a flaky transfer, a permissions hiccup) are caught and
      counted, never aborting the whole run.

Usage
    python src/production_ml/download_nse.py                 # download all NSE files
    python src/production_ml/download_nse.py --list-only      # enumerate only, no download
    python src/production_ml/download_nse.py --limit 3        # download first 3 (smoke test)
    python src/production_ml/download_nse.py --out data/raw_15min --exchange NSE

Requires ``gdown`` (see requirements.txt). The Drive folder must be shared as
"anyone with the link".
"""

import argparse
import os
import sys
import time
from typing import Final

import gdown

# Pace requests to stay under Google Drive's anonymous rate limit (which soft-
# blocks after ~50 rapid pulls). A small inter-file delay plus per-file retry
# with exponential backoff lets an unauthenticated run finish all files.
DEFAULT_DELAY_S: Final[float] = 1.5
DEFAULT_RETRIES: Final[int] = 4
_BACKOFF_BASE_S: Final[float] = 8.0

FOLDER_URL: Final[str] = (
    "https://drive.google.com/drive/folders/1nTR20NjidrUKBqld--dkva3Xgfonb-yT"
)
DEFAULT_OUT_DIR: Final[str] = "data/raw_15min"
DEFAULT_EXCHANGE: Final[str] = "NSE"


def enumerate_folder() -> list:
    """
    Return the full list of files in the Drive folder without downloading them.

    Each element is a gdown ``GoogleDriveFileToDownload(id, path, local_path)``.
    """
    files = gdown.download_folder(
        FOLDER_URL, skip_download=True, quiet=True, use_cookies=False
    )
    return files or []


def filter_exchange(files: list, exchange: str) -> list:
    """Keep only ``<SYMBOL>.<exchange>_15minute_ohlcv.csv`` files."""
    token = f".{exchange.upper()}_"
    return [f for f in files if token in os.path.basename(f.path)]


def _download_one(file_id: str, target: str, retries: int) -> bool:
    """Download a single file with exponential-backoff retries. True on success."""
    for attempt in range(1, retries + 1):
        try:
            gdown.download(id=file_id, output=target, quiet=True)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                return True
        except Exception as exc:  # noqa: BLE001 — retry, then give up gracefully
            last = f"{type(exc).__name__}"
            if attempt < retries:
                wait = _BACKOFF_BASE_S * (2 ** (attempt - 1))
                print(f"          retry {attempt}/{retries - 1} in {wait:.0f}s ({last}) …")
                time.sleep(wait)
    return False


def download(
    files: list,
    out_dir: str,
    limit: int = 0,
    delay: float = DEFAULT_DELAY_S,
    retries: int = DEFAULT_RETRIES,
) -> None:
    """
    Download each file into ``out_dir`` under its original basename.

    Paced (``delay`` between files) and resilient (``retries`` with exponential
    backoff per file) so an unauthenticated run stays under Google's rate limit.

    Args:
        files:   filtered list of GoogleDriveFileToDownload entries.
        out_dir: destination directory (created if missing).
        limit:   if > 0, download at most this many files (smoke testing).
        delay:   seconds to sleep between file downloads.
        retries: attempts per file before marking it failed.
    """
    os.makedirs(out_dir, exist_ok=True)
    todo = files[:limit] if limit > 0 else files

    n_ok = n_skip = n_fail = 0
    total = len(todo)
    for i, f in enumerate(todo, start=1):
        name = os.path.basename(f.path)
        target = os.path.join(out_dir, name)

        if os.path.exists(target) and os.path.getsize(target) > 0:
            n_skip += 1
            print(f"[{i}/{total}] skip (exists)  {name}")
            continue

        if _download_one(f.id, target, retries):
            n_ok += 1
            print(f"[{i}/{total}] ok            {name}")
        else:
            n_fail += 1
            print(f"[{i}/{total}] FAIL          {name}")

        if i < total:
            time.sleep(delay)

    print(
        f"\n[download_nse] done → {out_dir} | "
        f"ok={n_ok} skipped={n_skip} failed={n_fail} (of {total})"
    )
    if n_fail:
        print("[download_nse] Some files failed — re-run to retry (idempotent).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download NSE 15-min OHLCV CSVs from Drive.")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    ap.add_argument("--exchange", default=DEFAULT_EXCHANGE, help="exchange suffix to keep")
    ap.add_argument("--limit", type=int, default=0, help="download at most N files (0=all)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="seconds between files")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="attempts per file")
    ap.add_argument("--list-only", action="store_true", help="enumerate + filter, no download")
    args = ap.parse_args()

    print(f"[download_nse] Enumerating folder …")
    all_files = enumerate_folder()
    kept = filter_exchange(all_files, args.exchange)
    print(
        f"[download_nse] folder has {len(all_files)} files | "
        f"{args.exchange.upper()} matches: {len(kept)}"
    )

    if args.list_only:
        for f in kept[:10]:
            print(f"  {os.path.basename(f.path)}")
        if len(kept) > 10:
            print(f"  … and {len(kept) - 10} more")
        return

    if not kept:
        print(f"[download_nse] No {args.exchange.upper()} files found — nothing to do.")
        sys.exit(1)

    download(kept, args.out, limit=args.limit, delay=args.delay, retries=args.retries)


if __name__ == "__main__":
    main()
