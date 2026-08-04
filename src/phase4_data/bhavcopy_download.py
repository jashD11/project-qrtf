"""
Phase 4 acquisition — bulk-download NSE bhavcopy and index archives.

NSE publishes one file per trading day covering **every listed stock**, so the
download cost scales with *days*, not *stocks*: a 68-name build and a 2,000-name
build cost exactly the same. Three archives are needed, on ``nsearchives.nseindia.com``,
which serves them to a plain ``User-Agent`` with no cookies and no session priming
(unlike the ``www.nseindia.com/api/*`` surface, which is cookie-gated and rate-limited
— this module deliberately never touches it):

    eq     /content/historical/EQUITIES/<YYYY>/<MON>/cm<DD><MON><YYYY>bhav.csv.zip
           legacy bhavcopy, through ~2024-07. OHLCV + ISIN + turnover + trade count.
    udiff  /content/cm/BhavCopy_NSE_CM_0_0_0_<YYYYMMDD>_F_0000.csv.zip
           the replacement format, from ~2024-07. Same content, new column names.
    index  /content/indices/ind_close_all_<DDMMYYYY>.csv
           every NSE index close for the day; the Tier 2 regime market series.

Design notes
    - **Era router by probe, not by constant.** The eq/udiff cutover is tried in the
      likely order for the date and falls back to the other format, so the exact
      changeover day never has to be hard-coded (and a date served by both still
      resolves).
    - **Idempotent and resumable.** An existing non-empty target is never re-fetched,
      so a killed overnight run resumes for free.
    - **Holidays vs outages stay distinguishable.** A 404 is definitive absence (market
      holiday) and is recorded in ``_missing.json`` so later runs skip it without a
      request. Any other failure is transient, recorded with its status, and retried
      on the next run. ``--recheck-missing`` forces even the 404s to be re-probed.
    - **Paced.** Zero-delay bursts were observed to fail silently; 3 s spacing was
      reliable across probing. Default 2.5 s with retry/backoff.
    - **Bodies are verified, not assumed.** NSE can answer a missing file with a 200
      HTML error page; ZIPs must start with ``PK`` and CSVs must not look like HTML,
      so a decoy never lands on disk as a valid-looking file.

Usage
    python src/phase4_data/bhavcopy_download.py --list-only            # plan, no requests
    python src/phase4_data/bhavcopy_download.py --start 2013-01-01 --end 2013-01-10
    python src/phase4_data/bhavcopy_download.py                        # full window (~5 h)
    python src/phase4_data/bhavcopy_download.py --kinds eq             # prices only
"""

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta
from typing import Final

import requests

# ── Window ──────────────────────────────────────────────────────────────────────
# 2013-01-01 is the later of two hard archive boundaries: ISIN (the survivorship key)
# appears in bhavcopy from 2012, but the index archive only begins 2013-01-02, and
# Tier 2 needs a market series across the whole panel. See docs/phase4_plan.md.
DEFAULT_START: Final[str] = "2013-01-01"
DEFAULT_END: Final[str] = "2025-06-30"

# ── Pacing ──────────────────────────────────────────────────────────────────────
DEFAULT_DELAY_S: Final[float] = 2.5
DEFAULT_RETRIES: Final[int] = 3
_BACKOFF_BASE_S: Final[float] = 5.0
_TIMEOUT_S: Final[float] = 30.0

BASE: Final[str] = "https://nsearchives.nseindia.com"
_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

DEFAULT_OUT_DIR: Final[str] = "data/bhavcopy"
MISSING_JSON: Final[str] = "_missing.json"

KINDS: Final[tuple[str, ...]] = ("eq", "index")
_ERAS: Final[tuple[str, ...]] = ("eq", "udiff")  # both are the "eq" kind on disk

# Format changeover, used only to order the two candidate URLs — never to decide
# that one era is impossible. A wrong guess costs one extra request on one day.
_UDIFF_FROM: Final[date] = date(2024, 7, 1)

_MONTHS: Final[tuple[str, ...]] = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)


# ── URL / filename construction ─────────────────────────────────────────────────

def _eq_old(d: date) -> tuple[str, str]:
    """Legacy bhavcopy: (subdir, url). Filename mixes zero-padded day + MON + year."""
    mon = _MONTHS[d.month - 1]
    name = f"cm{d.day:02d}{mon}{d.year}bhav.csv.zip"
    return "eq", f"{BASE}/content/historical/EQUITIES/{d.year}/{mon}/{name}"


def _eq_udiff(d: date) -> tuple[str, str]:
    """UDiFF bhavcopy: (subdir, url)."""
    name = f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return "udiff", f"{BASE}/content/cm/{name}"


def _index(d: date) -> tuple[str, str]:
    """All-index close file: (subdir, url). Note DDMMYYYY, unlike UDiFF's YYYYMMDD."""
    name = f"ind_close_all_{d:%d%m%Y}.csv"
    return "index", f"{BASE}/content/indices/{name}"


def candidates(kind: str, d: date) -> list[tuple[str, str]]:
    """
    Candidate ``(subdir, url)`` pairs for one (kind, date), most likely first.

    The eq kind returns both era formats so the archive's own cutover decides which
    one exists, rather than a hard-coded date in this file.
    """
    if kind == "index":
        return [_index(d)]
    if d >= _UDIFF_FROM:
        return [_eq_udiff(d), _eq_old(d)]
    return [_eq_old(d), _eq_udiff(d)]


def target_path(out_dir: str, subdir: str, url: str) -> str:
    """Local path for a candidate — raw archive files are stored as received."""
    return os.path.join(out_dir, subdir, url.rsplit("/", 1)[-1])


# ── Body validation ─────────────────────────────────────────────────────────────

def _body_ok(url: str, body: bytes) -> bool:
    """
    True when the payload really is the archive file, not an HTML error page.

    NSE occasionally answers a missing file with 200 + HTML. Without this check such
    a page would be written to disk, skipped as "already downloaded" on every later
    run, and only surface as a parse failure hours into the build.
    """
    if len(body) < 64:
        return False
    if url.endswith(".zip"):
        return body[:2] == b"PK"
    head = body[:256].lstrip().lower()
    return not (head.startswith(b"<") or b"<html" in head)


# ── Missing ledger ──────────────────────────────────────────────────────────────

def _load_missing(path: str) -> dict[str, int]:
    """Read the {kind:YYYY-MM-DD → http status} ledger; empty when absent/corrupt."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        return {str(k): int(v) for k, v in data.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        print(f"[bhavcopy_download] WARN: unreadable {path} — starting a fresh ledger")
        return {}


def _save_missing(path: str, missing: dict[str, int]) -> None:
    """Atomically persist the ledger (write-then-rename survives a mid-write kill)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(sorted(missing.items())), fh, indent=1)
    os.replace(tmp, path)


# ── Fetch ───────────────────────────────────────────────────────────────────────

def _get(session: requests.Session, url: str, retries: int) -> tuple[int, bytes | None]:
    """
    Fetch one URL. Returns ``(status, body)``; body is None unless the fetch succeeded.

    A 404 returns immediately — it means the file is genuinely absent (holiday, or the
    wrong era for this date) and retrying only burns rate budget. Everything else
    (timeouts, resets, 5xx) is retried with exponential backoff.
    """
    status = 0
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=_HEADERS, timeout=_TIMEOUT_S)
            status = resp.status_code
            if status == 404:
                return 404, None
            if status == 200 and _body_ok(url, resp.content):
                return 200, resp.content
            if status == 200:
                status = -200  # 200 OK but the body was an HTML decoy
        except requests.RequestException:
            status = -1
        if attempt < retries:
            time.sleep(_BACKOFF_BASE_S * (2 ** (attempt - 1)))
    return status, None


def _fetch_one(
    session: requests.Session,
    kind: str,
    d: date,
    out_dir: str,
    retries: int,
    delay: float,
) -> tuple[str, int]:
    """
    Resolve one (kind, date) to a file on disk.

    Returns ``(outcome, status)`` where outcome is ``"skip"`` (already present),
    ``"ok"``, ``"missing"`` (every candidate 404'd — a holiday), or ``"fail"``.
    Candidates are tried in order, so the eq kind self-discovers its era; only the
    *between*-candidate wait is charged here, the caller paces the outer loop.
    """
    cands = candidates(kind, d)

    for subdir, url in cands:
        path = target_path(out_dir, subdir, url)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return "skip", 200

    last_status = 0
    for i, (subdir, url) in enumerate(cands):
        status, body = _get(session, url, retries)
        if status == 200 and body is not None:
            path = target_path(out_dir, subdir, url)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.part"
            with open(tmp, "wb") as fh:
                fh.write(body)
            os.replace(tmp, path)  # a partial file never looks complete to a resume
            return "ok", 200
        last_status = status
        if status != 404:
            break  # a transient failure is not evidence about the other era
        if i + 1 < len(cands):
            time.sleep(delay)

    return ("missing", 404) if last_status == 404 else ("fail", last_status)


# ── Driver ──────────────────────────────────────────────────────────────────────

def trading_dates(start: date, end: date, weekdays_only: bool = False) -> list[date]:
    """
    Candidate session dates for the window — **every calendar day** by default.

    Weekends are scanned deliberately, because NSE does trade on some of them and a
    missed session silently merges two days of return into one bar:

      * **Muhurat trading** — a ceremonial Diwali session, which falls on a Sunday
        roughly every third year (2013-11-03, 2016-10-30, 2019-10-27, 2023-11-12).
      * **Budget and special sessions** — e.g. Saturday 2025-02-01.
      * Saturday disaster-recovery live sessions.

    A weekday-only scan looked obviously right and cost real data: TATAMOTORS read
    +36.4% on 2019-10-29 instead of +15.3%, because the Sunday Muhurat bar in between
    was never downloaded.

    Non-session days simply 404, cost one request each, and are then remembered in the
    missing ledger so a re-run skips them for free. ``weekdays_only`` is kept for a
    fast partial scan, not for production.
    """
    out, d = [], start
    while d <= end:
        if not weekdays_only or d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def download(
    start: date,
    end: date,
    out_dir: str = DEFAULT_OUT_DIR,
    kinds: tuple[str, ...] = KINDS,
    delay: float = DEFAULT_DELAY_S,
    retries: int = DEFAULT_RETRIES,
    recheck_missing: bool = False,
    weekdays_only: bool = False,
) -> dict[str, int]:
    """
    Download every requested archive over ``[start, end]``, interleaved by date.

    Interleaving (all kinds for one date, then the next date) rather than one full
    pass per kind means the panel and the index advance together, so an interrupted
    run leaves a usable prefix of *both* streams instead of all of one and none of
    the other. Total request count is identical either way.

    Returns the tally dict; the missing ledger is flushed even if the run is killed.
    """
    dates = trading_dates(start, end, weekdays_only=weekdays_only)
    missing_path = os.path.join(out_dir, MISSING_JSON)
    missing = _load_missing(missing_path)

    tally = {"ok": 0, "skip": 0, "missing": 0, "fail": 0, "known": 0}
    n_req = 0
    t0 = time.time()

    print(
        f"[bhavcopy_download] {start} → {end} | {len(dates)} days × "
        f"{len(kinds)} kinds = {len(dates) * len(kinds)} slots | delay={delay}s"
    )
    if missing:
        print(f"[bhavcopy_download] ledger: {len(missing)} known-absent slots"
              f"{' (rechecking)' if recheck_missing else ' (skipped)'}")

    session = requests.Session()
    try:
        for i, d in enumerate(dates, start=1):
            marks = []
            for kind in kinds:
                key = f"{kind}:{d.isoformat()}"

                if key in missing and not recheck_missing:
                    tally["known"] += 1
                    marks.append(f"{kind}=holiday")
                    continue

                outcome, status = _fetch_one(session, kind, d, out_dir, retries, delay)
                tally[outcome] += 1
                if outcome in ("ok", "missing", "fail"):
                    n_req += 1
                if outcome == "missing":
                    missing[key] = 404
                    marks.append(f"{kind}=holiday")
                elif outcome == "fail":
                    missing[key] = status
                    marks.append(f"{kind}=FAIL({status})")
                else:
                    missing.pop(key, None)  # recovered — stop remembering it as absent
                    marks.append(f"{kind}={'ok' if outcome == 'ok' else 'have'}")

                if outcome != "skip":
                    time.sleep(delay)

            if i % 25 == 0 or i == len(dates):
                done = tally["ok"] + tally["skip"] + tally["known"]
                rate = n_req / max(time.time() - t0, 1e-9)
                eta = (len(dates) - i) * len(kinds) * delay / 60.0
                print(
                    f"[{i}/{len(dates)}] {d}  {' '.join(marks)}  | "
                    f"have={done} miss={tally['missing']} fail={tally['fail']} | "
                    f"{rate:.2f} req/s | ~{eta:.0f} min left"
                )
                _save_missing(missing_path, missing)
    except KeyboardInterrupt:
        print("\n[bhavcopy_download] interrupted — progress is on disk, re-run to resume")
    finally:
        _save_missing(missing_path, missing)

    mins = (time.time() - t0) / 60.0
    print(
        f"\n[bhavcopy_download] done in {mins:.1f} min → {out_dir} | "
        f"new={tally['ok']} already={tally['skip']} holiday={tally['missing']}"
        f"(+{tally['known']} known) failed={tally['fail']}"
    )
    if tally["fail"]:
        print("[bhavcopy_download] Some slots failed — re-run to retry (idempotent).")
    return tally


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download NSE bhavcopy + index archives.")
    ap.add_argument("--start", default=DEFAULT_START, help="first date (YYYY-MM-DD)")
    ap.add_argument("--end", default=DEFAULT_END, help="last date (YYYY-MM-DD)")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    ap.add_argument(
        "--kinds", default=",".join(KINDS),
        help=f"comma-separated subset of {KINDS} (default: both, interleaved)",
    )
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="seconds between requests")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="attempts per URL")
    ap.add_argument(
        "--weekdays-only", action="store_true",
        help="skip weekends (fast partial scan; misses Muhurat/budget sessions)",
    )
    ap.add_argument(
        "--recheck-missing", action="store_true",
        help="re-probe slots previously recorded as absent (default: skip them)",
    )
    ap.add_argument("--list-only", action="store_true", help="print the plan, make no requests")
    args = ap.parse_args()

    start, end = _parse_date(args.start), _parse_date(args.end)
    if start > end:
        raise SystemExit(f"--start {start} is after --end {end}")

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise SystemExit(f"unknown --kinds {bad}; choose from {list(KINDS)}")

    if args.list_only:
        dates = trading_dates(start, end, weekdays_only=args.weekdays_only)
        slots = len(dates) * len(kinds)
        print(
            f"[bhavcopy_download] plan: {start} → {end} | {len(dates)} days × "
            f"{len(kinds)} kinds = {slots} slots\n"
            f"  ≈ {slots * args.delay / 3600:.1f} h at {args.delay}s "
            f"(minus holidays and anything already on disk)"
        )
        for d in (dates[:3] + dates[-2:]) if len(dates) > 5 else dates:
            for kind in kinds:
                for subdir, url in candidates(kind, d):
                    print(f"  {d} {subdir:<5} {url}")
        return

    download(
        start, end, out_dir=args.out, kinds=kinds, delay=args.delay,
        retries=args.retries, recheck_missing=args.recheck_missing,
        weekdays_only=args.weekdays_only,
    )


if __name__ == "__main__":
    main()
