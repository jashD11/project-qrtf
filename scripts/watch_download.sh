#!/usr/bin/env bash
# Periodic progress snapshot for the Phase 4 bhavcopy download.
#
# Appends one line every INTERVAL seconds to logs/bhavcopy_progress.log, and exits on
# its own once the download process is gone — so it never outlives the job it watches.
# Read-only: it only tails the download log and stats the output directory.
#
#   ./scripts/watch_download.sh            # 15-minute ticks (default)
#   ./scripts/watch_download.sh 300        # 5-minute ticks
#   tail -f logs/bhavcopy_progress.log     # follow it

set -uo pipefail

INTERVAL="${1:-900}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/logs/bhavcopy_download.log"
OUT="$ROOT/logs/bhavcopy_progress.log"
RAW="$ROOT/data/bhavcopy"

mkdir -p "$ROOT/logs"

# Match the downloader by its script path so an unrelated python process never
# looks like the job, and a stale PID never reads as alive.
pgrep -f "bhavcopy_download.py" >/dev/null 2>&1 || {
    echo "[watch] no bhavcopy_download.py process running — nothing to watch" >&2
    exit 1
}

printf '[watch] started %s | interval %ss | → %s\n' "$(date '+%F %T')" "$INTERVAL" "$OUT"
printf '\n=== watch started %s (every %ss) ===\n' "$(date '+%F %T')" "$INTERVAL" >> "$OUT"

while true; do
    alive=no
    pgrep -f "bhavcopy_download.py" >/dev/null 2>&1 && alive=yes

    # Last progress line the downloader emitted, e.g.
    #   [300/3260] 2014-02-24  eq=ok index=ok  | have=573 miss=27 fail=0 | ...
    line="$(grep -E '^\[[0-9]+/[0-9]+\]' "$LOG" 2>/dev/null | tail -1)"

    if [[ -n "$line" ]]; then
        done_n="$(sed -E 's#^\[([0-9]+)/.*#\1#' <<< "$line")"
        total="$(sed -E 's#^\[[0-9]+/([0-9]+)\].*#\1#' <<< "$line")"
        at="$(awk '{print $2}' <<< "$line")"
        stats="$(sed -E 's#.*\| (have=[^|]*)\|.*#\1#' <<< "$line")"
        eta="$(sed -E 's#.*~([0-9]+) min left#\1#' <<< "$line")"
        pct="$(awk -v d="$done_n" -v t="$total" 'BEGIN{printf "%.1f", (t?100*d/t:0)}')"
    else
        done_n=0; total='?'; at='-'; stats='-'; eta='?'; pct='0.0'
    fi

    files=$(find "$RAW" -type f \( -name '*.zip' -o -name '*.csv' \) 2>/dev/null | wc -l | tr -d ' ')
    size=$(du -sh "$RAW" 2>/dev/null | awk '{print $1}')

    printf '%s | %s/%s (%s%%) at %s | %s| files=%s size=%s | eta=%smin | alive=%s\n' \
        "$(date '+%F %T')" "$done_n" "$total" "$pct" "$at" "$stats" \
        "$files" "${size:-0}" "$eta" "$alive" >> "$OUT"

    if [[ "$alive" == no ]]; then
        {
            printf '=== download process gone at %s ===\n' "$(date '+%F %T')"
            tail -3 "$LOG" 2>/dev/null
            printf '=== watch exiting ===\n'
        } >> "$OUT"
        exit 0
    fi

    sleep "$INTERVAL"
done
