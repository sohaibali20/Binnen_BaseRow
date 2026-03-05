"""
Translate English entries to Dutch (Netherlands).
Reads english_entries_all.json, translates only the values (keys and row_id unchanged),
writes dutch_entries_all.json (or dutch_entries_first_N.json when using --limit N).

Uses Google Translate unofficial API directly via `requests` (free, no API key).
The requests call uses a real socket-level timeout so it NEVER hangs.

Features:
  - Real socket-level timeout (no more hanging forever)
  - Automatic retry with exponential backoff on rate limits / timeouts / errors
  - Checkpoint: saves after every row; restarts skip already-translated rows
  - Clear per-entry + per-column progress output

Example:
    python translate_to_dutch.py --limit 10   # test with first 10 entries
    python translate_to_dutch.py              # translate all
"""

import argparse
import json
import time
import urllib.parse
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_CHUNK_LEN   = 4500           # chars per translate call
DELAY_SECONDS   = 1.5            # pause between every successful translate call
RETRY_DELAYS    = [3, 6, 12, 24] # backoff between retries (seconds)
REQUEST_TIMEOUT = 12             # socket-level timeout per HTTP request (seconds)

INPUT_FILE  = Path(__file__).resolve().parent / "english_entries_all.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "dutch_entries_all.json"

# Google Translate unofficial endpoint (same one used by deep-translator internally)
_GT_URL = "https://translate.googleapis.com/translate_a/single"
_GT_PARAMS_BASE = {
    "client":   "gtx",
    "sl":       "en",
    "tl":       "nl",
    "dt":       "t",
}


# ---------------------------------------------------------------------------
# Core translation (direct requests, real timeout)
# ---------------------------------------------------------------------------

def _google_translate(text: str) -> str:
    """
    Call the Google Translate unofficial endpoint with a socket-level timeout.
    Returns the translated string, or raises on failure.
    """
    params = {**_GT_PARAMS_BASE, "q": text}
    resp = requests.get(
        _GT_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    data = resp.json()
    # Response structure: [ [ [translated, original, ...], ... ], ... ]
    parts = [seg[0] for seg in data[0] if seg[0]]
    return "".join(parts)


def _translate_with_retry(text: str, row_id, col_name: str) -> str:
    """
    Translate one chunk with timeout + exponential backoff retry.
    Returns original text if all retries fail.
    """
    for attempt, wait in enumerate([0] + RETRY_DELAYS, start=1):
        if wait:
            print(f"        ↻ retry {attempt}/{len(RETRY_DELAYS)+1} — waiting {wait}s ...", flush=True)
            time.sleep(wait)
        try:
            return _google_translate(text)
        except requests.exceptions.Timeout:
            print(f"        ⚠ [row {row_id} | {col_name}] attempt {attempt} timed out after {REQUEST_TIMEOUT}s", flush=True)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"        ⚠ [row {row_id} | {col_name}] attempt {attempt} HTTP {status} error", flush=True)
            if status == 429:
                # Rate limited — wait longer than usual
                extra = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)] * 2
                print(f"        Rate-limited (429). Extra wait: {extra}s ...", flush=True)
                time.sleep(extra)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"        ⚠ [row {row_id} | {col_name}] attempt {attempt} error: {str(e)[:120]}", flush=True)

    print(f"        ✗ [row {row_id} | {col_name}] all retries exhausted — keeping original.", flush=True)
    return text


def _chunk_text(text: str, max_len: int = MAX_CHUNK_LEN) -> list[str]:
    """Split long text into chunks at paragraph/sentence boundaries."""
    if len(text) <= max_len:
        return [text] if text.strip() else []
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_len:
            chunks.append(rest)
            break
        segment = rest[: max_len + 1]
        for sep in ("\n\n", "\n", ". ", " "):
            idx = segment.rfind(sep)
            if idx != -1:
                chunk = rest[: idx + len(sep)].strip()
                if chunk:
                    chunks.append(chunk)
                rest = rest[idx + len(sep) :].strip()
                break
        else:
            chunks.append(rest[:max_len])
            rest = rest[max_len:].strip()
    return [c for c in chunks if c]


def _translate_value(value, row_id, col_name: str) -> str:
    """Translate a column value (chunking long text if needed)."""
    if value is None:
        return value
    s = str(value).strip()
    if not s:
        return value

    chunks = _chunk_text(s)
    if not chunks:
        return value

    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        parts.append(_translate_with_retry(chunk, row_id, col_name))
        if i < len(chunks) - 1:
            time.sleep(DELAY_SECONDS)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_existing(out_file: Path) -> dict:
    """Load already-translated row_ids from the output file."""
    if not out_file.exists():
        return {}
    try:
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(e["row_id"]): e["dutch_columns"] for e in data if "row_id" in e}
    except Exception:
        return {}


def _save_result(out_file: Path, result: list[dict]) -> None:
    """Atomically write the result list to disk via a .tmp swap."""
    tmp = out_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(out_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: int | None = None) -> None:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    entries = all_entries[:limit] if limit is not None else all_entries
    out_file = (
        Path(__file__).resolve().parent / f"dutch_entries_first_{limit}.json"
        if limit is not None
        else OUTPUT_FILE
    )

    # Load checkpoint
    already_done = _load_existing(out_file)
    result: list[dict] = [
        {"row_id": rid, "dutch_columns": cols}
        for rid, cols in already_done.items()
    ]
    if already_done:
        print(f"Checkpoint: {len(already_done)} already-translated row(s) skipped.", flush=True)

    to_process = [e for e in entries if str(e.get("row_id", "")) not in already_done]
    total      = len(entries)
    done_count = len(already_done)

    if not to_process:
        print("All entries already translated. Nothing to do!", flush=True)
        return

    print(f"Translator     : Google Translate (free, direct API)", flush=True)
    print(f"Total entries  : {total}", flush=True)
    print(f"Already done   : {done_count}", flush=True)
    print(f"Remaining      : {len(to_process)}", flush=True)
    print(f"Timeout/call   : {REQUEST_TIMEOUT}s   Delay between calls: {DELAY_SECONDS}s", flush=True)
    print("-" * 60, flush=True)

    for i, entry in enumerate(to_process, start=1):
        global_idx = done_count + i
        row_id     = entry.get("row_id", "(no id)")
        eng_cols   = entry.get("english_columns") or {}
        non_empty  = {k: v for k, v in eng_cols.items() if v and str(v).strip()}
        num_cols   = len(non_empty)

        print(f"\n[{global_idx:>4}/{total}] row_id={row_id}  ({num_cols} column(s) to translate)", flush=True)
        t_start = time.time()

        dutch_columns: dict = {}
        col_num = 0
        for key, value in eng_cols.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                dutch_columns[key] = value
                continue
            col_num += 1
            print(f"      [{col_num}/{num_cols}] {key} ...", flush=True)
            dutch_columns[key] = _translate_value(value, row_id, key)
            time.sleep(DELAY_SECONDS)

        elapsed = time.time() - t_start
        result.append({"row_id": row_id, "dutch_columns": dutch_columns})
        _save_result(out_file, result)
        print(f"      ✓ done in {elapsed:.1f}s  →  saved to {out_file.name}", flush=True)

    print("-" * 60, flush=True)
    print(f"Done. {len(result)} entries total → {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English entries to Dutch (NL).")
    parser.add_argument(
        "--limit", "-n",
        type=int, default=None,
        help="Only translate first N entries (e.g. 10 for testing)",
    )
    args = parser.parse_args()
    main(limit=args.limit)
