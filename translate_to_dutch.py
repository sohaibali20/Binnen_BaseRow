"""
Translate English entries to Dutch (Netherlands).
Reads english_entries_all.json, translates only the values (keys and row_id unchanged),
writes dutch_entries_all.json (or dutch_entries_first_N.json when using --limit N).

Uses Google Translate unofficial API directly via `requests` (free, no API key).
BATCHING ARCHITECTURE:
    Instead of translating column-by-column, this joins columns (and even rows!) 
    up to 4,000 characters using a hidden separator `===||===`. This is then 
    sent in a single HTTP request, making the script 5x-10x faster and avoiding 
    rate limits.

Example:
    python translate_to_dutch.py --limit 10   # test with first 10 entries
    python translate_to_dutch.py              # translate all
"""

import argparse
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_BATCH_LEN   = 4000           # max chars per batch request (Google limit ~5000)
DELAY_SECONDS   = 1.0            # pause between successful batches
RETRY_DELAYS    = [3, 6, 12, 24] # backoff between retries (seconds)
REQUEST_TIMEOUT = 12             # socket-level timeout per HTTP request (seconds)

# Separator for chunking. Google Translate generally respects this pattern.
SEP           = "\n\n===||===\n\n"
SEP_CLEAN     = "===||==="

INPUT_FILE  = Path(__file__).resolve().parent / "english_entries_all.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "dutch_entries_all.json"

_GT_URL = "https://translate.googleapis.com/translate_a/single"
_GT_PARAMS_BASE = {
    "client":   "gtx",
    "sl":       "en",
    "tl":       "nl",
    "dt":       "t",
}


# ---------------------------------------------------------------------------
# Core translation
# ---------------------------------------------------------------------------

def _google_translate(text: str) -> str:
    """Call Google Translate API with timeouts."""
    params = {**_GT_PARAMS_BASE, "q": text}
    resp = requests.get(
        _GT_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    data = resp.json()
    parts = [seg[0] for seg in data[0] if seg[0]]
    return "".join(parts)


def _translate_batch_with_retry(batch_text: str, batch_num: int) -> str:
    """Translate a text chunk with timeout + exponential backoff."""
    for attempt, wait in enumerate([0] + RETRY_DELAYS, start=1):
        if wait:
            print(f"        ↻ retry {attempt}/{len(RETRY_DELAYS)+1} — waiting {wait}s ...", flush=True)
            time.sleep(wait)
        try:
            return _google_translate(batch_text)
        except requests.exceptions.Timeout:
            print(f"        ⚠ [Batch {batch_num}] attempt {attempt} timed out after {REQUEST_TIMEOUT}s", flush=True)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"        ⚠ [Batch {batch_num}] attempt {attempt} HTTP {status} error", flush=True)
            if status == 429:
                extra = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)] * 2
                print(f"        Rate-limited (429). Extra wait: {extra}s ...", flush=True)
                time.sleep(extra)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"        ⚠ [Batch {batch_num}] attempt {attempt} error: {str(e)[:120]}", flush=True)

    print(f"        ✗ [Batch {batch_num}] all retries exhausted — keeping original.", flush=True)
    return batch_text


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_existing(out_file: Path) -> dict:
    if not out_file.exists():
        return {}
    try:
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Only consider a row "done" if it has actual translated columns.
        return {
            str(e["row_id"]): e["dutch_columns"] 
            for e in data 
            if "row_id" in e and e.get("dutch_columns")
        }
    except Exception:
        return {}


def _save_result(out_file: Path, result: list[dict]) -> None:
    tmp = out_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    tmp.replace(out_file)

# ---------------------------------------------------------------------------
# Main Routine (Batching engine)
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

    already_done = _load_existing(out_file)
    result_map: dict[str, dict] = {rid: cols for rid, cols in already_done.items()}
    if already_done:
        print(f"Checkpoint: {len(already_done)} already-translated row(s) skipped.", flush=True)

    to_process = [e for e in entries if str(e.get("row_id", "")) not in already_done]
    if not to_process:
        print("All entries already translated. Nothing to do!", flush=True)
        return

    print(f"Translator     : Google Translate (Batched API)")
    print(f"Total entries  : {len(entries)}")
    print(f"Already done   : {len(already_done)}")
    print(f"Remaining      : {len(to_process)}")
    print(f"Batch settings : Max {MAX_BATCH_LEN} chars/request, {DELAY_SECONDS}s delay")
    print("-" * 60, flush=True)

    # 1) Flattener: extract all non-empty (row_id, key, text) into a flat queue
    work_queue = []
    for entry in to_process:
        row_id = str(entry.get("row_id", "(no id)"))
        eng_cols = entry.get("english_columns") or {}
        
        for key, value in eng_cols.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                # This field is empty in English, so it stays empty in Dutch.
                # Initialize result entries only on demand to prevent empty checkpoint shells.
                if row_id not in result_map:
                    result_map[row_id] = {}
                result_map[row_id][key] = value
            else:
                work_queue.append((row_id, key, str(value).strip()))
    
    total_fields = len(work_queue)
    print(f"Found {total_fields} distinct text fields to translate.", flush=True)

    if total_fields == 0:
        return

    # 2) Batch builder
    batches = []      # list of strings (joined by SEP)
    batch_meta = []   # list of list of (row_id, key) — maps back exactly to elements in the batch
    
    current_batch_text = ""
    current_batch_meta = []

    for row_id, key, text in work_queue:
        # If adding this field + SEP exceeds max chars
        if current_batch_text and len(current_batch_text) + len(SEP) + len(text) > MAX_BATCH_LEN:
            batches.append(current_batch_text)
            batch_meta.append(current_batch_meta)
            current_batch_text = ""
            current_batch_meta = []
        
        # Add to current batch
        if current_batch_text:
            current_batch_text += SEP
        current_batch_text += text
        current_batch_meta.append((row_id, key))

    # Add final partial batch
    if current_batch_text:
        batches.append(current_batch_text)
        batch_meta.append(current_batch_meta)

    print(f"Packed into {len(batches)} batches (API requests). Huge speedup!", flush=True)
    t_start_all = time.time()

    # 3) Execute translations
    for b_idx, (batch_text, meta_list) in enumerate(zip(batches, batch_meta), start=1):
        print(f"[{b_idx:>4}/{len(batches)}] Translating batch ({len(meta_list)} fields, {len(batch_text)} chars) ...", flush=True)
        t_batch_start = time.time()
        
        translated_batch = _translate_batch_with_retry(batch_text, b_idx)
        
        # 4) Split by separator
        # Google Translate sometimes messes up spacing around the separator. 
        # Attempt to split cleanly. We use SEP_CLEAN as it might strip the newlines.
        
        # normalize separator variance the API may introduce
        normalized_resp = translated_batch.replace("=== || ===", "===||===").replace("=== | | ===", "===||===").replace("===|| ===", "===||===")
        
        split_parts = normalized_resp.split("===||===")
        split_parts = [p.strip() for p in split_parts]

        if len(split_parts) != len(meta_list):
            # Critical split failure: Google swallowed a separator. 
            # Fallback: keep original text for this batch to avoid data corruption.
            print(f"        ⚠ [Batch {b_idx}] Split mismatch! Expected {len(meta_list)} parts, got {len(split_parts)}. Keeping English for safety.", flush=True)
            for (row_id, key), orig_text in zip(meta_list, batch_text.split(SEP)):
                result_map[row_id][key] = orig_text
        else:
            # Map translated chunks back to objects
            for (row_id, key), t_text in zip(meta_list, split_parts):
                if row_id not in result_map:
                    result_map[row_id] = {}
                result_map[row_id][key] = t_text
        
        elapsed = time.time() - t_batch_start
        print(f"        ✓ done in {elapsed:.1f}s", flush=True)

        # 5) Periodically save checkpoint (convert map to sorted list)
        ordered_result = [
            {"row_id": int(r) if str(r).isdigit() else r, "dutch_columns": cols}
            for r, cols in result_map.items()
        ]
        # Sort by row_id to keep order deterministic
        ordered_result.sort(key=lambda x: str(x["row_id"]))
        _save_result(out_file, ordered_result)

        if b_idx < len(batches):
            time.sleep(DELAY_SECONDS)

    total_elapsed = time.time() - t_start_all
    print("-" * 60, flush=True)
    print(f"Finished {total_fields} fields in {total_elapsed:.1f}s — average {(total_elapsed/total_fields):.2f}s per field.")
    print(f"Saved into {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English entries to Dutch (NL).")
    parser.add_argument(
        "--limit", "-n",
        type=int, default=None,
        help="Only translate first N entries (e.g. 10 for testing)",
    )
    args = parser.parse_args()
    main(limit=args.limit)
