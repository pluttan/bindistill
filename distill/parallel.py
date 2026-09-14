# Filling the corpus with several shards at once.
#
# One shard at a time leaves a large machine idle: reading and decompressing
# is one connection, and the tokeniser only parallelises within a batch, so a
# 190-core box ran at fourteen cores. Both halves release the GIL - urllib in
# the socket, the Rust tokeniser in its batch - so plain threads do the job.
#
# Writing stays in one place, in the order results arrive. Documents from
# different shards end up interleaved, which costs nothing: training draws
# random windows anyway, and a mixture stays mixed.
#
# Signed: pluttan

from __future__ import annotations

import json
import queue
import threading
import time

import numpy as np

from . import ui

STOP = object()


def restore_cursors(note: dict) -> tuple[set[int], dict[int, int]]:
    """Where each shard was left, from a note written by either filler.

    A note from the sequential version carries a single shard and row; every
    shard before it is finished. Reading it this way is what lets a download
    that is already hundreds of gigabytes in continue rather than restart.
    """
    if "cursors" in note or "finished" in note:
        finished = set(int(i) for i in note.get("finished", []))
        cursors = {int(k): int(v) for k, v in note.get("cursors", {}).items()}
        return finished, cursors
    shard, row = int(note.get("shard", 0)), int(note.get("row", 0))
    return set(range(shard)), ({shard: row} if row else {})


def _worker(jobs, results, batches, stopping, config, tokenizer, eos, dtype):
    """Read a shard, tokenise it, hand the pieces over."""
    from . import data

    while not stopping.is_set():
        try:
            index, url, skip = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            ui.detail(f"shard {index} {url.split('/')[-1]} from line {skip}")
            timeout = float(config.get("data.timeout", 120))
            attempts = max(1, int(config.get("data.retries", 5)))
            import time as time_module

            for row, texts in data._shard_batches(
                    lambda u: data._shard_lines(u, timeout), url, skip,
                    attempts, time_module.sleep, batches):
                if stopping.is_set():
                    return
                encoded = tokenizer(texts, add_special_tokens=False)
                flat = []
                for ids in encoded["input_ids"]:
                    flat.extend(ids)
                    flat.append(eos)
                results.put((index, row, np.array(flat, dtype=dtype)))
            ui.detail(f"shard {index} done")
            results.put((index, None, None))
        except Exception as problem:  # noqa: BLE001 - reported, not raised
            ui.detail(f"shard {index} stopped: "
                      f"{type(problem).__name__}: {problem}")
            results.put((index, None, None))
        finally:
            jobs.task_done()


def fill(tokens, note: dict, note_path, urls: list[str], config, tokenizer,
         eos: int, target: int, progress) -> int:
    """Write tokens until `target`, reading `workers` shards at a time.

    Returns how many tokens the file holds when it stops - on the target, on
    running out of shards, or on an interruption, all of which leave the note
    consistent with what is actually on disk.
    """
    from . import data

    workers = max(1, int(config.get("data.fetch_workers", 8)))
    batches = max(16, int(config.get("data.batch_documents", 256)))
    written = int(note["written"])

    finished, cursors = restore_cursors(note)
    pending = [i for i in range(len(urls)) if i not in finished]
    if not pending:
        return written

    ui.detail(f"{workers} workers, {len(finished)} shards done, "
              f"{len(pending)} to go")

    jobs: queue.Queue = queue.Queue()
    for index in pending:
        jobs.put((index, urls[index], cursors.get(index, 0)))
    # Bounded, or the readers race ahead of the writer and the machine fills
    # with tokenised text nobody has stored yet.
    results: queue.Queue = queue.Queue(maxsize=workers * 4)
    stopping = threading.Event()

    dtype = tokens.dtype
    threads = [threading.Thread(target=_worker, daemon=True,
                                args=(jobs, results, batches, stopping, config,
                                      tokenizer, eos, dtype))
               for _ in range(workers)]
    for thread in threads:
        thread.start()

    live = len(threads)
    marked, marked_at = time.time(), written
    try:
        while written < target and live > 0:
            try:
                index, row, piece = results.get(timeout=1.0)
            except queue.Empty:
                if not any(t.is_alive() for t in threads):
                    break
                continue

            if piece is None:
                finished.add(index)
                cursors.pop(index, None)
                if jobs.empty():
                    live = sum(1 for t in threads if t.is_alive())
                continue

            room = min(len(piece), target - written)
            tokens[written:written + room] = piece[:room]
            written += room
            cursors[index] = row

            note.update({"written": written, "finished": sorted(finished),
                         "cursors": {str(k): v for k, v in cursors.items()},
                         # Kept so an older build can still read this note.
                         "shard": min(pending, default=0), "row": 0})
            data.write_note(note_path, note)
            progress.update(written)

            now = time.time()
            if now - marked >= 60:
                rate = (written - marked_at) / max(1e-6, now - marked)
                ui.detail(f"{written / 1e6:.1f}M tokens, {len(finished)} "
                          f"shards done, {rate / 1e3:.1f}k tokens/s")
                marked, marked_at = now, written
    finally:
        stopping.set()
        # Drain what the readers already produced, so they are not blocked on
        # a full queue while we wait for them to notice.
        while True:
            try:
                results.get_nowait()
            except queue.Empty:
                break
        for thread in threads:
            thread.join(timeout=2.0)
        tokens.flush()
        note.update({"written": written})
        data.write_note(note_path, note)

    return written
