# Filling the corpus with several shards at once.
#
# Processes, not threads. Two measurements decided it. The tokeniser's own
# pool scales badly - on a 256-thread machine 4 cores gave 2.05M tokens/s and
# 64 gave 5.99M, so sixteen times the cores bought less than three times the
# speed - and everything around it (splitting lines, json, building arrays)
# holds the GIL, so threads queued behind one another and the machine sat at
# one core busy while a 27 MB/s link idled.
#
# Sixteen processes of four cores get about 30M tokens/s out of the same box,
# which is past what the network delivers - the right place to stop.
#
# Writing stays in the parent, in the order pieces arrive. Documents from
# different shards end up interleaved, which costs nothing: training draws
# random windows anyway, and a mixture stays mixed.
#
# Signed: pluttan

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import queue
import time
from pathlib import Path

import numpy as np

from . import ui

_TOKENIZER = None
_RESULTS = None
_EOS = 0


def restore_cursors(note: dict) -> tuple[set[int], dict[int, int]]:
    """Where each shard was left, from a note written by any filler.

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


def start_method() -> str:
    """How to start workers on this platform.

    fork is cheap and keeps start-up instant, but on macOS it is unsafe once
    the process has loaded libraries that spawn threads of their own - the
    children die on start and the system puts up a crash dialog for each one.
    """
    return "fork" if sys.platform.startswith("linux") else "spawn"


def worker_cores(budget: int, workers: int) -> int:
    """Cores for one worker out of the whole allowance.

    Small pools are the point: four cores per process beat sixty four in one,
    and the ceiling still holds for the machine as a whole.
    """
    return max(1, int(budget) // max(1, int(workers)))


def _prepare(config_data: dict, source: str, cores: int, eos: int,
             results) -> None:
    """Runs once per worker: its own tokeniser, with its own small pool."""
    global _TOKENIZER, _RESULTS, _EOS

    # Set before the tokeniser is imported - the pool reads this when it is
    # built, and a worker that grabs every core defeats the whole point.
    os.environ["RAYON_NUM_THREADS"] = str(max(1, cores))
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    from . import config as config_module
    from . import models

    _TOKENIZER = models.load_tokenizer(
        config_module.Config(config_data, Path(source)))
    _RESULTS = results
    _EOS = eos


def _read_shard(task) -> None:
    """One shard, read and tokenised, handed over a piece at a time.

    A dclm shard is the better part of a gigabyte of tokens, so building a
    whole one before returning would put as many of those in memory as there
    are workers. Pieces go out as they are made, and the queue's own bound is
    what keeps the readers from running away from the writer.
    """
    index, url, skip, timeout, attempts, batch, dtype_name = task
    from . import data

    try:
        for row, texts in data._shard_batches(
                lambda u: data._shard_lines(u, timeout), url, skip, attempts,
                time.sleep, batch):
            encoded = _TOKENIZER(texts, add_special_tokens=False)["input_ids"]
            flat = []
            for ids in encoded:
                flat.extend(ids)
                flat.append(_EOS)
            _RESULTS.put((index, row, np.array(flat, dtype=dtype_name)))
    except Exception as problem:  # noqa: BLE001 - reported, not raised
        _RESULTS.put((index, None, str(problem)))
        return
    _RESULTS.put((index, None, None))


def fill(tokens, note: dict, note_path, urls: list[str], config, tokenizer,
         eos: int, target: int, progress) -> int:
    """Write tokens until `target`, reading several shards at a time.

    Returns how many tokens the file holds when it stops - on the target, on
    running out of shards, or on an interruption, all of which leave the note
    consistent with what is on disk.
    """
    from . import data

    workers = max(1, int(config.get("data.fetch_workers", 8)))
    batch = max(16, int(config.get("data.batch_documents", 256)))
    budget = int(config.get("data.cpu_cores", 0)) or (os.cpu_count() or 8) // 2
    cores = worker_cores(budget, workers)
    timeout = float(config.get("data.timeout", 120))
    attempts = max(1, int(config.get("data.retries", 5)))
    dtype_name = np.dtype(tokens.dtype).name

    written = int(note["written"])
    finished, cursors = restore_cursors(note)
    pending = [i for i in range(len(urls)) if i not in finished]
    if not pending:
        return written

    ui.detail(f"{workers} processes x {cores} cores, {len(finished)} shards "
              f"done, {len(pending)} to go")

    tasks = [(i, urls[i], cursors.get(i, 0), timeout, attempts, batch,
              dtype_name) for i in pending]

    marked, marked_at = time.time(), written
    context = mp.get_context(start_method())
    # Bounded, or the readers race ahead of the writer and fill the machine
    # with tokenised text nobody has stored yet.
    results = context.Queue(maxsize=workers * 4)
    # Each worker builds its own tokeniser after the split, so no thread pool
    # is inherited from the parent either way.
    pool = context.Pool(workers, initializer=_prepare,
                        initargs=(config.data, str(config.source), cores, eos,
                                  results))
    handed = pool.map_async(_read_shard, tasks, chunksize=1)
    live = len(tasks)

    def drain(deadline: float) -> None:
        """Take what the workers have already produced, until it stops
        coming. Their last piece travels through a feeder thread, so the
        queue going quiet for a moment is not proof that a shard is done -
        and leaving early loses the end-of-shard mark, which means reading
        that shard again on the next run.
        """
        nonlocal written, live
        while time.time() < deadline:
            try:
                index, row, piece = results.get(timeout=0.2)
            except queue.Empty:
                continue
            take(index, row, piece)

    def take(index, row, piece) -> None:
        nonlocal written, live
        if row is None:
            live -= 1
            if piece is None:
                finished.add(index)
                cursors.pop(index, None)
            else:
                ui.detail(f"shard {index} stopped: {piece}")
            return
        room = min(len(piece), target - written)
        if room > 0:
            tokens[written:written + room] = piece[:room]
            written += room
            cursors[index] = row

    try:
        while written < target and live > 0:
            try:
                index, row, piece = results.get(timeout=2.0)
            except queue.Empty:
                # A worker's last piece travels through a feeder thread, so
                # an empty queue is not proof that it is done: leaving here
                # too early loses the end-of-shard mark and the shard is read
                # again on the next run.
                if handed.ready() and results.empty():
                    break
                continue

            take(index, row, piece)

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
        # Workers that are still mid-shard have nothing more to tell us, but
        # the ones that just finished do; give their last word a moment to
        # arrive before the pool goes away.
        if live > 0 and written < target:
            drain(time.time() + 3.0)
        pool.terminate()
        pool.join()
        tokens.flush()
        note.update({"written": written})
        data.write_note(note_path, note)

    return written
