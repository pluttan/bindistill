"""Turning text into a flat token file, and reading windows out of it.

The corpus is one array on disk. Tokenising a hundred million tokens takes long
enough that being interrupted must not mean starting over, so the array is a
memory-mapped file written in place and a small json note records how far it
got. Re-running `fetch` continues from there.

The last `data.eval_tokens` of the array are held out. Nothing in training is
allowed to draw a window that reaches into them, which is the only reason the
numbers at the end mean anything.
"""

from __future__ import annotations

import http.client
import json
import time
from pathlib import Path

import numpy as np

from . import ui

# A dropped connection is not an error here, it is the weather. Reading
# hundreds of gigabytes over https, a read will time out sooner or later, and
# the whole fetch used to die with it.
NETWORK_TROUBLE = (OSError, EOFError, http.client.HTTPException)


# ==============================
# ===  Layout                ===
# ==============================

def corpus_name(config) -> str:
    teacher = str(config.require("model.teacher")).replace("/", "_")
    kind = config.get("data.kind", "fineweb")
    total = int(config.require("data.train_tokens")) + \
        int(config.require("data.eval_tokens"))
    return f"{teacher}-{kind}-{total // 1_000_000}M"


def corpus_files(config) -> tuple[Path, Path]:
    base = config.path("paths.corpus") / corpus_name(config)
    return base.with_suffix(".npy"), base.with_suffix(".json")


def token_dtype(vocab: int):
    return np.uint16 if vocab < 2 ** 16 else np.uint32


# ==============================
# ===  Sources               ===
# ==============================

def _fineweb_documents(config, start_shard: int, skip: int):
    """Yield text documents, shard by shard, without loading a shard at once."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo = config.require("data.repo")
    pattern = config.require("data.shard")
    shard = start_shard
    while True:
        ui.step(f"downloading shard {shard} of {repo}")
        path = hf_hub_download(repo, pattern.format(shard), repo_type="dataset")
        handle = pq.ParquetFile(path)
        seen = 0
        for batch in handle.iter_batches(batch_size=512, columns=["text"]):
            texts = batch.column("text").to_pylist()
            if seen + len(texts) <= skip:
                seen += len(texts)
                continue
            if seen < skip:
                texts = texts[skip - seen:]
            seen += len(texts)
            yield shard, seen, texts
        shard += 1
        skip = 0



def _dolma_urls(config) -> list[str]:
    """The file list, fetched once and kept next to the corpus.

    Dolma is served as plain https links to gzipped json lines, grouped by
    source: books, filtered web, news, scientific papers, code. The subset name
    is in the path, so a mixture of domains is a filter over this list rather
    than a separate download mechanism.
    """
    import urllib.request

    version = str(config.get("data.dolma_version", "v1_7"))
    cache = config.path("paths.corpus") / f"dolma-{version}-urls.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        listing = (f"https://huggingface.co/datasets/allenai/dolma/raw/main/"
                   f"urls/{version}.txt")
        ui.step(f"fetching the {version} file list")
        attempts = max(1, int(config.get("data.retries", 5)))
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(listing, timeout=60) as response:
                    cache.write_bytes(response.read())
                break
            except NETWORK_TROUBLE as problem:
                if attempt == attempts:
                    raise
                # The list is one small file, but the fetch that needs it may
                # be days long: failing here would waste the whole attempt.
                delay = min(60.0, 5.0 * 2 ** (attempt - 1))
                ui.warn(f"{problem} — retrying the file list in {delay:.0f}s")
                time.sleep(delay)

    urls = [line.strip() for line in cache.read_text().splitlines()
            if line.strip().startswith("http")]

    wanted = config.get("data.subsets") or []
    if wanted:
        # Interleave by source, so a mixture stays a mixture instead of reading
        # one domain to exhaustion first. Sources hold wildly different numbers
        # of files — books a handful, web hundreds — so zip_longest is used:
        # plain zip would cut every source down to the shortest one.
        from itertools import zip_longest

        grouped = {name: [u for u in urls if f"/{name}/" in u] for name in wanted}
        missing = [n for n, v in grouped.items() if not v]
        if missing:
            names = sorted({u.split("/")[-2] for u in urls})
            raise ValueError(f"unknown dolma subsets {missing}; have: {names}")
        urls = [u for row in zip_longest(*(grouped[n] for n in wanted))
                for u in row if u is not None]
    return urls


def _shard_lines(url: str, timeout: float):
    """The decompressed lines of one remote file."""
    import gzip
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "bindistill"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with gzip.GzipFile(fileobj=response) as stream:
            yield from stream


def _shard_batches(open_lines, url: str, skip: int, attempts: int,
                   pause, batch_size: int = 256):
    """Batches of texts from one file, re-opening it when the read breaks.

    `skip` counts lines already handed out, so a retry resumes where the
    previous attempt stopped instead of repeating documents. Re-reading the
    skipped part costs bandwidth, which is the price of not restarting the
    whole fetch. When the attempts run out the file is given up on: one
    unreachable shard out of hundreds should not end the run.
    """
    import json as json_module

    done = skip
    for attempt in range(1, attempts + 1):
        batch, seen = [], 0
        try:
            for line in open_lines(url):
                seen += 1
                if seen <= done:
                    continue
                try:
                    text = json_module.loads(line).get("text")
                except (ValueError, UnicodeDecodeError):
                    continue
                if not text:
                    continue
                batch.append(text)
                if len(batch) >= batch_size:
                    done = seen
                    yield seen, batch
                    batch = []
            if batch:
                done = seen
                yield seen, batch
            ui.detail(f"finished {url.split('/')[-1]} at line {seen}")
            return
        except NETWORK_TROUBLE as problem:
            ui.detail(f"broke at line {seen} of {url.split('/')[-1]}: "
                      f"{type(problem).__name__}: {problem}")
            if batch:
                done = seen
                yield seen, batch
            if attempt == attempts:
                ui.warn(f"giving up on {url.split('/')[-1]} after {attempts} "
                        f"tries ({problem}); moving to the next file")
                return
            delay = min(60.0, 5.0 * 2 ** (attempt - 1))
            ui.warn(f"{problem} — retrying in {delay:.0f}s "
                    f"({attempt} of {attempts})")
            pause(delay)


def _dolma_documents(config, start_shard: int, skip: int):
    """Stream the gzipped json lines; nothing is kept on disk."""
    urls = _dolma_urls(config)
    if not urls:
        raise ValueError("the dolma file list came back empty")

    timeout = float(config.get("data.timeout", 120))
    attempts = max(1, int(config.get("data.retries", 5)))

    index = start_shard
    while index < len(urls):
        url = urls[index]
        ui.step(f"streaming {url.split('/')[-2]}/{url.split('/')[-1]}")
        ui.detail(f"file {index + 1} of {len(urls)}, resuming at line {skip}")
        ui.detail(f"url {url}")
        for seen, batch in _shard_batches(
                lambda u: _shard_lines(u, timeout), url, skip, attempts,
                time.sleep):
            yield index, seen, batch
        index += 1
        skip = 0

def _text_documents(config, start_shard: int, skip: int):
    """A local file, split on blank lines, for training on your own material."""
    path = Path(str(config.require("data.text_file"))).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"data.text_file does not exist: {path}")
    chunks = path.read_text(encoding="utf-8", errors="replace").split("\n\n")
    chunks = [c.strip() for c in chunks if c.strip()]
    for index in range(skip, len(chunks), 512):
        yield 0, index + 512, chunks[index:index + 512]


# ==============================
# ===  Building              ===
# ==============================

def build_corpus(config, tokenizer) -> Path:
    """Tokenise until the target is reached; resume if a partial file exists."""
    array_path, note_path = corpus_files(config)
    array_path.parent.mkdir(parents=True, exist_ok=True)

    target = int(config.require("data.train_tokens")) + \
        int(config.require("data.eval_tokens"))
    dtype = token_dtype(len(tokenizer))

    note = {"written": 0, "shard": 0, "row": 0, "target": target,
            "dtype": np.dtype(dtype).name,
            "teacher": config.require("model.teacher")}
    if note_path.exists() and array_path.exists():
        stored = json.loads(note_path.read_text())
        if stored.get("target") == target and stored.get("dtype") == note["dtype"]:
            note = stored
            if 0 < note["written"] < target:
                ui.step(f"resuming at {note['written'] / 1e6:.1f}M tokens")
        else:
            ui.warn("existing corpus was built with other settings, rebuilding")

    if note["written"] >= target:
        ui.good(f"corpus ready: {target / 1e6:.1f}M tokens at {array_path.name}")
        return array_path

    mode = "r+" if array_path.exists() and note["written"] else "w+"
    tokens = np.lib.format.open_memmap(
        array_path, mode=mode, dtype=dtype, shape=(target,))

    source = {"fineweb": _fineweb_documents, "dolma": _dolma_documents,
              "text": _text_documents}
    kind = config.get("data.kind", "fineweb")
    if kind not in source:
        raise ValueError(f"unknown data.kind '{kind}'; have: {sorted(source)}")
    reader = source[kind]
    # `or` is wrong here: token id 0 is a perfectly ordinary end-of-text id.
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos = getattr(tokenizer, "pad_token_id", None)
    if eos is None:
        eos = 0

    progress = ui.Progress("tokenising", target)
    written = int(note["written"])
    last_note, last_written = time.time(), written
    ui.detail(f"corpus file {array_path} target {target} tokens "
              f"({target * np.dtype(dtype).itemsize / 2 ** 30:.2f} GB), "
              f"starting at {written}")
    try:
        for shard, row, texts in reader(config, int(note["shard"]), int(note["row"])):
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                piece = np.array(ids + [eos], dtype=dtype)
                room = min(len(piece), target - written)
                if room <= 0:
                    break
                tokens[written:written + room] = piece[:room]
                written += room
            note.update({"written": written, "shard": shard, "row": row})
            note_path.write_text(json.dumps(note, indent=2))
            progress.update(written)
            now = time.time()
            if now - last_note >= 60:
                rate = (written - last_written) / max(1e-6, now - last_note)
                ui.detail(f"{written / 1e6:.1f}M tokens, shard {shard}, "
                          f"line {row}, {rate / 1e3:.1f}k tokens/s")
                last_note, last_written = now, written
            if written >= target:
                break
    finally:
        tokens.flush()
        note.update({"written": written})
        note_path.write_text(json.dumps(note, indent=2))

    progress.done(f"{written / 1e6:.1f}M tokens")
    if written < target:
        ui.warn(f"stopped short of {target / 1e6:.1f}M; run fetch again to continue")
    return array_path


def open_corpus(config) -> np.memmap:
    array_path, note_path = corpus_files(config)
    if not array_path.exists():
        raise FileNotFoundError(
            f"no corpus at {array_path} — run `make fetch` on a machine with "
            f"network access first")
    tokens = np.load(array_path, mmap_mode="r")
    if note_path.exists():
        note = json.loads(note_path.read_text())
        written = int(note.get("written", 0))
        if 0 < written < note.get("target", 0):
            ui.warn(f"corpus is only {written / 1e6:.1f}M of "
                    f"{note['target'] / 1e6:.1f}M tokens; run `make fetch` "
                    f"again to finish it")
            # Hand back only what was actually written. The file is created at
            # its full length up front, so the tail is zeros: training would
            # sample them as text, and the held-out set - taken from the very
            # end of the corpus - would be nothing else, making perplexity and
            # every decision based on it meaningless.
            tokens = tokens[:written]
    return tokens


# ==============================
# ===  Reading windows       ===
# ==============================

class TokenWindows:
    """Random fixed-length windows from the training part of the corpus.

    Written as a plain iterator rather than an IterableDataset so that a worker
    process is optional: on a single card the copy is cheap and the extra
    processes only cost memory.
    """

    def __init__(self, config, rank: int = 0, world: int = 1):
        self.tokens = open_corpus(config)
        self.seq = int(config.require("train.seq"))
        held = int(config.require("data.eval_tokens"))
        self.limit = len(self.tokens) - held - self.seq - 1
        if self.limit <= 0:
            raise ValueError("corpus is too small for this sequence length")
        seed = int(config.get("run.seed", 0)) + rank
        self.rng = np.random.default_rng(seed)
        self.world = world

    def batch(self, size: int):
        import torch

        starts = self.rng.integers(0, self.limit, size)
        rows = np.stack([np.asarray(self.tokens[s:s + self.seq + 1],
                                    dtype=np.int64) for s in starts])
        chunk = torch.from_numpy(rows)
        inputs, targets = chunk[:, :-1], chunk[:, 1:]
        if torch.cuda.is_available():
            # Pinned pages let the copy to the card overlap with computation.
            inputs, targets = inputs.pin_memory(), targets.pin_memory()
        return inputs, targets

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state(self, state: dict) -> None:
        if state and "rng" in state:
            self.rng.bit_generator.state = state["rng"]


def held_out_windows(config, count: int | None = None):
    """The evaluation set: fixed, contiguous, never seen by training."""
    import torch

    tokens = open_corpus(config)
    seq = int(config.require("train.seq"))
    held = int(config.require("data.eval_tokens"))
    count = int(count or config.require("eval.windows"))

    start = len(tokens) - held
    available = (held - 1) // seq
    if available < 1:
        raise ValueError("data.eval_tokens is smaller than one window")
    count = min(count, available)
    rows = np.stack([
        np.asarray(tokens[start + i * seq: start + i * seq + seq + 1],
                   dtype=np.int64)
        for i in range(count)])
    chunk = torch.from_numpy(rows)
    return chunk[:, :-1], chunk[:, 1:]
