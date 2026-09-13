"""Terminal output. Colours are Catppuccin Mocha, dropped when piped to a file."""

from __future__ import annotations

import sys
import time

MOCHA = {
    "text": (205, 214, 244),
    "subtext": (166, 173, 200),
    "overlay": (108, 112, 134),
    "red": (243, 139, 168),
    "peach": (250, 179, 135),
    "yellow": (249, 226, 175),
    "green": (166, 227, 161),
    "teal": (148, 226, 213),
    "blue": (137, 180, 250),
    "mauve": (203, 166, 247),
}

_ENABLED = sys.stdout.isatty()


def paint(text: str, colour: str) -> str:
    if not _ENABLED or colour not in MOCHA:
        return text
    r, g, b = MOCHA[colour]
    return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"


# Everything printed goes through say(), so a transcript can be taken here
# rather than at every call site.
_LOG = None


def log_to(path, verbose: bool = False):
    """Start writing a timestamped copy of the output to a file.

    Returns the path, or None when it cannot be opened - a missing log is
    never a reason to fail the command it was meant to record.
    """
    global _LOG, _VERBOSE

    from pathlib import Path

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _LOG = open(target, "a", encoding="utf-8", buffering=1)
    except OSError:
        _LOG = None
        return None
    _VERBOSE = verbose
    _LOG.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    return target


_VERBOSE = False


def detail(text: str) -> None:
    """A line for the log. It reaches the screen only with --verbose."""
    if _VERBOSE:
        say(f"  {text}", "overlay")
    elif _LOG is not None:
        _LOG.write(f"{time.strftime('%H:%M:%S')}  {text}\n")


def say(text: str = "", colour: str = "text") -> None:
    print(paint(text, colour), flush=True)
    if _LOG is not None:
        _LOG.write(f"{time.strftime('%H:%M:%S')}  {text}\n")


def head(text: str) -> None:
    say()
    say(text, "mauve")
    say("─" * len(text), "overlay")


def step(text: str) -> None:
    say(f"  {text}", "subtext")


def good(text: str) -> None:
    say(f"  {text}", "green")


def warn(text: str) -> None:
    say(f"  {text}", "yellow")


def fail(text: str) -> None:
    say(f"  {text}", "red")


def field(name: str, value: object, colour: str = "teal") -> None:
    say(f"  {name:<22} {paint(str(value), colour)}", "subtext")


def table(rows: list[tuple], headers: tuple) -> None:
    """Small fixed-width table; every measurement in this package prints one."""
    columns = [headers] + [tuple(str(c) for c in row) for row in rows]
    widths = [max(len(row[i]) for row in columns) for i in range(len(headers))]
    line = "  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    say(line, "overlay")
    say("  " + "  ".join("─" * w for w in widths), "overlay")
    for row in columns[1:]:
        say("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


class Progress:
    """Rewrites one line while work runs, without needing a dependency."""

    def __init__(self, label: str, total: float):
        self.label, self.total = label, max(1.0, float(total))
        self.started = time.time()
        self.last = 0.0

    def update(self, done: float, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < 0.5:
            return
        self.last = now
        share = min(1.0, done / self.total)
        elapsed = now - self.started
        eta = elapsed / share - elapsed if share > 0.01 else 0.0
        bar = "█" * int(share * 24) + "·" * (24 - int(share * 24))
        text = (f"  {self.label} {paint(bar, 'blue')} {share * 100:5.1f}%  "
                f"{elapsed / 60:.0f}m elapsed, {eta / 60:.0f}m left")
        if _ENABLED:
            sys.stdout.write("\r\x1b[2K" + text)
            sys.stdout.flush()
        elif force:
            print(text.strip(), flush=True)

    def done(self, note: str = "") -> None:
        if _ENABLED:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
        good(f"{self.label} finished in {(time.time() - self.started) / 60:.1f}m"
             + (f" — {note}" if note else ""))
