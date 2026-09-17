#!/usr/bin/env python3.12
"""Entry point. Every command takes --preset, --config and --set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from distill import config as config_module  # noqa: E402
from distill import ui  # noqa: E402


def add_common(parser: argparse.ArgumentParser, subcommand: bool) -> None:
    """The shared flags, accepted both before and after the command name.

    A subcommand copy uses SUPPRESS as its default so that leaving the flag off
    does not overwrite a value already given ahead of the command.
    """
    blank = argparse.SUPPRESS if subcommand else None
    parser.add_argument("--config", default=blank, help="path to config.toml")
    parser.add_argument("--preset", default=blank,
                        help="smoke | tiny | small | full")
    parser.add_argument("--set", dest="overrides", action="append",
                        default=(argparse.SUPPRESS if subcommand else []),
                        metavar="KEY=VALUE", help="override one setting")
    parser.add_argument("--verbose", action="store_true",
                        default=(argparse.SUPPRESS if subcommand else False),
                        help="print everything that goes to the log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distill",
        description="Train a one-bit model against the full-precision original")
    add_common(parser, subcommand=False)

    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
            ("selftest", "check the machinery, no download needed"),
            ("status", "what is present and what is missing"),
            ("fetch", "download the teacher and build the corpus")):
        add_common(sub.add_parser(name, help=help_text), subcommand=True)

    train = sub.add_parser("train", help="run the distillation")
    train.add_argument("--fresh", action="store_true",
                       help="ignore any checkpoint and start over")
    add_common(train, subcommand=True)

    evaluate = sub.add_parser("eval", help="teacher vs. baseline vs. student")
    evaluate.add_argument("--checkpoint", default=None)
    add_common(evaluate, subcommand=True)

    profile = sub.add_parser(
        "profile", help="what the training left in the weights")
    profile.add_argument("--checkpoint", default=None)
    add_common(profile, subcommand=True)

    reference = sub.add_parser(
        "reference", help="perplexity on WikiText-2 and C4, as published work reports it")
    reference.add_argument("--checkpoint", default=None)
    reference.add_argument("--window", type=int, default=2048)
    reference.add_argument("--limit", type=int, default=None,
                           help="windows per corpus; a few is a dry run")
    reference.add_argument("--only", default=None)
    add_common(reference, subcommand=True)

    footprint = sub.add_parser(
        "footprint", help="what the compression is worth, end to end")
    footprint.add_argument("--checkpoint", default=None)
    footprint.add_argument("--no-speed", action="store_true",
                           help="size only, skip the generation timing")
    add_common(footprint, subcommand=True)

    bench = sub.add_parser(
        "bench", help="multiple-choice tasks, every model side by side")
    bench.add_argument("--checkpoint", default=None)
    bench.add_argument("--hours", type=float, default=18.0,
                       help="stop starting new models after this long")
    bench.add_argument("--limit", type=float, default=None,
                       help="examples per task; a few is a dry run")
    bench.add_argument("--core-only", action="store_true",
                       help="leave out the expensive wide task")
    bench.add_argument("--only", default=None,
                       help="comma-separated subset of the models")
    add_common(bench, subcommand=True)

    export = sub.add_parser("export", help="write the trained model out")
    export.add_argument("--checkpoint", default=None)
    export.add_argument("--out", default=None)
    add_common(export, subcommand=True)
    return parser


def resolve_checkpoint(config, given: str | None):
    from distill import checkpoint

    if given:
        path = Path(given)
        if not path.exists():
            ui.fail(f"no checkpoint at {path}")
            return None
        return path
    found = checkpoint.latest(config.run_dir())
    if found is None:
        ui.warn(f"no checkpoint in {config.run_dir()}")
    return found


# ==============================
# ===  Commands              ===
# ==============================

def command_status(config) -> int:
    from distill import checkpoint, data, models

    try:
        import torch
    except ImportError:
        torch = None

    ui.head("Status")
    ui.field("config", config.source)
    ui.field("preset", config.get("preset", "-"))
    ui.field("teacher", config.require("model.teacher"))

    model_dir = models.local_dir(config)
    have_model = (model_dir / "config.json").exists()
    ui.field("model files", "present" if have_model else "missing",
             "green" if have_model else "red")
    ui.say(f"      {model_dir}", "overlay")

    array_path, note_path = data.corpus_files(config)
    if array_path.exists():
        import json

        note = json.loads(note_path.read_text()) if note_path.exists() else {}
        written = note.get("written", 0)
        target = note.get("target", 0)
        done = written >= target and target > 0
        ui.field("corpus", f"{written / 1e6:.1f}M of {target / 1e6:.1f}M tokens",
                 "green" if done else "yellow")
    else:
        ui.field("corpus", "missing", "red")
    ui.say(f"      {array_path}", "overlay")

    found = checkpoint.latest(config.run_dir())
    if found is not None:
        meta = checkpoint.read_meta(found)
        ui.field("checkpoint", f"{found.name} (step {meta['step']}, "
                               f"{meta['tokens'] / 1e6:.0f}M tokens)", "green")
    else:
        ui.field("checkpoint", "none yet", "yellow")

    if torch is None:
        ui.field("torch", "not installed", "red")
        ui.say("      run `make install` first", "overlay")
        return 0

    from distill.config import device_type, resolve_device

    device = resolve_device(str(config.get("run.device", "auto")))
    kind = device_type(device)
    ui.field("torch", torch.__version__)
    ui.field("device", device)
    if not torch.cuda.is_available():
        from distill.train import cuda_unavailable_reason

        reason = cuda_unavailable_reason()
        if reason:
            ui.field("no gpu", reason, "red")
    if kind == "cuda" and torch.cuda.is_available():
        chosen = int(device.split(":")[1]) if ":" in device \
            else torch.cuda.current_device()
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            total = torch.cuda.get_device_properties(index).total_memory
            free, _ = torch.cuda.mem_get_info(index)
            mark = " <- selected" if index == chosen else ""
            ui.say(f"      gpu {index}: {name}, {free / 2 ** 30:.1f} of "
                   f"{total / 2 ** 30:.1f} GB free{mark}", "overlay")
    return 0


def command_fetch(config) -> int:
    from distill import data, models

    ui.head("Fetch")
    models.fetch_model(config)
    tokenizer = models.load_tokenizer(config)
    data.build_corpus(config, tokenizer)
    ui.good("ready — this folder can now be used without a network")
    return 0


def command_train(config, fresh: bool) -> int:
    from distill import train

    train.run(config, resume=not fresh)
    return 0


def command_eval(config, given: str | None) -> int:
    from distill import evaluate

    evaluate.run(config, resolve_checkpoint(config, given))
    return 0


def command_profile(config, given: str | None) -> int:
    from distill import profile

    source = resolve_checkpoint(config, given)
    if source is None:
        return 1
    profile.run(config, source)
    return 0


def command_reference(config, args) -> int:
    from distill import reference

    only = [p.strip() for p in args.only.split(",")] if args.only else None
    reference.run(config, resolve_checkpoint(config, args.checkpoint),
                  window=args.window, limit=args.limit, only=only)
    return 0


def command_footprint(config, args) -> int:
    from distill import footprint

    footprint.run(config, resolve_checkpoint(config, args.checkpoint),
                  speed=not args.no_speed)
    return 0


def command_bench(config, args) -> int:
    from distill import bench

    # A missing checkpoint is not fatal here: the other rows still measure.
    found = resolve_checkpoint(config, args.checkpoint)
    only = [p.strip() for p in args.only.split(",")] if args.only else None
    bench.run(config, found, hours=args.hours, limit=args.limit,
              wide=not args.core_only, only=only)
    return 0


def command_export(config, given: str | None, out: str | None) -> int:
    from distill import export

    source = resolve_checkpoint(config, given)
    if source is None:
        return 1
    export.run(config, source, Path(out) if out else None)
    return 0


def describe_environment(config) -> None:
    """The facts a report needs and nobody remembers to include."""
    import os
    import platform

    ui.detail(f"python {platform.python_version()} on {platform.platform()}")
    ui.detail(f"argv {' '.join(sys.argv[1:])}")
    for name in ("huggingface_hub", "transformers", "torch", "numpy"):
        try:
            module = __import__(name)
            ui.detail(f"{name} {getattr(module, '__version__', '?')}")
        except Exception as problem:  # noqa: BLE001 - absence is the answer
            ui.detail(f"{name} missing ({problem})")
    from distill import data

    token = data.hub_token()
    ui.detail(f"hub token {'present' if token else 'MISSING - anonymous '
                                                  'downloads are throttled'}")
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_DOWNLOAD_TIMEOUT",
                 "HF_HUB_OFFLINE", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        if os.environ.get(name):
            ui.detail(f"{name}={os.environ[name]}")
    for key in ("paths.models", "paths.corpus", "paths.cache", "paths.runs"):
        try:
            place = config.path(key)
            ui.detail(f"{key} {place} "
                      f"({'writable' if os.access(place.parent, os.W_OK) else 'NOT writable'})")
        except Exception as problem:  # noqa: BLE001
            ui.detail(f"{key} unavailable ({problem})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "selftest":
        from distill import selftest

        return 1 if selftest.run() else 0

    try:
        config = config_module.load(args.config, args.preset, args.overrides)
    except (FileNotFoundError, KeyError, ValueError) as problem:
        ui.fail(str(problem))
        return 1

    # Before anything imports huggingface_hub: it reads the cache location
    # once, at import time.
    from distill import models

    # A transcript of every run, so a failure can be read after the fact
    # instead of reproduced. --verbose puts the same detail on screen.
    log = ui.log_to(config.path("paths.runs") / f"{args.command}.log",
                    getattr(args, "verbose", False))

    try:
        models.use_local_cache(config)
    except PermissionError as problem:
        ui.fail(str(problem))
        return 1

    if log is not None:
        ui.detail(f"log {log}")
        describe_environment(config)

    if args.command == "status":
        return command_status(config)
    if args.command == "fetch":
        return command_fetch(config)
    if args.command == "train":
        return command_train(config, args.fresh)
    if args.command == "eval":
        return command_eval(config, args.checkpoint)
    if args.command == "profile":
        return command_profile(config, args.checkpoint)
    if args.command == "reference":
        return command_reference(config, args)
    if args.command == "footprint":
        return command_footprint(config, args)
    if args.command == "bench":
        return command_bench(config, args)
    if args.command == "export":
        return command_export(config, args.checkpoint, args.out)
    return 1


if __name__ == "__main__":
    sys.exit(main())
