#!/usr/bin/env python3
"""Render the matplotlib figures stored in a dataflow plot file.

Handles both formats the dataflow uses: the per-channel ``.pkl`` and the
merged shelve database (``.dir``/``.dat``, passed as either name).  Every
figure found anywhere in the nested dict is written out as an image so it
can be opened in the VS Code editor.

    python pklfig.py <file> --list            # just show what is inside
    python pklfig.py <file> [-o OUTDIR] [--ext png|pdf|svg]
"""

from __future__ import annotations

import argparse
import dbm.dumb
import pickle as pkl
import shelve
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.figure import Figure  # noqa: E402


def walk(obj, prefix=()):
    """Yield (key path, figure) for every Figure nested anywhere in obj."""
    if isinstance(obj, Figure):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, val in obj.items():
            yield from walk(val, (*prefix, str(key)))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            yield from walk(val, (*prefix, str(i)))


def load(path):
    if path.suffix in (".dir", ".dat", ".bak"):
        stem = path.with_suffix("")
        with (
            dbm.dumb.open(str(stem), "r") as db_object,
            shelve.Shelf(db_object) as shelf,
        ):
            return dict(shelf)
    with path.open("rb") as f:
        return pkl.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="plot .pkl or shelve .dir")
    parser.add_argument("-o", "--outdir", type=Path, default=None)
    parser.add_argument("--ext", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--list", action="store_true", help="only print the keys")
    args = parser.parse_args()

    figures = list(walk(load(args.file)))
    if not figures:
        print(f"no matplotlib figure found in {args.file}")
        return

    if args.list:
        for path, fig in figures:
            print("/".join(path) or "<root>", f"  {tuple(fig.get_size_inches())} in")
        return

    outdir = args.outdir or args.file.parent / (args.file.stem + "_figs")
    outdir.mkdir(parents=True, exist_ok=True)
    for path, fig in figures:
        out = outdir / f"{'_'.join(path) or args.file.stem}.{args.ext}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(out)


if __name__ == "__main__":
    main()
