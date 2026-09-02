"""Helpers shared by the p08 reproduction scripts.

The pygama cross-talk routines are pure computation: they take arrays and
return dicts, and none of them touches the disk.  This pipeline, though, runs
its four steps in separate SLURM jobs, so every step has to park its result
somewhere the next one can pick it up.  Those file formats live here, next to
the scripts that use them, rather than in pygama.

Three groups of helpers:

* **baseline JSON** -- :func:`write_baseline` puts one
  :func:`~pygama.pargen.xtc.prepare_baseline` result on disk;
  :func:`load_baseline`, :func:`load_baseline_dir` and
  :func:`load_baseline_in_order` join those files back into the ``baseline``
  mapping :func:`~pygama.pargen.xtc.xtalk_column` expects.
* **column lh5** -- :func:`write_xtalk_column` and :func:`read_xtalk_column`
  are inverses, so a column filled in one job can be fitted in another, or
  refitted with different thresholds, without refilling the histograms.
* **comparing two baseline sets** -- :func:`compare_baselines` and
  :func:`format_report` answer a narrower question: do two independently
  produced sets of baseline files describe the *same* measurement?  They will
  never be byte-identical -- they carry different provenance fields and were
  printed by different writers -- so the difference is reported at the level
  that matters: which detectors each side covers, which of them are usable,
  and how far apart the numbers actually are, measured both as a relative
  difference and in float32 ULPs.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path

import lgdo
import lh5
import numpy as np

log = logging.getLogger(__name__)

BASELINE_KEYS = ("detector_id", "positive_baseline", "negative_baseline")
VALUE_KEYS = ("positive_baseline", "negative_baseline")


# ----------------------------------------------------------------------------
# baseline JSON
# ----------------------------------------------------------------------------


def write_baseline(result: dict, out_path: str | Path) -> None:
    """Write one :func:`~pygama.pargen.xtc.prepare_baseline` result as JSON.

    Parent directories are created and an existing file is replaced.  The
    whole result goes in, provenance fields included, but only the three
    :data:`BASELINE_KEYS` are read back by :func:`load_baseline`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info(
        "baseline of channel %s written to %s", result.get("detector_id"), out_path
    )


def load_baseline(
    json_files: Iterable[str | Path],
    strict: bool = True,
) -> dict:
    """Join per-detector baseline JSON files into one ``baseline`` mapping.

    Parameters
    ----------
    json_files
        Paths of the JSON files written by :func:`write_baseline`.  Each must
        be an object holding ``detector_id``, ``positive_baseline`` and
        ``negative_baseline``; any other key it carries is ignored.
    strict
        If True, a file that cannot be parsed, is missing one of those three
        keys, or repeats a ``detector_id`` already seen raises ``ValueError``.
        If False, the file is logged and skipped instead (a repeated
        ``detector_id`` keeps the first file's values).

    Returns
    -------
    dict
        ``{detector_id: {"positive_baseline": float | None,
        "negative_baseline": float | None}}``, ready to pass as the
        ``baseline`` argument of :func:`pygama.pargen.xtc.xtalk_column`.
        Detectors whose measurement failed keep their ``None`` values --
        ``xtalk_column`` treats those as unusable and skips them, so dropping
        them here would only hide them.
    """
    baseline = {}
    seen_in = {}

    for path in json_files:
        path = Path(path)

        try:
            with path.open() as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            msg = f"{path}: cannot be read as JSON: {type(e).__name__}: {e}"
            if strict:
                raise ValueError(msg) from e
            log.warning("%s, skipping", msg)
            continue

        if not isinstance(record, dict):
            msg = f"{path}: holds a {type(record).__name__}, not a JSON object"
            if strict:
                raise ValueError(msg)
            log.warning("%s, skipping", msg)
            continue

        missing = [key for key in BASELINE_KEYS if key not in record]
        if missing:
            msg = f"{path}: missing key(s) {', '.join(missing)}"
            if strict:
                raise ValueError(msg)
            log.warning("%s, skipping", msg)
            continue

        detector_id = record["detector_id"]
        if detector_id in seen_in:
            msg = (
                f"{path}: detector_id {detector_id} already taken from "
                f"{seen_in[detector_id]}"
            )
            if strict:
                raise ValueError(msg)
            log.warning("%s, keeping the first", msg)
            continue

        seen_in[detector_id] = path
        baseline[detector_id] = {
            "positive_baseline": record["positive_baseline"],
            "negative_baseline": record["negative_baseline"],
        }

    return baseline


def baseline_files(directory: str | Path, pattern: str = "*.json") -> list[Path]:
    """Every JSON file in *directory*, sorted by name."""
    return sorted(Path(directory).glob(pattern))


def load_baseline_dir(
    directory: str | Path, pattern: str = "*.json", strict: bool = True
) -> dict:
    """:func:`load_baseline` over every JSON file in *directory*."""
    return load_baseline(baseline_files(directory, pattern), strict=strict)


def load_baseline_in_order(
    baseline_dir: str | Path, chn_id: list, file_list: str | Path | None = None
) -> tuple[dict, list]:
    """Collect the baseline files of *chn_id* into a ``baseline`` mapping.

    Unlike :func:`load_baseline_dir`, which keys the mapping by whatever the
    files happen to hold, this walks *chn_id* in order and takes the file each
    index wrote.  The keys of the result become the response list of a column,
    so building it this way is what makes every column cover the same
    detectors in the same order -- the condition
    :func:`~pygama.pargen.xtc.build_xtalk_matrix` checks later.

    A channel whose file is missing stays in the mapping with ``None`` values,
    which is what ``xtalk_column`` reads as "skip this element" while keeping
    the column the same length as every other one.

    Parameters
    ----------
    baseline_dir
        Directory holding the ``baseline_XXXX.json`` files.
    chn_id
        Detector ids in the order the columns must cover them, i.e. the
        ``chn_id`` list the whole pipeline is indexed by.
    file_list
        Where *chn_id* came from, named in the error raised when a file holds
        a different detector than its index says.  Cosmetic only.

    Returns
    -------
    tuple
        The ``baseline`` mapping, and the list of detectors that had no file.
    """
    baseline_dir = Path(baseline_dir)
    baseline = {}
    missing = []

    for index, detector in enumerate(chn_id):
        baseline_file = baseline_dir / f"baseline_{index:04d}.json"
        try:
            with baseline_file.open() as f:
                entry = json.load(f)
        except FileNotFoundError:
            missing.append(detector)
            baseline[detector] = {
                "positive_baseline": None,
                "negative_baseline": None,
            }
            continue

        if entry["detector_id"] != detector:
            source = f" of {file_list}" if file_list is not None else ""
            msg = (
                f"{baseline_file} holds detector {entry['detector_id']}, but "
                f"index {index}{source} is detector {detector}"
            )
            raise ValueError(msg)

        baseline[detector] = entry

    return baseline, missing


# ----------------------------------------------------------------------------
# cross-talk column lh5
# ----------------------------------------------------------------------------


def write_xtalk_column(result: dict, out_path: str | Path) -> bool:
    """Write a cross-talk column to *out_path* as one lh5 table.

    Both the :func:`~pygama.pargen.xtc.xtalk_column` and the
    :func:`~pygama.pargen.xtc.xtalk_histogram_fitter` results go through here,
    so the two steps produce the same kind of file and a key added to either
    result dict reaches the file without further work.  The mapping is by
    shape: a ``(N,)`` array becomes a column, a ``(N, m)`` array an
    ``ArrayOfEqualSizedArrays`` column, and anything else an attribute on the
    table -- ``dict`` and ``list`` values as JSON strings, which
    ``json_attrs`` then names so :func:`read_xtalk_column` knows to decode
    them.

    An existing *out_path* is replaced rather than appended to, which is what
    lets a refit be written back over the file it read the histograms from
    instead of spreading one column over two files.

    Returns
    -------
    bool
        False when the channel ids are not integral -- lh5 has no array type
        for those, so nothing is written and a warning is logged.  The caller
        still has its result in memory.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trigger_detector_id = result["trigger_id"]
    try:
        trigger_id = int(trigger_detector_id)
        response_ids = np.array(
            [int(i) for i in result["response_ids"]], dtype=np.int64
        )
    except (TypeError, ValueError) as e:
        log.warning(
            "xtalk column of trigger %s not written to %s: channel ids are "
            "not integral, so lh5 cannot store them (%s)",
            trigger_detector_id,
            out_path,
            e,
        )
        return False

    col_dict = {"response_ids": lgdo.Array(response_ids)}
    attrs = {"trigger_id": trigger_id}
    json_attrs = []

    for key, value in result.items():
        if key in ("trigger_id", "response_ids"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 1:
            col_dict[key] = lgdo.Array(value)
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            col_dict[key] = lgdo.ArrayOfEqualSizedArrays(nda=value)
        elif isinstance(value, (str, int, float)):
            attrs[key] = value
        else:
            attrs[key] = json.dumps(value)
            json_attrs.append(key)

    attrs["json_attrs"] = json.dumps(sorted(json_attrs))

    if out_path.exists():
        log.info("replacing the existing %s", out_path)

    lh5.write(
        lgdo.Table(col_dict=col_dict, attrs=attrs),
        name=f"ch{trigger_id}/xtalk_column",
        lh5_file=out_path,
        wo_mode="overwrite_file",
        compression="gzip",
    )
    return True


def read_xtalk_column(
    in_path: str | Path, trigger_detector_id: str | int | None = None
) -> dict:
    """Read a file written by :func:`write_xtalk_column` back into its dict.

    Parameters
    ----------
    in_path
        ``.lh5`` file written by :func:`write_xtalk_column`.
    trigger_detector_id
        Which column to read, when the file holds more than one.  ``None``
        reads the only one there.

    Returns
    -------
    dict
        The result dict the writer was given, with every array back as a numpy
        array -- ready to pass to
        :func:`~pygama.pargen.xtc.xtalk_histogram_fitter`.  Channel ids come
        back as ``int64`` even if the caller that wrote them had strings.
    """
    in_path = Path(in_path)

    if trigger_detector_id is None:
        groups = [g for g in lh5.ls(in_path) if g.startswith("ch")]
        if len(groups) != 1:
            msg = (
                f"{in_path} holds {len(groups)} columns, name the trigger "
                f"detector explicitly"
            )
            raise ValueError(msg)
        group = groups[0]
    else:
        group = f"ch{trigger_detector_id}"

    table = lh5.read(f"{group}/xtalk_column", in_path)

    result = {key: table[key].nda for key in table.keys()}

    json_attrs = set(json.loads(table.attrs.get("json_attrs", "[]")))
    for key, value in table.attrs.items():
        if key in ("datatype", "json_attrs"):
            continue
        result[key] = json.loads(value) if key in json_attrs else value

    return result


# ----------------------------------------------------------------------------
# describing the difference between two baseline objects
# ----------------------------------------------------------------------------


def _usable(value) -> bool:
    """Whether *value* is a number ``xtalk_column`` would actually use."""
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _float32_rank(x: float) -> int:
    """Index of ``float32(x)`` in the ascending order of all float32 values.

    Adjacent float32 values differ by a rank of one, so the difference of two
    ranks counts the representable numbers between them -- the ULP distance.
    """
    bits = int(np.float32(x).view(np.uint32))
    magnitude = bits & 0x7FFFFFFF
    return -magnitude if bits & 0x80000000 else magnitude


def float32_ulps(a: float, b: float) -> int:
    """How many float32 values lie between *a* and *b* once both are rounded."""
    return abs(_float32_rank(a) - _float32_rank(b))


def relative_difference(a: float, b: float) -> float:
    """``|a - b|`` over the larger magnitude; 0.0 when both are zero."""
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def agreeing_digits(a: float, b: float) -> float:
    """Leading significant decimal digits *a* and *b* have in common.

    ``inf`` when they are bit-for-bit equal.
    """
    rel = relative_difference(a, b)
    return math.inf if rel == 0.0 else -math.log10(rel)


def compare_baselines(baseline_a: dict, baseline_b: dict) -> dict:
    """Describe how two baseline objects differ.

    Coverage and usability are compared first, because a detector present in
    only one of the two, or usable in only one of the two, is a difference no
    numerical tolerance can express.  The numbers themselves are then compared
    only over the detectors both sides can actually use.

    Returns
    -------
    dict
        ``only_in_a``/``only_in_b``/``common`` detector id lists,
        ``usable_in_a_only``/``usable_in_b_only``/``usable_in_both``, and
        ``values``: one record per (detector, field) pair compared, each with
        ``a``, ``b``, ``abs_diff``, ``rel_diff``, ``ulps`` (float32) and
        ``digits``.
    """
    ids_a = set(baseline_a)
    ids_b = set(baseline_b)
    common = sorted(ids_a & ids_b, key=str)

    usable_in_both = []
    usable_in_a_only = []
    usable_in_b_only = []
    values = []

    for detector_id in common:
        entry_a = baseline_a[detector_id]
        entry_b = baseline_b[detector_id]

        ok_a = all(_usable(entry_a.get(key)) for key in VALUE_KEYS)
        ok_b = all(_usable(entry_b.get(key)) for key in VALUE_KEYS)

        if ok_a and not ok_b:
            usable_in_a_only.append(detector_id)
            continue
        if ok_b and not ok_a:
            usable_in_b_only.append(detector_id)
            continue
        if not (ok_a or ok_b):
            continue

        usable_in_both.append(detector_id)
        for key in VALUE_KEYS:
            a = float(entry_a[key])
            b = float(entry_b[key])
            values.append(
                {
                    "detector_id": detector_id,
                    "field": key,
                    "a": a,
                    "b": b,
                    "abs_diff": abs(a - b),
                    "rel_diff": relative_difference(a, b),
                    "ulps": float32_ulps(a, b),
                    "digits": agreeing_digits(a, b),
                }
            )

    return {
        "only_in_a": sorted(ids_a - ids_b, key=str),
        "only_in_b": sorted(ids_b - ids_a, key=str),
        "common": common,
        "usable_in_both": usable_in_both,
        "usable_in_a_only": usable_in_a_only,
        "usable_in_b_only": usable_in_b_only,
        "values": values,
    }


# ----------------------------------------------------------------------------
# how many digits each source actually wrote
# ----------------------------------------------------------------------------

_NUMBER = re.compile(
    r'"(positive_baseline|negative_baseline)"\s*:\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)'
)


def printed_digits(json_files: Iterable[str | Path]) -> list[int]:
    """Significant decimal digits of every baseline literal, as written.

    Read off the file text rather than the parsed floats: a float carries no
    memory of how it was printed, and "did the two sources write the same
    number of digits" is a question about the text.
    """
    counts = []
    for path in json_files:
        try:
            text = Path(path).read_text()
        except OSError as e:
            log.warning("%s: cannot be read: %s, skipping", path, e)
            continue
        for _, literal in _NUMBER.findall(text):
            mantissa = literal.split("e")[0].split("E")[0]
            digits = mantissa.lstrip("-").replace(".", "").lstrip("0")
            counts.append(len(digits))
    return counts


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------


def _summarise(name: str, xs: list[float]) -> str:
    if not xs:
        return f"  {name}: no values"
    xs = sorted(xs)
    return (
        f"  {name}: min {xs[0]:.4g}, median {xs[len(xs) // 2]:.4g}, "
        f"max {xs[-1]:.4g}"
    )


def _id_list(ids: list, limit: int = 12) -> str:
    if not ids:
        return "none"
    shown = ", ".join(str(i) for i in ids[:limit])
    return shown if len(ids) <= limit else f"{shown}, ... ({len(ids)} total)"


def format_report(
    report: dict,
    label_a: str,
    label_b: str,
    digits_a: list[int] | None = None,
    digits_b: list[int] | None = None,
    n_worst: int = 5,
) -> str:
    """Render :func:`compare_baselines` output as a readable block of text."""
    lines = [
        "=" * 78,
        f"A = {label_a}",
        f"B = {label_b}",
        "=" * 78,
        "",
        "coverage",
        f"  in both:    {len(report['common'])} detectors",
        f"  only in A:  {_id_list(report['only_in_a'])}",
        f"  only in B:  {_id_list(report['only_in_b'])}",
        "",
        "usability (both baselines present and not NaN)",
        f"  usable in both:   {len(report['usable_in_both'])}",
        f"  usable in A only: {_id_list(report['usable_in_a_only'])}",
        f"  usable in B only: {_id_list(report['usable_in_b_only'])}",
        "",
    ]

    values = report["values"]
    if not values:
        lines.append("no detector is usable on both sides, nothing to compare")
        return "\n".join(lines)

    identical = [v for v in values if v["abs_diff"] == 0.0]
    within_1_ulp = [v for v in values if v["ulps"] <= 1]
    lines += [
        f"numeric agreement over {len(values)} values "
        f"({len(report['usable_in_both'])} detectors x {len(VALUE_KEYS)} fields)",
        f"  bit-for-bit identical:       {len(identical)} / {len(values)}",
        f"  within 1 float32 ULP:        {len(within_1_ulp)} / {len(values)}",
        _summarise("relative difference", [v["rel_diff"] for v in values]),
        _summarise("float32 ULP distance", [float(v["ulps"]) for v in values]),
        _summarise(
            "agreeing significant digits",
            [v["digits"] for v in values if math.isfinite(v["digits"])],
        ),
        "",
    ]

    worst = sorted(values, key=lambda v: v["rel_diff"], reverse=True)[:n_worst]
    lines.append(f"{len(worst)} largest relative differences")
    for v in worst:
        lines.append(
            f"  ch{v['detector_id']} {v['field']:18s} "
            f"A={v['a']!r:22s} B={v['b']!r:22s} "
            f"rel={v['rel_diff']:.3e}  {v['ulps']} float32 ulp"
        )
    lines.append("")

    if digits_a is not None and digits_b is not None:
        lines.append("significant digits as printed in the source JSON")
        for label, counts in ((f"A ({len(digits_a)} values)", digits_a),
                              (f"B ({len(digits_b)} values)", digits_b)):
            if counts:
                lines.append(
                    f"  {label}: min {min(counts)}, max {max(counts)}, "
                    f"{sorted(set(counts))}"
                )
            else:
                lines.append(f"  {label}: no values")
        lines.append("")

    return "\n".join(lines)
