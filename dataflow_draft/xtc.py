"""Dataflow entry points for cross-talk (XTC) par generation.

The measurement is split into two steps so that the expensive per-detector
reading parallelises over channels the way the rest of the dataflow does:

``build_xtc_detector_info``
    Runs once per channel.  Reads that channel's hit and dsp tiers, makes the
    baseline, trigger and response selections with
    :func:`~pygama.pargen.xtc.prepare_detector`, and writes the result to a
    small lh5 file.
``build_xtc_matrix``
    Runs once per run, over every channel's file from the first step.  Measures
    each ``(trigger, response)`` pair with
    :func:`~pygama.pargen.xtc.xtalk_element` and assembles them into the
    cross-talk matrix with :func:`~pygama.pargen.xtc.build_xtalk_matrix`.

Both steps are also runnable directly, which is what the SLURM/bash scripts
that exercise them before there are snakemake rules do::

    python xtc.py detector-info --configs ... --output ...
    python xtc.py matrix --configs ... --output ...
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle as pkl
import sys
from pathlib import Path

import lgdo
import lh5
import matplotlib as mpl
import numpy as np
from dbetto import Props, TextDB
from legenddataflowscripts.utils import build_log

mpl.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from pygama.pargen.xtc import (  # noqa: E402
    attach_response_amps,
    build_xtalk_matrix,
    plot_xtalk_matrix,
    prepare_detector,
    xtalk_element,
)

log = logging.getLogger(__name__)

#: The cross-talk ratio is only meaningful when every detector reads out on
#: every trigger, so only the datatypes taken in that trigger mode are
#: supported.
SUPPORTED_DATATYPES = ("ssc", "xtc")

#: Group the cross-talk table is written under.  ``pygama.evt.modules.xtalk``
#: reads ``xtc/rawid_index`` and ``xtc/xtalk_matrix_{negative,positive}`` by
#: default, so the name is not free to change.
XTC_LH5_GROUP = "xtc"

#: Fields of a :func:`~pygama.pargen.xtc.prepare_detector` result that are
#: written as lh5 datasets.  They are the only large ones.
DETECTOR_INFO_ARRAYS = ("response_keep", "trigger_idxs", "trigger_amplitudes")

#: Fields written as lh5 scalars.  The two baselines are ``None`` when they
#: could not be measured and are stored as NaN; the three flags are booleans
#: and are stored as 0/1.
DETECTOR_INFO_BASELINES = ("positive_baseline", "negative_baseline")
DETECTOR_INFO_FLAGS = ("read_success", "baseline_success", "trigger_success")

#: Per-element histograms, dropped once the element has been fitted.  They are
#: about 22 kB an element, which over the N^2 elements of a full array is more
#: memory than the matrix they feed is worth.
ELEMENT_HISTOGRAM_FIELDS = tuple(
    f"{polarity}_{field}" for polarity in ("neg", "pos") for field in ("counts", "bins")
)


def _check_datatype(datatype: str) -> None:
    """Raise unless *datatype* is one this routine can be run on."""
    if datatype not in SUPPORTED_DATATYPES:
        msg = (
            f"unsupported datatype {datatype}: the cross-talk measurement needs "
            f"every detector read out on every trigger, which is only the case "
            f"for {SUPPORTED_DATATYPES}"
        )
        raise NotImplementedError(msg)


def _expand_filelist(files: list[str] | None) -> list[str]:
    """Resolve any ``.filelist`` arguments to the sorted files they name.

    Snakemake hands long file lists over as ``.filelist`` files rather than as
    thousands of arguments -- one per run, so a rule spanning several runs
    passes several of them.  Entries that are not ``.filelist`` files are
    already paths to data and are taken as they are.

    Sorting and de-duplicating whichever form was given is what makes a hit
    list and a dsp list line up: the tier is the only part of the two paths
    that differs, so the same sort puts both in the same run order, which is
    the order :func:`~pygama.pargen.xtc.prepare_detector` needs them in to
    match an event in one tier to the same event in the other.  The duplicates
    are real -- a file can be named by both a run filelist and a period one.
    """
    expanded = []
    for entry in files or []:
        if Path(entry).suffix == ".filelist":
            with Path(entry).open() as f:
                expanded += [line for line in f.read().splitlines() if line]
        else:
            expanded.append(entry)

    return sorted(set(expanded))


def _parse_rawid(rawid: str) -> int:
    """``"1104000"`` or ``"ch1104000"`` to ``1104000``."""
    rawid = str(rawid)
    return int(rawid[2:] if rawid.startswith("ch") else rawid)


def _write_detector_info(detector_info: dict, output: str) -> None:
    """Write one :func:`~pygama.pargen.xtc.prepare_detector` result to lh5.

    The per-event arrays become the datasets of a ``ch{rawid}`` struct and the
    scalars its :class:`~lgdo.types.scalar.Scalar` fields.  What is neither --
    the detector id, the timestamp and the nested ``parameters`` dict -- is
    json-encoded into the struct's attributes, ``parameters`` because
    :func:`~pygama.pargen.xtc.attach_response_amps` reads the dsp files and
    fields back out of it in the second step.
    """
    detector_id = detector_info["detector_id"]

    obj_dict = {
        name: lgdo.Array(np.asarray(detector_info[name]))
        for name in DETECTOR_INFO_ARRAYS
    }
    obj_dict["n_rows"] = lgdo.Scalar(int(detector_info["n_rows"]))
    for name in DETECTOR_INFO_BASELINES:
        value = detector_info[name]
        obj_dict[name] = lgdo.Scalar(np.nan if value is None else float(value))
    for name in DETECTOR_INFO_FLAGS:
        obj_dict[name] = lgdo.Scalar(int(bool(detector_info[name])))

    struct = lgdo.Struct(
        obj_dict=obj_dict,
        attrs={
            "detector_id": str(detector_id),
            # the rule knows it, and carrying it here is what lets the matrix
            # step name a detector without a channel map of its own
            "detector_name": str(detector_info.get("detector_name") or ""),
            "processed_at": detector_info["processed_at"],
            "parameters": json.dumps(detector_info["parameters"]),
        },
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    lh5.write(struct, name=f"ch{detector_id}", lh5_file=output, wo_mode="of")


def _detector_name(detector_info: dict) -> str:
    """``"1104000 (V02160A)"``, for a log line somebody has to read."""
    name = detector_info.get("detector_name")
    return f"{detector_info['detector_id']}{f' ({name})' if name else ''}"


def _read_detector_info(path: str) -> dict:
    """Read back what :func:`_write_detector_info` wrote.

    The returned dict is what :func:`~pygama.pargen.xtc.prepare_detector`
    returned, minus nothing that the second step uses.  The channel is taken
    from the file rather than from its name, so the caller does not have to
    know which channel it is handing over.
    """
    groups = lh5.ls(path)
    if len(groups) != 1:
        msg = (
            f"{path} holds {len(groups)} groups ({groups}), but a detector info "
            f"file holds exactly one, the channel it was prepared from"
        )
        raise RuntimeError(msg)

    struct = lh5.read(groups[0], path)

    detector_info = {
        "detector_id": _parse_rawid(struct.attrs["detector_id"]),
        "detector_name": struct.attrs.get("detector_name", ""),
        "processed_at": struct.attrs["processed_at"],
        "parameters": json.loads(struct.attrs["parameters"]),
        "n_rows": int(struct["n_rows"].value),
    }
    for name in DETECTOR_INFO_ARRAYS:
        detector_info[name] = struct[name].nda
    for name in DETECTOR_INFO_BASELINES:
        value = float(struct[name].value)
        detector_info[name] = None if np.isnan(value) else value
    for name in DETECTOR_INFO_FLAGS:
        detector_info[name] = bool(struct[name].value)

    return detector_info


def build_xtc_detector_info() -> None:
    """Prepare one channel for the cross-talk measurement."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--hit-files", help="hit files", nargs="*", type=str)
    argparser.add_argument("--dsp-files", help="dsp files", nargs="*", type=str)

    argparser.add_argument("--configs", help="configs path", type=str, required=True)
    argparser.add_argument("--log", help="log file", type=str)

    argparser.add_argument("--datatype", help="datatype", type=str, required=True)
    argparser.add_argument("--timestamp", help="timestamp", type=str, required=True)
    # the rule resolves the detector name its output file is keyed by to the
    # rawid the tiers are keyed by, so that this script needs no channel map
    argparser.add_argument("--channel", help="channel name", type=str, required=True)
    argparser.add_argument("--rawid", help="rawid", type=str, required=True)

    argparser.add_argument("--output", help="output file", type=str, required=True)
    argparser.add_argument("-d", "--debug", help="debug mode", action="store_true")
    args = argparser.parse_args()

    _check_datatype(args.datatype)

    df_config = (
        TextDB(args.configs, lazy=True)
        .on(args.timestamp, system=args.datatype)
        .snakemake_rules.pars_geds_xtc_detector_info
    )
    log = build_log(df_config, args.log)

    # both steps read the same file, so that the matrix and the detectors it is
    # measured from cannot be configured apart from one another
    config = Props.read_from(df_config.inputs.xtc_config)

    hit_files = _expand_filelist(args.hit_files)
    dsp_files = _expand_filelist(args.dsp_files)
    if not hit_files or not dsp_files:
        msg = (
            f"channel {args.rawid} was given {len(hit_files)} hit files and "
            f"{len(dsp_files)} dsp files, and it needs both"
        )
        raise ValueError(msg)

    rawid = _parse_rawid(args.rawid)
    log.info(
        "preparing channel %s (%s) from %s hit and %s dsp files",
        rawid,
        args.channel,
        len(hit_files),
        len(dsp_files),
    )

    detector_info = prepare_detector(
        hit_files=hit_files,
        dsp_files=dsp_files,
        chn_id=rawid,
        config=config.get("detector", {}),
        debug_mode=args.debug,
    )
    detector_info["detector_name"] = args.channel

    _write_detector_info(detector_info, args.output)
    log.info("wrote the selections of channel %s to %s", rawid, args.output)


def build_xtc_matrix() -> None:
    """Measure every pair of prepared channels and assemble the matrix."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--detector-files", help="detector info files", nargs="*", required=True
    )

    argparser.add_argument("--configs", help="configs path", type=str, required=True)
    argparser.add_argument("--log", help="log file", type=str)

    argparser.add_argument("--datatype", help="datatype", type=str, required=True)
    argparser.add_argument("--timestamp", help="timestamp", type=str, required=True)

    argparser.add_argument("--plot-file", help="plot file", type=str)
    argparser.add_argument("--output", help="output file", type=str, required=True)
    argparser.add_argument("-d", "--debug", help="debug mode", action="store_true")
    args = argparser.parse_args()

    _check_datatype(args.datatype)

    df_config = (
        TextDB(args.configs, lazy=True)
        .on(args.timestamp, system=args.datatype)
        .snakemake_rules.pars_geds_xtc_matrix
    )
    log = build_log(df_config, args.log)

    config = Props.read_from(df_config.inputs.xtc_config)

    # every channel is held at once: the pairs are measured against each other,
    # so re-reading one per pair would be N^2 reads of the same file
    detector_files = _expand_filelist(args.detector_files)
    detector_info = {}
    for path in detector_files:
        info = _read_detector_info(path)
        rawid = info["detector_id"]
        if rawid in detector_info:
            msg = f"channel {rawid} was prepared by more than one input file"
            raise ValueError(msg)
        detector_info[rawid] = info

    if not detector_info:
        msg = "no detector info files were given, so there is nothing to measure"
        raise ValueError(msg)

    # the matrix is indexed in this order, so fix it rather than leaving it to
    # the order snakemake happened to pass the files in
    rawids = sorted(detector_info)
    log.info(
        "measuring the %s x %s elements of %s",
        len(rawids),
        len(rawids),
        ", ".join(_detector_name(detector_info[rawid]) for rawid in rawids),
    )

    # responding detector on the outside: attaching its amplitudes reads its
    # dsp files, which is the expensive part, and the triggering detector needs
    # nothing that is not already in its detector info
    element_config = config.get("element", {})
    fitted_elements = []
    for response_id in rawids:
        response_info = attach_response_amps(
            detector_info[response_id], debug_mode=args.debug
        )
        for trigger_id in rawids:
            element = xtalk_element(
                detector_info[trigger_id],
                response_info,
                config=element_config,
                debug_mode=args.debug,
            )
            for field in ELEMENT_HISTOGRAM_FIELDS:
                element.pop(field, None)
            fitted_elements.append(element)
        del response_info

    xtalk_table = build_xtalk_matrix(fitted_elements, config=config.get("matrix", {}))

    # TODO: review if this is really necessary 
    written_rawids = xtalk_table["rawid_index"].nda
    if not np.array_equal(written_rawids, rawids):
        msg = (
            f"the matrix is indexed {written_rawids.tolist()} but its elements "
            f"were measured over {rawids}"
        )
        raise RuntimeError(msg)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    lh5.write(xtalk_table, name=XTC_LH5_GROUP, lh5_file=args.output, wo_mode="of")
    log.info("wrote the cross-talk matrix to %s", args.output)

    if args.plot_file:
        plot_config = config.get("plot", {})
        plot_dict = {}
        for polarity in ("neg", "pos"):
            kwargs = dict(plot_config)
            kwargs.setdefault(
                "title", f"{polarity} cross-talk matrix, {args.timestamp}"
            )
            plot_dict[polarity] = plot_xtalk_matrix(
                xtalk_table, polarity=polarity, **kwargs
            )

        Path(args.plot_file).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.plot_file).open("wb") as w:
            pkl.dump(plot_dict, w, protocol=pkl.HIGHEST_PROTOCOL)

        for figure in plot_dict.values():
            plt.close(figure)
        log.info("wrote the cross-talk matrix plots to %s", args.plot_file)


#: The two steps, under the names ``python xtc.py <step>`` takes.
STEPS = {
    "detector-info": build_xtc_detector_info,
    "matrix": build_xtc_matrix,
}


def main() -> None:
    """Dispatch to the step named by the first argument.

    Snakemake calls the two entry points directly; this is for the bash and
    SLURM scripts that call them before there are rules to do so.
    """
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        msg = f"usage: {Path(sys.argv[0]).name} {{{'|'.join(STEPS)}}} [options]"
        raise SystemExit(msg)

    STEPS[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
