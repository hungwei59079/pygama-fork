"""
This module provides routines for measuring cross-talk (XTC) between
germanium channels and for building the resulting cross-talk matrix.

The four main functions, in order of execution, are:
get_baseline_and_trigger_amps, xtalk_column, xtalk_histogram_fitter, and
build_xtalk_matrix.

A column of the matrix fixes the *responding* detector and runs over the
triggers.  Everything a trigger detector contributes to every column -- the
events it selected and the amplitude each of them fired with -- is measured
once per channel by get_baseline_and_trigger_amps, in the same pass that
measures that channel's baselines, so filling a column costs one pass over
one channel rather than one pass per detector pair.

:func:`plot_xtalk_matrix` draws what the last of them returns.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import lgdo
import lh5
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import pygama.math.histogram as pgh
from pygama.math.functions.gauss import nb_gauss_amp

log = logging.getLogger(__name__)

DEFAULT_ENERGY_PARAM = "cuspEmax_ctc_cal"
DEFAULT_BASELINE_CONDITIONS = {"is_empty_candidate": 63}
DEFAULT_TRIGGER_CONDITIONS = {"is_highly_positive_polarity_candidate": 511}
DEFAULT_POSITIVE_PARAM = "trapTmax"
DEFAULT_NEGATIVE_PARAM = "trapTmin"
DEFAULT_TRIGGER_PARAM = "trapTmax"
DEFAULT_TRIGGER_ENERGY_RANGE = (1500, 99999)
DEFAULT_RESPONSE_ENERGY_RANGE = (-99999, 100)
DEFAULT_NBINS = 700
DEFAULT_RANGE_MULTIPLIER = 3
DEFAULT_LOW_STATS_THRESHOLD = 100
DEFAULT_Y_MASK_THRESHOLD = 0.05
DEFAULT_SHARP_FIT_MIN_POINTS = 5

DEFAULT_BUFFER_LEN = 100000

_DSP_SUFFIX = "_dsp"

#: Outcome of fitting one histogram, ordered from the most to the least
#: trustworthy.  Written into the lh5 file as ``fit_status_codes`` so a
#: reader never has to hard-code these numbers.
FIT_STATUS = {
    "ok": 0,
    "ok_few_points": 1,
    "low_stats": 2,
    "fit_failed": 3,
    "no_stats": 4,
    "not_filled": 5,
}
FIT_STATUS_SUCCESS = (FIT_STATUS["ok"], FIT_STATUS["ok_few_points"])

XTC_LH5_FIELD = {"neg": "xtalk_matrix_negative", "pos": "xtalk_matrix_positive"}
XTC_PLOT_RANGE = {"neg": (-0.003, 0.001), "pos": (-0.0007, 0.003)}

def _selection_mask(
    table,
    ene_field: str,
    conditions: dict | None = None,
    energy_range: tuple | None = None,
) -> np.ndarray:
    """
    Rows of *table* that survive the event cuts, as a boolean mask.
    """
    energies = table[ene_field].nda
    mask = ~np.isnan(energies)

    for flag, value in (conditions or {}).items():
        mask &= table[flag].nda == value

    if energy_range is not None:
        emin, emax = energy_range
        mask &= (energies >= emin) & (energies <= emax)

    return mask


def _usable_baseline(value) -> float | None:
    """*value* as a float, or None when it is not a number to subtract."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def get_baseline_and_trigger_amps(
    hit_files: str | list,
    dsp_files: str | list,
    chn_id: str | int,
    config: dict | None = None,
    buffer_len: int = DEFAULT_BUFFER_LEN,
    debug_mode: bool = False,
) -> dict:
    """Measure the baselines of one channel and select the events it triggers.

    Both products come out of a single pass over the channel because both are
    per-channel quantities that every cross-talk column needs: the baselines
    are subtracted when this channel responds, and the trigger events are the
    ones every other channel is histogrammed over when this channel triggers.

    The baseline half selects the events whose hit-tier flags match
    ``config["baseline_conditions"]`` and averages the positive- and
    negative-going DSP amplitudes over exactly those events.  The trigger half
    selects the events whose flags match ``config["trigger_conditions"]`` and
    whose energy falls inside ``config["trigger_energy_range"]``, and keeps
    their global entry numbers together with the DSP amplitude
    :func:`xtalk_column` divides by.

    The two selections are independent: either can fail on its own, and each
    reports its own flag.  A failure to read the channel at all fails both.

    Parameters
    ----------
    hit_files
        Hit-tier file, or list of files, to read the selection flags from.
    dsp_files
        DSP-tier file, or list of files, holding the amplitudes.  Must cover
        the same events, in the same order, as *hit_files*.
    chn_id
        Channel identifier (rawid) of the detector, without the ``ch``
        prefix.  Tables are read from ``ch{chn_id}/hit/`` and
        ``ch{chn_id}/dsp/``.
    config
        Selection configuration.  Recognised keys, all optional:

        ``baseline_conditions``
            Mapping of hit-tier flag field to the value it must equal for an
            event to count as baseline.  Default ``{"is_empty_candidate": 63}``.
        ``trigger_conditions``
            Mapping of hit-tier flag field to the value it must equal for an
            event to count as a trigger.  Default
            ``{"is_highly_positive_polarity_candidate": 511}``.
        ``trigger_energy_range``
            ``(emin, emax)`` on ``energy_param`` selecting real triggers.
            Default ``(1500, 99999)``.
        ``energy_param``
            Hit-tier field both selections are applied to.  Default
            ``"cuspEmax_ctc_cal"``.
        ``positive_param``, ``negative_param``
            DSP-tier fields averaged to give the positive and negative
            baselines.  Default ``"trapTmax"`` and ``"trapTmin"``.
        ``trigger_param``
            DSP-tier field giving the trigger energy that ends up in the
            denominator of the cross-talk ratio.  Default ``"trapTmax"``.
    buffer_len
        Rows read per chunk.
    debug_mode
        If True, re-raise instead of falling back to a null result.

    Returns
    -------
    dict
        Keys ``detector_id``, ``positive_baseline``, ``negative_baseline``
        (``None`` when that measurement failed), ``trigger_idxs`` and
        ``trigger_amplitudes`` (both empty when the trigger selection failed),
        ``baseline_success``, ``trigger_success``, ``processed_at`` and
        ``parameters``.

    Notes
    -----
    ``trigger_idxs`` are global entry numbers into *hit_files* read as one
    concatenated table, so they address a row of another channel correctly
    only if that channel covers the same events in the same order.  That is
    the same assumption :func:`xtalk_column` makes of the pair.
    """
    config = config or {}
    baseline_conditions = dict(
        config.get("baseline_conditions", DEFAULT_BASELINE_CONDITIONS)
    )
    trigger_conditions = dict(
        config.get("trigger_conditions", DEFAULT_TRIGGER_CONDITIONS)
    )
    trigger_energy_range = tuple(
        config.get("trigger_energy_range", DEFAULT_TRIGGER_ENERGY_RANGE)
    )
    energy_param = config.get("energy_param", DEFAULT_ENERGY_PARAM)
    positive_param = config.get("positive_param", DEFAULT_POSITIVE_PARAM)
    negative_param = config.get("negative_param", DEFAULT_NEGATIVE_PARAM)
    trigger_param = config.get("trigger_param", DEFAULT_TRIGGER_PARAM)

    # the three dsp fields and the flags of both selections are usually not
    # three and two distinct fields, so read each of them only once
    dsp_fields = list(dict.fromkeys([positive_param, negative_param, trigger_param]))
    hit_fields = list(
        dict.fromkeys([energy_param, *baseline_conditions, *trigger_conditions])
    )

    baseline_success = True
    trigger_success = True
    positive_baseline = None
    negative_baseline = None
    trigger_idxs = np.empty(0, dtype=np.int64)
    trigger_amplitudes = np.empty(0)

    n_baseline = 0
    positive_chunks = []
    negative_chunks = []
    idx_chunks = []
    amplitude_chunks = []

    read_ok = True
    try:
        dsp_iterator = lh5.LH5Iterator(
            dsp_files,
            f"ch{chn_id}/dsp",
            field_mask=dsp_fields,
            buffer_len=buffer_len,
        )
        hit_iterator = lh5.LH5Iterator(
            hit_files,
            f"ch{chn_id}/hit",
            field_mask=hit_fields,
            buffer_len=buffer_len,
            friend=dsp_iterator,
            friend_suffix=_DSP_SUFFIX,
        )

        for table in hit_iterator:
            baseline_mask = _selection_mask(table, energy_param, baseline_conditions)
            n_baseline += int(baseline_mask.sum())
            positive = table[f"{positive_param}{_DSP_SUFFIX}"].nda[baseline_mask]
            negative = table[f"{negative_param}{_DSP_SUFFIX}"].nda[baseline_mask]
            positive_chunks.append(positive[np.isfinite(positive)])
            negative_chunks.append(negative[np.isfinite(negative)])

            trigger_mask = _selection_mask(
                table, energy_param, trigger_conditions, trigger_energy_range
            )
            idx_chunks.append(hit_iterator.current_global_entries[trigger_mask])
            amplitude_chunks.append(
                table[f"{trigger_param}{_DSP_SUFFIX}"].nda[trigger_mask]
            )
    except Exception as e:
        if debug_mode:
            raise
        log.error(
            "reading channel %s failed, neither its baseline nor its triggers "
            "were measured: %s: %s",
            chn_id,
            type(e).__name__,
            e,
        )
        read_ok = False
        baseline_success = False
        trigger_success = False

    if read_ok:
        try:
            if n_baseline == 0:
                msg = "no events passed the baseline selection"
                raise RuntimeError(msg)

            positive_selected = np.concatenate(positive_chunks)
            negative_selected = np.concatenate(negative_chunks)
            if len(positive_selected) == 0 or len(negative_selected) == 0:
                msg = "no baseline events survived the dsp-tier non-finite cut"
                raise RuntimeError(msg)

            positive_baseline = float(np.mean(positive_selected))
            negative_baseline = float(np.mean(negative_selected))
        except Exception as e:
            if debug_mode:
                raise
            log.error("baseline preparation failed for channel %s: %s", chn_id, e)
            positive_baseline = None
            negative_baseline = None
            baseline_success = False

        try:
            if not idx_chunks:
                msg = "no events passed the trigger selection"
                raise RuntimeError(msg)

            idxs = np.concatenate(idx_chunks).astype(np.int64)
            amplitudes = np.concatenate(amplitude_chunks)

            # a zero or non-finite trigger amplitude cannot be divided by
            usable = np.isfinite(amplitudes) & (amplitudes != 0)
            if not usable.all():
                log.debug(
                    "trigger %s: dropping %d of %d events with a non-finite or zero %s",
                    chn_id,
                    int((~usable).sum()),
                    len(usable),
                    trigger_param,
                )
                idxs = idxs[usable]
                amplitudes = amplitudes[usable]

            if len(idxs) == 0:
                msg = "no events passed the trigger selection"
                raise RuntimeError(msg)

            trigger_idxs = idxs
            trigger_amplitudes = amplitudes
        except Exception as e:
            if debug_mode:
                raise
            log.error("trigger selection failed for channel %s: %s", chn_id, e)
            trigger_idxs = np.empty(0, dtype=np.int64)
            trigger_amplitudes = np.empty(0)
            trigger_success = False

    log.info(
        "channel %s: baseline %s, %s trigger events",
        chn_id,
        "measured" if baseline_success else "not measured",
        len(trigger_idxs),
    )

    return {
        "detector_id": chn_id,
        "positive_baseline": positive_baseline,
        "negative_baseline": negative_baseline,
        "trigger_idxs": trigger_idxs,
        "trigger_amplitudes": trigger_amplitudes,
        "baseline_success": baseline_success,
        "trigger_success": trigger_success,
        "processed_at": datetime.now().isoformat(),
        "parameters": {
            "baseline_conditions": baseline_conditions,
            "trigger_conditions": trigger_conditions,
            "trigger_energy_range": list(trigger_energy_range),
            "energy_param": energy_param,
            "positive_param": positive_param,
            "negative_param": negative_param,
            "trigger_param": trigger_param,
        },
    }


def _resolve_trigger(
    triggers: dict, chn_id: str | int
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return ``(idxs, amplitudes)`` for *chn_id*, or None if unusable.
    """
    entry = triggers.get(chn_id)
    if entry is None:
        entry = triggers.get(str(chn_id))
    if not isinstance(entry, dict):
        return None

    idxs = entry.get("trigger_idxs")
    amplitudes = entry.get("trigger_amplitudes")
    if idxs is None or amplitudes is None:
        return None

    idxs = np.asarray(idxs, dtype=np.int64)
    # not cast to float64: the cross-talk ratio is computed in whatever
    # precision the dsp tier stored the amplitude in
    amplitudes = np.asarray(amplitudes)
    if idxs.size != amplitudes.size:
        msg = (
            f"channel {chn_id} has {idxs.size} trigger indices but "
            f"{amplitudes.size} trigger amplitudes"
        )
        raise ValueError(msg)
    if idxs.size == 0:
        return None

    return idxs, amplitudes


def _build_hist(
    vals: np.ndarray, nbins: int, range_multiplier: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Histogram *vals* over ``mean +/- range_multiplier * stdev``.

    Returns ``None`` when the sample is empty or has no usable spread.
    """
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    mean = np.mean(vals)
    stdev = np.std(vals)
    if not np.isfinite(mean) or not np.isfinite(stdev) or stdev <= 0:
        return None

    return np.histogram(
        vals,
        bins=nbins,
        range=(mean - range_multiplier * stdev, mean + range_multiplier * stdev),
    )


def xtalk_column(
    hit_files: str | list,
    dsp_files: str | list,
    response_detector_id: str | int,
    positive_baseline: float | str,
    negative_baseline: float | str,
    triggers: dict,
    config: dict | None = None,
    buffer_len: int = DEFAULT_BUFFER_LEN,
    debug_mode: bool = False,
) -> dict:
    """Fill the histograms for one column of the cross-talk matrix.

    A column fixes the *responding* detector: it holds one element per trigger
    detector, each the distribution of the energy *response_detector_id* picked
    up while that trigger fired.

    The response channel is read once, up front, and every element is then
    served by indexing those arrays at the trigger events that
    :func:`get_baseline_and_trigger_amps` already selected.  That is what makes
    a column cost one pass over one channel rather than one pass per pair, and
    it is why the only baseline a column needs is the response's own.

    Elements skipped are recorded with ``valid = False`` and an empty
    histogram.  This happens when the trigger channel is the response itself,
    or when the trigger channel selected no usable events.  If the response
    channel has no usable baseline the whole column is skipped.

    Parameters
    ----------
    hit_files
        Hit-tier file, or list of files, holding the selection flags.
    dsp_files
        DSP-tier file, or list of files, holding the amplitudes.  Must cover
        the same events, in the same order, as *hit_files*.
    response_detector_id
        Channel id of the responding detector, without the ``ch`` prefix.
    positive_baseline, negative_baseline
        The response channel's own baselines, as measured by
        :func:`get_baseline_and_trigger_amps`.  Subtracted from the positive-
        and negative-going amplitudes before the ratio is taken.  Anything
        :func:`float` accepts will do.
    triggers
        Per-channel trigger selections, as produced by
        :func:`get_baseline_and_trigger_amps` and collected by channel id.
        The keys are the trigger detectors the column covers, in the order
        its elements come out in.  Each value needs at least the structure:
        {chn_id: {"trigger_idxs": array, "trigger_amplitudes": array}, ...}

        for example:
            {
                1104000: {"trigger_idxs": [3, 17], "trigger_amplitudes": [2.1, 3.4]},
                1104001: {"trigger_idxs": [], "trigger_amplitudes": []},
                ...
            }
    config
        Selection and histogram configuration.  Recognised keys, all
        optional:

        ``energy_param``
            Hit-tier field the response selection is applied to.  Default
            ``"cuspEmax_ctc_cal"``.
        ``positive_param``, ``negative_param``
            DSP-tier response fields histogrammed against the positive and
            negative baselines.  Default ``"trapTmax"`` and ``"trapTmin"``.
        ``response_conditions``
            Mapping of hit-tier flag field to the value it must equal for the
            response selection.  Default ``{}``.
        ``response_energy_range``
            ``(emin, emax)`` on ``energy_param`` selecting events in which the
            response channel did *not* see a real hit -- an event in which it
            did is a multiplicity event, not cross-talk.  Default
            ``(-99999, 100)``.
        ``nbins``
            Bins per histogram.  Default 700.
        ``range_multiplier``
            Histogram half-width in standard deviations about the mean.
            Default 3.

        The trigger side of the selection is not configured here: it was
        already applied by :func:`get_baseline_and_trigger_amps`, and the
        configuration it used is recorded in its own result.
    buffer_len
        Rows read per chunk during the response selection.
    debug_mode
        If True, re-raise instead of falling back to an empty column or an
        empty element.

    Returns
    -------
    dict
        ``response_id``, ``trigger_ids`` ``(N,)``, ``valid`` ``(N,)`` bool,
        ``n_events`` ``(N,)``, ``neg_counts``/``pos_counts`` ``(N, nbins)``,
        ``neg_bins``/``pos_bins`` ``(N, nbins + 1)``, ``parameters`` and
        ``processed_at``.  Bins are NaN wherever the histogram is empty.
    """

    config = config or {}
    energy_param = config.get("energy_param", DEFAULT_ENERGY_PARAM)
    positive_param = config.get("positive_param", DEFAULT_POSITIVE_PARAM)
    negative_param = config.get("negative_param", DEFAULT_NEGATIVE_PARAM)
    response_conditions = dict(config.get("response_conditions", {}))
    response_energy_range = tuple(
        config.get("response_energy_range", DEFAULT_RESPONSE_ENERGY_RANGE)
    )
    nbins = int(config.get("nbins", DEFAULT_NBINS))
    range_multiplier = float(config.get("range_multiplier", DEFAULT_RANGE_MULTIPLIER))

    trigger_id_list = list(triggers.keys())
    n_trigger = len(trigger_id_list)
    neg_counts = np.zeros((n_trigger, nbins), dtype=np.int64)
    pos_counts = np.zeros((n_trigger, nbins), dtype=np.int64)
    neg_bins = np.full((n_trigger, nbins + 1), np.nan)
    pos_bins = np.full((n_trigger, nbins + 1), np.nan)
    valid = np.zeros(n_trigger, dtype=bool)
    n_events = np.zeros(n_trigger, dtype=np.int64)

    response_keep = None
    positive_response = None
    negative_response = None

    # response selection. Only needs to be done once per column, because it
    # does not depend on which channel triggered.
    try:
        positive_baseline = _usable_baseline(positive_baseline)
        negative_baseline = _usable_baseline(negative_baseline)
        if positive_baseline is None or negative_baseline is None:
            msg = f"response channel {response_detector_id} has no usable baseline"
            raise RuntimeError(msg)

        try:
            dsp_iterator = lh5.LH5Iterator(
                dsp_files,
                f"ch{response_detector_id}/dsp",
                field_mask=[positive_param, negative_param],
                buffer_len=buffer_len,
            )
            hit_iterator = lh5.LH5Iterator(
                hit_files,
                f"ch{response_detector_id}/hit",
                field_mask=[energy_param, *response_conditions],
                buffer_len=buffer_len,
                friend=dsp_iterator,
                friend_suffix=_DSP_SUFFIX,
            )

            keep_chunks = []
            positive_chunks = []
            negative_chunks = []
            n_read = 0
            for table in hit_iterator:
                # a trigger index addresses a row of these arrays directly, so
                # the chunks have to arrive in order and leave no gap
                entries = hit_iterator.current_global_entries
                if len(entries) and entries[0] != n_read:
                    msg = (
                        f"the buffer starts at global entry {entries[0]} where "
                        f"{n_read} was expected, so row {n_read} of this channel "
                        f"is not the event a trigger index of {n_read} means"
                    )
                    raise RuntimeError(msg)
                n_read += len(entries)

                keep_chunks.append(
                    _selection_mask(
                        table,
                        energy_param,
                        response_conditions,
                        response_energy_range,
                    )
                )
                positive_chunks.append(table[f"{positive_param}{_DSP_SUFFIX}"].nda)
                negative_chunks.append(table[f"{negative_param}{_DSP_SUFFIX}"].nda)
        except Exception as e:
            msg = f"response event selection failed: {type(e).__name__}: {e}"
            raise RuntimeError(msg) from e

        if n_read == 0:
            msg = "the response channel holds no events"
            raise RuntimeError(msg)

        response_keep = np.concatenate(keep_chunks)
        positive_response = np.concatenate(positive_chunks)
        negative_response = np.concatenate(negative_chunks)

    except Exception as e:
        if debug_mode:
            raise
        log.error(
            "xtalk column for response %s failed, writing an empty column: %s",
            response_detector_id,
            e,
        )
        response_keep = None

    # loop over trigger detectors starts here
    # If the response selection failed or the response baseline is None,
    # skip the whole loop to saving an empty column.
    if response_keep is not None:
        n_total = len(response_keep)
        for k, trigger_id in enumerate(trigger_id_list):
            if str(trigger_id) == str(response_detector_id):
                log.debug("self-interaction at channel %s ignored", trigger_id)
                continue

            resolved = _resolve_trigger(triggers, trigger_id)
            if resolved is None:
                log.debug(
                    "trigger channel %s selected no usable events, skipping",
                    trigger_id,
                )
                continue
            trigger_idxs, trigger_amplitudes_all = resolved

            try:
                if trigger_idxs.max() >= n_total:
                    msg = (
                        f"trigger index {trigger_idxs.max()} is past the "
                        f"{n_total} events of channel {response_detector_id}, so "
                        f"the two channels do not cover the same events"
                    )
                    raise IndexError(msg)

                keep = response_keep[trigger_idxs]
                coincident_idxs = trigger_idxs[keep]
                trigger_amplitudes = trigger_amplitudes_all[keep]

                neg_vals = (
                    negative_response[coincident_idxs] - negative_baseline
                ) / trigger_amplitudes
                pos_vals = (
                    positive_response[coincident_idxs] - positive_baseline
                ) / trigger_amplitudes
            except Exception as e:
                if debug_mode:
                    raise
                log.error(
                    "xtalk element (%s, %s) failed: %s",
                    trigger_id,
                    response_detector_id,
                    e,
                )
                continue

            valid[k] = True
            n_events[k] = len(trigger_amplitudes)

            neg_hist = _build_hist(neg_vals, nbins, range_multiplier)
            if neg_hist is not None:
                neg_counts[k], neg_bins[k] = neg_hist
            else:
                log.debug(
                    "negative histogram for element (%s, %s) is empty",
                    trigger_id,
                    response_detector_id,
                )

            pos_hist = _build_hist(pos_vals, nbins, range_multiplier)
            if pos_hist is not None:
                pos_counts[k], pos_bins[k] = pos_hist
            else:
                log.debug(
                    "positive histogram for element (%s, %s) is empty",
                    trigger_id,
                    response_detector_id,
                )

    parameters = {
        "energy_param": energy_param,
        "positive_param": positive_param,
        "negative_param": negative_param,
        "response_conditions": response_conditions,
        "response_energy_range": list(response_energy_range),
        "positive_baseline": positive_baseline,
        "negative_baseline": negative_baseline,
        "nbins": nbins,
        "range_multiplier": range_multiplier,
    }

    log.info(
        "xtalk column of response %s filled, %s/%s elements",
        response_detector_id,
        int(valid.sum()),
        n_trigger,
    )

    return {
        "response_id": response_detector_id,
        "trigger_ids": np.asarray(trigger_id_list),
        "valid": valid,
        "n_events": n_events,
        "neg_counts": neg_counts,
        "neg_bins": neg_bins,
        "pos_counts": pos_counts,
        "pos_bins": pos_bins,
        "parameters": parameters,
        "processed_at": datetime.now().isoformat(),
    }


def _fit_gaussian_with_fallbacks(
    counts: np.ndarray,
    bins: np.ndarray,
    low_stats_threshold: float,
    y_mask_threshold: float,
    sharp_fit_min_points: int,
) -> tuple[float, float, float, int, int]:
    """Fit a gaussian to one histogram.

    Returns ``(A, mu, sigma, total_counts, status)``, where *status* is one of
    the values of :data:`FIT_STATUS` and the three fit parameters are NaN
    wherever that status says they are not available.
    """
    y = np.asarray(counts, dtype=float)
    total_counts = int(y.sum())

    if total_counts == 0:
        return np.nan, np.nan, np.nan, 0, FIT_STATUS["no_stats"]

    x = pgh.get_bin_centers(bins)

    # too few counts, fallback to histogram arithmetic mean 
    if total_counts < low_stats_threshold:
        mu = float(np.sum(x * y) / total_counts)
        sigma = float(np.sqrt(np.sum(y * (x - mu) ** 2) / total_counts))
        return np.nan, mu, sigma, total_counts, FIT_STATUS["low_stats"]

    # fit the peak rather than the tails
    mask = y > y_mask_threshold * np.max(y)
    if int(mask.sum()) < sharp_fit_min_points:
        # peak too sharp, fallback to no mask
        mask = np.ones_like(y, dtype=bool)
        status = FIT_STATUS["ok_few_points"]
    else:
        status = FIT_STATUS["ok"]

    x_fit = x[mask]
    y_fit = y[mask]
    amplitude_0 = float(np.max(y_fit))
    mu_0 = float(np.average(x_fit, weights=y_fit))
    sigma_0 = float(np.sqrt(np.average((x_fit - mu_0) ** 2, weights=y_fit)))
    if sigma_0 <= 0:
        sigma_0 = float(x[1] - x[0]) if len(x) > 1 else 1.0 # Prevent ZeroDivisionError

    try:
        popt, _ = curve_fit(nb_gauss_amp, x_fit, y_fit, p0=[mu_0, sigma_0, amplitude_0])
    except (RuntimeError, ValueError) as e:
        log.debug("gaussian fit did not converge: %s", e)
        return np.nan, np.nan, np.nan, total_counts, FIT_STATUS["fit_failed"]

    mu, sigma, amplitude = (float(v) for v in popt)
    return amplitude, mu, abs(sigma), total_counts, status


def xtalk_histogram_fitter(
    histogram_data: dict,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Fit a gaussian to every histogram of one cross-talk column.

    *histogram_data* is exactly the dict :func:`xtalk_column` returns.

    Every element is fitted twice, once against the negative and once against
    the positive response, and each fit lands in one of the outcomes of
    :data:`FIT_STATUS`:

    ``ok``
        the fit converged on the bins above ``y_mask_threshold`` of the peak.
    ``ok_few_points``
        that mask left fewer than ``sharp_fit_min_points`` bins, so the fit
        converged on all of them instead.
    ``low_stats``
        fewer than ``low_stats_threshold`` counts, so ``mu`` and ``sigma``
        are the moments of the histogram rather than a fit, and ``A`` is NaN.
    ``fit_failed``
        the fit did not converge; all three parameters are NaN.
    ``no_stats``
        the histogram is empty.
    ``not_filled``
        The fitting is skipped if "valid" is False, leaving FIT_STATUS "not_filled".

    Parameters
    ----------
    histogram_data
        Result dict of :func:`xtalk_column`.
    config
        Fit configuration, read straight off the top level.  Recognised
        keys, all optional:

        ``low_stats_threshold``
            Counts below which the moments replace the fit.  Default 100.
        ``y_mask_threshold``
            Bins below this fraction of the tallest one are dropped before
            fitting.  Default 0.05.
        ``sharp_fit_min_points``
            Bins that mask must leave for the fit to use it.  Default 5.
    debug_mode
        If True, re-raise instead of recording an element as ``fit_failed``.

    Returns
    -------
    dict
        Every key of *histogram_data*, unchanged, plus ``{neg,pos}_A``,
        ``{neg,pos}_mu``, ``{neg,pos}_sigma``, ``{neg,pos}_total_counts``,
        ``{neg,pos}_status`` and ``{neg,pos}_success``, all ``(N,)``, and
        ``fit_parameters``, ``fit_status_codes`` and ``fitted_at``.
        ``total_counts`` is the integral of the histogram, which is at most
        ``n_events`` -- events outside the histogram range are not in it.
    """

    config = config or {}
    low_stats_threshold = float(
        config.get("low_stats_threshold", DEFAULT_LOW_STATS_THRESHOLD)
    )
    y_mask_threshold = float(config.get("y_mask_threshold", DEFAULT_Y_MASK_THRESHOLD))
    sharp_fit_min_points = int(
        config.get("sharp_fit_min_points", DEFAULT_SHARP_FIT_MIN_POINTS)
    )

    response_id = histogram_data["response_id"]
    trigger_ids = np.asarray(histogram_data["trigger_ids"])
    valid = np.asarray(histogram_data["valid"], dtype=bool)
    n_trigger = len(trigger_ids)

    result = dict(histogram_data)

    for polarity in ("neg", "pos"):
        counts = np.asarray(histogram_data[f"{polarity}_counts"])
        bins = np.asarray(histogram_data[f"{polarity}_bins"])

        amplitude = np.full(n_trigger, np.nan)
        mu = np.full(n_trigger, np.nan)
        sigma = np.full(n_trigger, np.nan)
        total_counts = np.zeros(n_trigger, dtype=np.int64)
        status = np.full(n_trigger, FIT_STATUS["not_filled"], dtype=np.int8)

        for k in range(n_trigger):
            if not valid[k]:
                continue

            try:
                fit = _fit_gaussian_with_fallbacks(
                    counts[k],
                    bins[k],
                    low_stats_threshold,
                    y_mask_threshold,
                    sharp_fit_min_points,
                )
            except Exception as e:
                if debug_mode:
                    raise
                log.error(
                    "%s fit of element (%s, %s) failed: %s",
                    polarity,
                    trigger_ids[k],
                    response_id,
                    e,
                )
                status[k] = FIT_STATUS["fit_failed"]
                continue

            amplitude[k], mu[k], sigma[k], total_counts[k], status[k] = fit

        result[f"{polarity}_A"] = amplitude
        result[f"{polarity}_mu"] = mu
        result[f"{polarity}_sigma"] = sigma
        result[f"{polarity}_total_counts"] = total_counts
        result[f"{polarity}_status"] = status
        result[f"{polarity}_success"] = np.isin(status, FIT_STATUS_SUCCESS)

    result["fit_parameters"] = {
        "low_stats_threshold": low_stats_threshold,
        "y_mask_threshold": y_mask_threshold,
        "sharp_fit_min_points": sharp_fit_min_points,
    }
    result["fit_status_codes"] = FIT_STATUS
    result["fitted_at"] = datetime.now().isoformat()

    log.info(
        "xtalk fits of response %s: %s negative and %s positive of %s elements "
        "converged",
        response_id,
        int(result["neg_success"].sum()),
        int(result["pos_success"].sum()),
        n_trigger,
    )

    return result


def build_xtalk_matrix(
    fitted_columns: dict,
    config: dict | None = None,
) -> lgdo.Table:
    """Assemble the fitted columns of a cross-talk matrix into the matrix.

    Element ``[j1, j2]`` of a matrix:  ``rawid_index[j1]`` represents triggered
    detector, while ``rawid_index[j2]`` represents the responding detector.

    A column fixes the responding detector and runs over the triggers, so it
    lands in column ``j2`` of the matrix, and the detector order of the matrix
    is the trigger order every column shares.

    Parameters
    ----------
    fitted_columns
        Result dicts of :func:`xtalk_histogram_fitter`, keyed by response
        channel id.
    config
        Recognised keys, all optional:

        ``max_status``
            Highest :data:`FIT_STATUS` code to accept into the matrix. Default
            ``FIT_STATUS["low_stats"]``.

    Returns
    -------
    lgdo.Table
        A table with the following fields. Values are sorted in the order of
        ``rawid_index``:

        ``rawid_index`` ``(N,)``
            Detector ids: ``rawid_index[j]`` is the detector at row and
            column ``j`` of every matrix.
        ``xtalk_matrix_negative``, ``xtalk_matrix_positive`` ``(N, N)``
            The fitted peak positions, as **fractions**, which is the unit
            the production files store.
        ``..._sigma`` ``(N, N)``
            The width of each of those fits, also as fractions.
        ``..._status`` ``(N, N)``
            The :data:`FIT_STATUS` code of each element, meaning explained in
            :func:`xtalk_histogram_fitter`.
    """
    config = config or {}
    max_status = int(config.get("max_status", FIT_STATUS["low_stats"]))

    columns = {
        int(response_id): column for response_id, column in fitted_columns.items()
    }
    rawids = None

    # Check whether the trigger id lists are identical across all columns
    for response_id, column in columns.items():
        trigger_ids = np.asarray(column["trigger_ids"], dtype=np.int64)
        if rawids is None:
            rawids = trigger_ids
        elif not np.array_equal(rawids, trigger_ids):
            msg = (
                f"column {response_id} covers different detectors, or covers "
                f"them in a different order, than the columns before it"
            )
            raise ValueError(msg)

        if int(column["response_id"]) != response_id:
            msg = (
                f"column filed under response {response_id} reports "
                f"response_id {column['response_id']}"
            )
            raise ValueError(msg)

    index_of = {int(rawid): j for j, rawid in enumerate(rawids)}
    unknown = sorted(set(columns) - set(index_of))
    if unknown:
        msg = (
            f"columns {unknown} respond on detectors that are not among the "
            f"triggers, so they have no column in the matrix"
        )
        raise ValueError(msg)

    n_detectors = len(rawids)
    shape = (n_detectors, n_detectors)
    mu = {p: np.full(shape, np.nan) for p in ("neg", "pos")}
    sigma = {p: np.full(shape, np.nan) for p in ("neg", "pos")}
    status = {
        p: np.full(shape, FIT_STATUS["not_filled"], dtype=np.int8)
        for p in ("neg", "pos")
    }

    for response_id, column in columns.items():
        col = index_of[response_id]
        for polarity in ("neg", "pos"):
            column_status = np.asarray(column[f"{polarity}_status"], dtype=np.int8)
            accepted = column_status <= max_status

            status[polarity][:, col] = column_status
            mu[polarity][:, col] = np.where(
                accepted, np.asarray(column[f"{polarity}_mu"], dtype=float), np.nan
            )
            sigma[polarity][:, col] = np.where(
                accepted, np.asarray(column[f"{polarity}_sigma"], dtype=float), np.nan
            )

    col_dict = {"rawid_index": lgdo.Array(np.asarray(rawids, dtype=np.int64))}
    for polarity in ("neg", "pos"):
        field = XTC_LH5_FIELD[polarity]
        col_dict[field] = lgdo.Array(mu[polarity])
        col_dict[f"{field}_sigma"] = lgdo.Array(sigma[polarity])
        col_dict[f"{field}_status"] = lgdo.Array(status[polarity])

    missing = n_detectors - len(columns)
    if missing:
        log.warning(
            "%s of %s detectors have no fitted column, their columns stay NaN",
            missing,
            n_detectors,
        )

    return lgdo.Table(
        col_dict=col_dict, attrs={"fit_status_codes": json.dumps(FIT_STATUS)}
    )


def plot_xtalk_matrix(
    matrix: lgdo.Table,
    polarity: str = "neg",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
    title: str | None = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Draw a heatmap of one polarity of a cross-talk matrix.

    Parameters
    ----------
    matrix
        Table :func:`build_xtalk_matrix` returned, or one read back from an
        xtc lh5 file with :func:`lh5.read`.  Only the polarity's own column
        is used, so a production file that carries nothing but the two
        matrices plots as well as one this module wrote.
    polarity
        ``"neg"`` or ``"pos"``, naming the column through
        :data:`XTC_LH5_FIELD`.
    vmin, vmax
        Colour-scale limits, as fractions.  ``None`` takes the polarity's
        entry in :data:`XTC_PLOT_RANGE`.
    cmap
        Colormap, defaulting to reversed jet as in the original analysis.
    title
        Figure title.  ``None`` names the polarity.
    figsize
        Figure size, in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, for the caller to show or to ``savefig``.
    """
    if polarity not in XTC_LH5_FIELD:
        msg = f"polarity must be one of {tuple(XTC_LH5_FIELD)}, got {polarity!r}"
        raise ValueError(msg)

    values = matrix[XTC_LH5_FIELD[polarity]]
    values = np.asarray(values.nda if hasattr(values, "nda") else values)

    default_vmin, default_vmax = XTC_PLOT_RANGE[polarity]

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        values,
        origin="lower",
        vmin=default_vmin if vmin is None else vmin,
        vmax=default_vmax if vmax is None else vmax,
        cmap=plt.cm.jet_r if cmap is None else cmap,
    )
    fig.colorbar(image, ax=ax, label="Cross-talk (fraction)")
    ax.set_xlabel("Response channel index")
    ax.set_ylabel("Trigger channel index")
    ax.set_title(title if title is not None else f"{polarity} cross-talk matrix")
    fig.tight_layout()

    return fig
