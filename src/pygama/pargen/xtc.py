"""
This module provides routines for measuring cross-talk (XTC) between
germanium channels and for building the resulting cross-talk matrix.

The four main functions, in order of execution, are:
prepare_detector, xtalk_column, xtalk_histogram_fitter, and build_xtalk_matrix.

All of the file reading happens in the first of them, once per channel and
independently of every other channel.  It selects what that detector
contributes in both of its roles: the events it triggered on and the
amplitude each of them fired with, and the baselines and per-event amplitudes
it shows when it responds instead.  The three functions after it never open a
file, so an N x N matrix costs N reads rather than one per detector pair, and
the reads all happen in the same step.

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
    mask = np.isfinite(energies) 

    for flag, value in (conditions or {}).items():
        mask &= table[flag].nda == value

    if energy_range is not None:
        emin, emax = energy_range
        mask &= (energies >= emin) & (energies <= emax)

    return mask


def prepare_detector(
    hit_files: str | list,
    dsp_files: str | list,
    chn_id: str | int,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Read one channel once and select everything the matrix needs from it.

    A detector enters the cross-talk matrix in two roles, and this measures
    both of them in a single pass:

    *as a trigger*
        the events in which it fired, as global entry numbers, together with
        the DSP amplitude each of them fired with -- the denominator of the
        cross-talk ratio.
    *as a response*
        its baselines, the amplitudes it recorded in every event, and which
        of those events it may be measured in at all.

    Nothing here depends on any other channel, so the N calls that cover a
    whole array are independent of each other, and no later step has to open
    a hit or dsp file again.  The three selections are independent of each
    other too: each reports its own flag, and a failure to read the channel
    at all fails every one of them.

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
        ``response_conditions``
            Mapping of hit-tier flag field to the value it must equal for the
            channel to be measurable as a response.  Default ``{}``.
        ``response_energy_range``
            ``(emin, emax)`` on ``energy_param`` selecting the events in which
            this channel did *not* see a real hit -- an event in which it did
            is a multiplicity event, not cross-talk.  Default
            ``(-99999, 100)``.
        ``energy_param``
            Hit-tier field all three selections are applied to.  Default
            ``"cuspEmax_ctc_cal"``.
        ``positive_param``, ``negative_param``
            DSP-tier fields averaged to give the baselines, and histogrammed
            when this channel responds.  Default ``"trapTmax"`` and
            ``"trapTmin"``.
        ``trigger_param``
            DSP-tier field giving the amplitude this channel triggered with.
            Default ``"trapTmax"``.
    debug_mode
        If True, re-raise instead of falling back to a null result.

    Returns
    -------
    dict
        ``detector_id``, ``n_rows``, ``processed_at``, ``parameters``, and
        ``read_success``; then, for the response role, ``positive_baseline``
        and ``negative_baseline``, ``response_keep`` ``(n_rows,)`` bool and
        ``positive_response``/``negative_response`` ``(n_rows,)``, guarded by
        ``baseline_success``; and for the trigger role, ``trigger_idxs`` and
        ``trigger_amplitudes``, guarded by ``trigger_success``.

        The two baselines are either both finite floats or both ``None``;
        there is no third outcome, and in particular never a NaN, an infinity
        or one of each.  The three response arrays are empty when the channel
        could not be read, and the two trigger arrays are empty when its
        trigger selection found nothing.

    Notes
    -----
    ``trigger_idxs`` are positions in *hit_files* read as one concatenated
    table, and they are used to index the response arrays of a *different*
    channel, so both channels must cover the same events in the same order.
    That is the same assumption the pair of files already carries.
    """
    config = config or {}
    baseline_conditions = dict(
        config.get("baseline_conditions", DEFAULT_BASELINE_CONDITIONS)
    )
    trigger_conditions = dict(
        config.get("trigger_conditions", DEFAULT_TRIGGER_CONDITIONS)
    )
    response_conditions = dict(config.get("response_conditions", {}))
    trigger_energy_range = tuple(
        config.get("trigger_energy_range", DEFAULT_TRIGGER_ENERGY_RANGE)
    )
    response_energy_range = tuple(
        config.get("response_energy_range", DEFAULT_RESPONSE_ENERGY_RANGE)
    )
    energy_param = config.get("energy_param", DEFAULT_ENERGY_PARAM)
    positive_param = config.get("positive_param", DEFAULT_POSITIVE_PARAM)
    negative_param = config.get("negative_param", DEFAULT_NEGATIVE_PARAM)
    trigger_param = config.get("trigger_param", DEFAULT_TRIGGER_PARAM)

    dsp_fields = list(dict.fromkeys([positive_param, negative_param, trigger_param]))
    hit_fields = list(
        dict.fromkeys(
            [
                energy_param,
                *baseline_conditions,
                *trigger_conditions,
                *response_conditions,
            ]
        )
    )

    read_success = True
    baseline_success = True
    trigger_success = True
    positive_baseline = None
    negative_baseline = None
    n_rows = 0
    response_keep = np.empty(0, dtype=bool)
    positive_response = np.empty(0, dtype=np.float32)
    negative_response = np.empty(0, dtype=np.float32)
    trigger_idxs = np.empty(0, dtype=np.int64)
    trigger_amplitudes = np.empty(0, dtype=np.float32)

    baseline_mask = None
    trigger_mask = None

    try:
        # the hit tier is only ever asked which events to take, so the masks
        # are built and the tier dropped before the far larger dsp tier is
        # opened, rather than holding both at once
        hit_table = lh5.read(f"ch{chn_id}/hit/", hit_files, field_mask=hit_fields)
        n_rows = len(hit_table[energy_param].nda)

        baseline_mask = _selection_mask(hit_table, energy_param, baseline_conditions)
        trigger_mask = _selection_mask(
            hit_table, energy_param, trigger_conditions, trigger_energy_range
        )
        response_keep = _selection_mask(
            hit_table, energy_param, response_conditions, response_energy_range
        )
        del hit_table

        dsp_table = lh5.read(f"ch{chn_id}/dsp/", dsp_files, field_mask=dsp_fields)
        positive_response = dsp_table[positive_param].nda
        negative_response = dsp_table[negative_param].nda
        trigger_all = dsp_table[trigger_param].nda

        if len(positive_response) != n_rows:
            msg = (
                f"the hit tier holds {n_rows} events and the dsp tier "
                f"{len(positive_response)}, so they do not describe the same "
                f"events"
            )
            raise RuntimeError(msg)
    except Exception as e:
        if debug_mode:
            raise
        log.error(
            "reading channel %s failed, none of its selections were made: %s: %s",
            chn_id,
            type(e).__name__,
            e,
        )
        read_success = False
        baseline_success = False
        trigger_success = False
        n_rows = 0
        response_keep = np.empty(0, dtype=bool)
        positive_response = np.empty(0, dtype=np.float32)
        negative_response = np.empty(0, dtype=np.float32)

    if read_success:
        try:
            positive_selected = positive_response[baseline_mask]
            negative_selected = negative_response[baseline_mask]
            positive_selected = positive_selected[np.isfinite(positive_selected)]
            negative_selected = negative_selected[np.isfinite(negative_selected)]
            if len(positive_selected) == 0 or len(negative_selected) == 0:
                msg = "no events passed the baseline selection"
                raise RuntimeError(msg)

            positive_baseline = float(np.mean(positive_selected))
            negative_baseline = float(np.mean(negative_selected))
            if not np.isfinite(positive_baseline) or not np.isfinite(negative_baseline):
                msg = (
                    f"the baseline average came out as "
                    f"({positive_baseline}, {negative_baseline}), which is not "
                    f"a pair of numbers to subtract"
                )
                raise RuntimeError(msg)
        except Exception as e:
            if debug_mode:
                raise
            log.error("baseline preparation failed for channel %s: %s", chn_id, e)
            positive_baseline = None
            negative_baseline = None
            baseline_success = False

        try:
            # a position in the concatenated table is a global entry number,
            # which is what indexes another channel's response arrays
            idxs = np.flatnonzero(trigger_mask).astype(np.int64)
            amplitudes = trigger_all[trigger_mask]

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
            trigger_amplitudes = np.empty(0, dtype=np.float32)
            trigger_success = False

    log.info(
        "channel %s: %s events, baseline %s, %s of them measurable as a "
        "response, %s trigger events",
        chn_id,
        n_rows,
        "measured" if baseline_success else "not measured",
        int(response_keep.sum()),
        len(trigger_idxs),
    )

    return {
        "detector_id": chn_id,
        "n_rows": n_rows,
        "positive_baseline": positive_baseline,
        "negative_baseline": negative_baseline,
        "response_keep": response_keep,
        "positive_response": positive_response,
        "negative_response": negative_response,
        "trigger_idxs": trigger_idxs,
        "trigger_amplitudes": trigger_amplitudes,
        "read_success": read_success,
        "baseline_success": baseline_success,
        "trigger_success": trigger_success,
        "processed_at": datetime.now().isoformat(),
        "parameters": {
            "baseline_conditions": baseline_conditions,
            "trigger_conditions": trigger_conditions,
            "response_conditions": response_conditions,
            "trigger_energy_range": list(trigger_energy_range),
            "response_energy_range": list(response_energy_range),
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
    response_detector_id: str | int,
    response: dict,
    triggers: dict,
    config: dict | None = None,
    debug_mode: bool = False,
) -> dict:
    """Fill the histograms for one column of the cross-talk matrix.

    A column fixes the *responding* detector: it holds one element per trigger
    detector, each the distribution of the energy *response_detector_id* picked
    up while that trigger fired.

    Nothing is read here.  Both sides arrive as :func:`prepare_detector`
    results, so an element costs one index into arrays already in memory, and
    a whole N x N matrix costs the N reads that produced them.

    Elements skipped are recorded with ``valid = False`` and an empty
    histogram.  This happens when the trigger channel is the response itself,
    or when the trigger channel selected no usable events.  If the response
    channel has no usable baseline the whole column is skipped.

    Parameters
    ----------
    response_detector_id
        Channel id of the responding detector, without the ``ch`` prefix.
    response
        The :func:`prepare_detector` result of the responding channel.  Read
        for ``positive_baseline`` and ``negative_baseline``, ``response_keep``
        and ``positive_response``/``negative_response``.  A ``None`` baseline,
        which is what that function returns when it could not measure the
        channel, skips the whole column: there is nothing to subtract, so no
        element of it can be filled.
    triggers
        The :func:`prepare_detector` results of the triggering channels,
        collected by channel id.  The keys are the trigger detectors the
        column covers, in the order its elements come out in.  Only
        ``trigger_idxs`` and ``trigger_amplitudes`` are read, so a caller that
        has dropped the bulky response arrays may pass what is left:
        {chn_id: {"trigger_idxs": array, "trigger_amplitudes": array}, ...}

        for example:
            {
                1104000: {"trigger_idxs": [3, 17], "trigger_amplitudes": [2.1, 3.4]},
                1104001: {"trigger_idxs": [], "trigger_amplitudes": []},
                ...
            }
    config
        Histogram configuration.  Recognised keys, both optional:

        ``nbins``
            Bins per histogram.  Default 700.
        ``range_multiplier``
            Histogram half-width in standard deviations about the mean.
            Default 3.

        The event selection is not configured here: it was already applied by
        :func:`prepare_detector`, and the configuration it used is recorded in
        its own result.
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
    positive_baseline = response.get("positive_baseline")
    negative_baseline = response.get("negative_baseline")

    try:
        if positive_baseline is None or negative_baseline is None:
            msg = f"response channel {response_detector_id} has no usable baseline"
            raise RuntimeError(msg)

        # a python float stays weak against a float32 array, so the ratio is
        # taken in the precision the dsp tier stored, however the baseline
        # itself was carried here
        positive_baseline = float(positive_baseline)
        negative_baseline = float(negative_baseline)

        response_keep = np.asarray(response["response_keep"], dtype=bool)
        positive_response = np.asarray(response["positive_response"])
        negative_response = np.asarray(response["negative_response"])

        n_total = len(response_keep)
        if n_total == 0:
            msg = f"response channel {response_detector_id} holds no events"
            raise RuntimeError(msg)
        if len(positive_response) != n_total or len(negative_response) != n_total:
            msg = (
                f"response channel {response_detector_id} has {n_total} selected "
                f"events but {len(positive_response)} positive and "
                f"{len(negative_response)} negative amplitudes"
            )
            raise RuntimeError(msg)

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
    # If the response is unusable, skip the whole loop to saving an empty column.
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
