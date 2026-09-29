# /// script
# dependencies = [
#     "dr-datacube",
#     "matplotlib",
#     "marimo",
#     "polars==1.43.2",
#     "npc_session",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///
from __future__ import annotations

import logging
import pathlib
import typing

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import npc_session
import numpy as np
import numpy.typing as npt
import polars as pl

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["pdf.fonttype"] = 42


class NoSpikesInTrialsError(ValueError):
    pass


def makePsth(
    spikeTimes: npt.NDArray[np.floating],
    startTimes: npt.NDArray[np.floating],
    baselineDur: float = 0.1,
    responseDur: float = 1.0,
    binSize: float = 0.001,
    convolution_kernel: float = 0.01,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Compute a trial-averaged, boxcar-smoothed PSTH using NumPy."""
    spikeTimes = np.asarray(spikeTimes).flatten()
    startTimes = np.asarray(startTimes) - (baselineDur + convolution_kernel / 2)
    windowDur = responseDur + baselineDur + convolution_kernel
    bins = np.arange(0, baselineDur + responseDur + binSize, binSize)
    convkernel = np.ones(int(convolution_kernel / binSize))
    counts = np.zeros(bins.size - 1)
    for startTime in startTimes:
        startInd = np.searchsorted(spikeTimes, startTime)
        endInd = np.searchsorted(spikeTimes, startTime + windowDur)
        counts += np.histogram(spikeTimes[startInd:endInd] - startTime, bins)[0]
    counts = counts / startTimes.size
    counts = np.convolve(counts, convkernel) / (binSize * convkernel.size)
    return (
        counts[convkernel.size - 1 : -convkernel.size],
        bins[: -convkernel.size - 1] - baselineDur,
    )


def plot(
    unit_id: str | typing.Sequence[str],
    unit_spike_times: npt.NDArray[np.floating]
    | typing.Sequence[npt.NDArray[np.floating] | None]
    | None = None,
    stim_names=("vis1", "sound1", "vis2", "sound2"),
    with_instruction_trial_whitespace: bool = False,
    max_psth_spike_rate: float | typing.Sequence[float] | None = 60,  # Hz
    use_session_obj: bool = False,
    session=None,
    xlim_0=-1.0,  # seconds before stim onset
    xlim_1=2.0,  # seconds after stim onset
    display_block_index=False,
    plot_blockwise_psths=False,
    show_epoch_schematic: bool = True,
    fig_width_per_unit: float = 1.5,  # inches, width allocated to each unit's column
    fig_height: float | None = None,  # inches; if None, derived from panel count
    figscale: float = 1.0,  # multiplies both width and height
) -> plt.Figure:
    """Raster + PSTH figure for one or more units.

    ``unit_id`` may be a single unit id (str) or a sequence of unit ids. When a
    sequence is passed, each unit is drawn in its own column using exactly the
    same layout as the single-unit plot.

    ``max_psth_spike_rate`` controls the Hz that maps to the full height of a
    PSTH panel. Pass ``None`` to auto-scale each unit independently so its PSTH
    traces just fit within the panels, or pass a sequence to set it per unit.

    ``show_epoch_schematic`` adds the Baseline / Stim / Resp / ITI bar above the
    rasters. ``fig_width_per_unit``, ``fig_height`` and ``figscale`` control the
    figure size.
    """
    # normalise unit_id to a list while remembering whether input was scalar
    if isinstance(unit_id, str) or not isinstance(unit_id, typing.Sequence):
        unit_ids = [unit_id]
        single_unit_input = True
    else:
        unit_ids = list(unit_id)
        single_unit_input = False
    n_units = len(unit_ids)

    # normalise per-unit spike times
    if unit_spike_times is None or isinstance(unit_spike_times, np.ndarray):
        spike_times_list: list = [unit_spike_times] * n_units
    else:
        spike_times_list = list(unit_spike_times)
        if len(spike_times_list) != n_units:
            raise ValueError(
                f"unit_spike_times length {len(spike_times_list)} != number of units {n_units}"
            )

    # normalise per-unit max firing rate (None -> auto-scale per unit)
    if max_psth_spike_rate is None or isinstance(max_psth_spike_rate, (int, float)):
        max_rate_list = [max_psth_spike_rate] * n_units
    else:
        max_rate_list = list(max_psth_spike_rate)
        if len(max_rate_list) != n_units:
            raise ValueError(
                f"max_psth_spike_rate length {len(max_rate_list)} != number of units {n_units}"
            )

    n_psth = 2  # two stacked PSTHs: vis target (top), aud target (bottom)
    default_height = (6 + n_psth) * 0.7
    height = default_height if fig_height is None else fig_height
    width = fig_width_per_unit * n_units
    fig, axes = plt.subplots(
        1, n_units, figsize=(width * figscale, height * figscale), sharey=True
    )
    if n_units == 1:
        axes = [axes]

    locations: list[str] = []
    for idx, (ax, this_unit_id, this_spike_times, this_max_rate) in enumerate(
        zip(axes, unit_ids, spike_times_list, max_rate_list)
    ):
        location = _plot_unit(
            ax=ax,
            unit_id=this_unit_id,
            unit_spike_times=this_spike_times,
            stim_names=stim_names,
            with_instruction_trial_whitespace=with_instruction_trial_whitespace,
            max_psth_spike_rate=this_max_rate,
            use_session_obj=use_session_obj,
            session=session,
            xlim_0=xlim_0,
            xlim_1=xlim_1,
            display_block_index=display_block_index,
            plot_blockwise_psths=plot_blockwise_psths,
            show_epoch_schematic=show_epoch_schematic,
            n_psth=n_psth,
            is_first_column=(idx == 0),
            single_unit=single_unit_input,
        )
        locations.append(location)

    if single_unit_input:
        fig.suptitle(locations[0], fontsize=8)
    else:
        # one title per column (unit) instead of a single figure title.
        # keep the unit number on the same line as the structure id so the title
        # stays a single row and leaves vertical room for the stim schematic.
        for ax, this_unit_id, location in zip(axes, unit_ids, locations):
            suffix = str(this_unit_id).split("_")[-1]
            ax.set_title(f"{location}  {suffix}", fontsize=8)
    fig.set_dpi(300)
    return fig


def _plot_unit(
    ax: plt.Axes,
    unit_id: str,
    unit_spike_times: npt.NDArray[np.floating] | None,
    stim_names,
    with_instruction_trial_whitespace: bool,
    max_psth_spike_rate: float | None,
    use_session_obj: bool,
    session,
    xlim_0: float,
    xlim_1: float,
    display_block_index: bool,
    plot_blockwise_psths: bool,
    show_epoch_schematic: bool,
    n_psth: int,
    is_first_column: bool,
    single_unit: bool = True,
) -> str:
    """Draw a single unit's raster + PSTH into ``ax`` and return its location label."""
    # in case unit_id is an npc_sessions object
    try:
        session_id = npc_session.SessionRecord(
            unit_id.id
        ).id  # in case unit_id is an npc_sessions object
    except (AttributeError, TypeError):
        session_id = npc_session.SessionRecord(unit_id).id

    if len(session_id.split("_")) == 3:
        session_id = "_".join(session_id.split("_")[:2])

    session_obj = session
    if use_session_obj or session_obj is not None:
        if session_obj is not None:
            obj = session_obj
        else:
            import npc_sessions

            obj = npc_sessions.Session(session_id)
        trials = pl.DataFrame(obj.trials[:])
        try:
            units = pl.DataFrame(
                obj.units[:][["spike_times", "location", "structure", "unit_id"]]
            )
        except KeyError:
            units = pl.DataFrame(obj.units[:][["spike_times", "unit_id"]])
        unit = units.filter(pl.col("unit_id") == unit_id)
        if unit_spike_times is None:
            unit_spike_times = unit["spike_times"].to_numpy()[0]
        performance: pl.DataFrame = pl.DataFrame(obj.intervals["performance"][:])
    else:
        units_all_sessions = utils.get_df("units")
        trials_all_sessions = utils.get_df("trials")

        trials = trials_all_sessions.filter(pl.col("session_id") == session_id)

        unit = units_all_sessions.filter(pl.col("unit_id") == unit_id)

        #! session id is without idx for spike times
        spike_times_session_id = "_".join(unit_id.split("_")[:2])
        if unit_spike_times is None:
            unit_spike_times: npt.NDArray[np.floating] = utils.get_spike_times(unit_id)[unit_id]
        performance_all_sessions = utils.get_df("performance")
        performance = performance_all_sessions.filter(
            pl.col("session_id") == session_id
        )

    pad_start = -xlim_0 + 0.5  # seconds before stim onset
    if trials.is_empty():
        raise ValueError(f"No trials found for {session_id}")
    if not unit_spike_times.size:
        raise NoSpikesInTrialsError(f"No spike times found for {unit_id}")
    modality_to_rewarded_stim = {"aud": "sound1", "vis": "vis1"}
    # add spikes to trials:
    spike_times_by_trial = tuple(
        (
            unit_spike_times[slice(start, stop)]
            if 0 <= start < stop <= len(unit_spike_times)
            else []
        )
        for start, stop in np.searchsorted(
            unit_spike_times,
            trials.select(pl.col("start_time") - pad_start, "stop_time"),
        )
    )
    if not spike_times_by_trial or not any(
        np.array(a).any() for a in spike_times_by_trial
    ):
        raise NoSpikesInTrialsError(
            f"No spike times found matching trial times {unit} - either no task presented or major timing issue"
        )
    trials = (
        trials.with_columns(
            pl.Series(
                name="spike_times",
                values=spike_times_by_trial,
                dtype=pl.List(pl.Float64),
            ),  # doesn't handle empty entries well without explicit dtype
        )
        .with_row_index()
        .explode("spike_times")
        .with_columns(
            stim_centered_spike_times=(
                pl.col("spike_times")
                - pl.col("stim_start_time").alias("stim_centered_spike_times")
            )
        )
        .group_by(
            pl.all().exclude("spike_times", "stim_centered_spike_times"),
            maintain_order=True,
        )
        .all()
        .filter(
            pl.col("stim_name").is_in(stim_names),
            #! filter out autoreward trials triggered by 10 misses:
            # (pl.col('is_reward_scheduled').eq(True) & (pl.col('trial_index_in_block') < 5)) | pl.col('is_reward_scheduled').eq(False),
        )
    )

    line_params = {
        "color": "grey",
        "lw": 0.3,
    }
    response_window_start_time = 0.1  # np.median(np.diff(trials.select('stim_start_time', 'response_window_start_time')))
    response_window_stop_time = 1  # np.median(np.diff(trials.select('stim_start_time', 'response_window_stop_time')))
    aud_block_color = "orange"
    vis_psth_color = "#0000ff"  # blue (vis context), sampled from raster.png
    aud_psth_color = "#ec008c"  # magenta (aud context), sampled from raster.png
    add_psth = True
    nominal_rows_per_block = 12
    block_height_on_page = (
        (6 + n_psth) * nominal_rows_per_block / trials.n_unique("block_index")
    )  # height of each row will be this value / len(block_df)
    axes = [ax]
    last_ypos: list[float] = []
    for ax in axes:
        ax: plt.Axes

        stim_trials = trials
        idx_in_block = 0
        for _idx, trial in enumerate(stim_trials.iter_rows(named=True)):

            num_instructed_trials = max(
                len(
                    trials.filter(  # check original trials, not modified ones with dummy instruction trials
                        pl.col("block_index") == trial["block_index"],
                        pl.col(f"is_{c}_rewarded"),
                        pl.col("is_reward_scheduled"),
                        pl.col("trial_index_in_block") < 14,
                    )
                )
                for c in ("aud", "vis")
            )

            is_vis_block: bool = "vis" in trial["rewarded_modality"]
            is_rewarded_stim: bool = True # trials for all stims are grouped together, so this is always True

            if trial["trial_index_in_block"] == 0:
                idx_in_block = 0
                block_df = stim_trials.filter(
                    pl.col("block_index") == trial["block_index"]
                ).sort('trial_index')
                ypositions = (
                    np.linspace(0, block_height_on_page, len(block_df), endpoint=False)
                    + trial["block_index"] * block_height_on_page
                )
                halfline = 0.5 * np.diff(ypositions).mean()
                ypositions = ypositions + halfline
            ypos = ypositions[idx_in_block]

            idx_in_block += 1  # updated for next trial - don't use after this point

            if trial["trial_index_in_block"] == 0:
                if is_rewarded_stim:
                    assert num_instructed_trials == (
                        x := len(
                            block_df.filter(
                                (pl.col("trial_index_in_block") < 10)
                                & (pl.col("is_reward_scheduled"))
                            )
                        )
                    ), f"{x} != {num_instructed_trials=}"

                if is_first_column:
                    # block label
                    rotation = 0
                    ax.text(
                        x=xlim_0 - 0.6,
                        y=ypositions[0] + block_height_on_page // 2,
                        s=str(trial["block_index"] + 1) if display_block_index else trial["rewarded_modality"],
                        fontsize=8,
                        ha="center",
                        va="center",
                        color=vis_psth_color if is_vis_block else aud_psth_color,
                        rotation=rotation,
                    )

                # # block switch horizontal lines
                # if trial["block_index"] > 0:
                #     ax.axhline(
                #         y=ypos - halfline,
                #         **line_params,
                #         zorder=99,
                #     )

                if is_rewarded_stim:
                    # autoreward trials green patch
                    autoreward_patch_params = {
                        "color": 'slateblue',
                        "alpha": 0.5,
                        "lw": 0,
                        "zorder": -1,
                    }
                    ax.axhspan(
                        ymin=max(ypos, 0) - halfline,
                        ymax=ypositions[num_instructed_trials - 1] + halfline,
                        **autoreward_patch_params,
                    )

                if trial["is_vis_rewarded"] and len(block_df) > num_instructed_trials:
                    pass
                    # vis block grey patch
                    # ax.axhspan(
                    #     ymin=(
                    #         ypositions[num_instructed_trials] - halfline
                    #         if is_rewarded_stim or with_instruction_trial_whitespace
                    #         else ypositions[0] - halfline
                    #     ),
                    #     ymax=ypositions[-1] + halfline,
                    #     color=[0.95] * 3,
                    #     lw=0,
                    #     zorder=-1,
                    # )

                # response window cyan patch
                rect = patches.Rectangle(
                    xy=(
                        response_window_start_time,
                        (
                            y0 := (
                                ypos
                                if is_rewarded_stim
                                or not with_instruction_trial_whitespace
                                else ypositions[
                                    min(num_instructed_trials, len(block_df) - 1)
                                ]
                            )
                            - halfline
                        ),
                    ),
                    width=response_window_stop_time - response_window_start_time,
                    height=(ypositions[-1] + halfline) - y0,
                    linewidth=0,
                    edgecolor="none",
                    facecolor=[0.85, 0.95, 1, 0.5],
                    zorder=20,
                )
                # ax.add_patch(rect)

            # green patch for instruction trials triggered after 10 consecutive misses
            if trial["is_reward_scheduled"] and trial["trial_index_in_block"] > 10:
                ax.axhspan(ypos - halfline, ypos + halfline, **autoreward_patch_params)

            # spikes
            trial_spike_times = np.array(trial["stim_centered_spike_times"])
            eventplot_params = {
                "lineoffsets": ypos,
                "linewidths": 0.2,
                "linelengths": 0.4,
                "color": [0] * 3,
                "zorder": 99,
            }
            if trial_spike_times.size == 1 and trial_spike_times[0] is None:
                pass
            else:
                ax.eventplot(positions=trial_spike_times, **eventplot_params)

            # times of interest not plot

        last_ypos.append(ypos)
    # format axes and add PSTH
    for ax in axes:
        ax: plt.Axes
        if add_psth:
            bin_size_s = 25 / 1000
            ypad = 5

            # top panel: vis target (vis1); bottom panel: aud target (sound1)
            psth_panels = (("vis1", "VIS+"), ("sound1", "AUD+"))

            def compute_panel_traces(target_stim):
                # PSTH aligned to a single target stimulus, split by context:
                # blue = vis-rewarded blocks, magenta = aud-rewarded blocks.
                # Faint traces are per-block; bold trace is the across-block mean.
                # Returns a list of (hist, bin_edges, color, is_mean) tuples so the
                # peak rate can be measured before anything is drawn (for auto-scaling).
                traces: list = []
                for context_modality, color in (
                    ("vis", vis_psth_color),
                    ("aud", aud_psth_color),
                ):
                    hist_results = []
                    bin_edges = None
                    for _, block_trials in trials.group_by("block_index"):
                        df = block_trials.filter(
                            pl.col(f"is_{context_modality}_rewarded"),
                            pl.col("stim_name") == target_stim,
                        )
                        if df.is_empty():
                            continue
                        hist, bin_edges = makePsth(
                            spikeTimes=np.sort(unit_spike_times),
                            startTimes=np.array(df["stim_start_time"]),
                            baselineDur=pad_start,
                            responseDur=xlim_1,
                            binSize=bin_size_s,
                            convolution_kernel=bin_size_s,
                        )
                        hist_results.append(hist)
                        if plot_blockwise_psths:
                            traces.append((hist, bin_edges, color, False))
                    if not hist_results:
                        continue
                    traces.append(
                        (np.mean(hist_results, axis=0), bin_edges, color, True)
                    )
                return traces

            panel_traces = {
                label: compute_panel_traces(stim) for stim, label in psth_panels
            }

            # independently adjust the firing-rate maximum for this unit so its
            # traces just fit within the PSTH panels
            if max_psth_spike_rate is None:
                peak = max(
                    (
                        float(np.max(hist))
                        for traces in panel_traces.values()
                        for hist, *_ in traces
                    ),
                    default=0.0,
                )
                # small headroom so the peak doesn't touch the panel edge
                max_psth_spike_rate = max(1.0, peak / 0.95)

            scale_bar_len = max(1, int(max_psth_spike_rate / 4))  # Hz

            def add_psth_plot(hist, bin_edges, panel_ymin, panel_ymax, **plot_kwargs):
                # need to plot upside down, scaled within [panel_ymin, panel_ymax]
                norm_spike_rate = (hist / max_psth_spike_rate) * (
                    panel_ymax - panel_ymin
                )
                ax.plot(
                    bin_edges + np.diff(bin_edges)[0] / 2,
                    panel_ymax - norm_spike_rate,
                    **plot_kwargs,
                )

            panel_height = nominal_rows_per_block
            panel_top = max(last_ypos) + ypad
            first_panel_ymin = first_panel_ymax = None
            second_panel_ymin = None
            for target_stim, panel_label in psth_panels:
                panel_ymin = panel_top
                panel_ymax = panel_ymin + panel_height
                if first_panel_ymin is None:
                    first_panel_ymin, first_panel_ymax = panel_ymin, panel_ymax
                elif second_panel_ymin is None:
                    second_panel_ymin = panel_ymin
                for hist, bin_edges, color, is_mean in panel_traces[panel_label]:
                    if is_mean:
                        add_psth_plot(
                            hist, bin_edges, panel_ymin, panel_ymax,
                            lw=0.75, c=color,
                        )
                    else:
                        add_psth_plot(
                            hist, bin_edges, panel_ymin, panel_ymax,
                            lw=0.3, c=color, alpha=0.3,
                        )
                if is_first_column:
                    # target label, aligned with the block context labels column
                    ax.text(
                        x=xlim_0 - 0.6,
                        y=(panel_ymin + panel_ymax) / 2,
                        s=panel_label,
                        fontsize=7,
                        ha="center",
                        va="center",
                        color="k",
                        rotation=0,
                    )
                panel_top = panel_ymax + ypad
            ypos = panel_top - ypad + 0.5

            # scale bar drawn per column because each unit may be scaled to a
            # different max firing rate. Positioned vertically between the AUD+
            # and VIS+ PSTH panels so it doesn't overlap the traces.
            length = (
                (first_panel_ymax - first_panel_ymin)
                * scale_bar_len
                / max_psth_spike_rate
            )
            # midpoint of the gap between the VIS+ (top) and AUD+ (bottom) panels
            if second_panel_ymin is not None:
                bar_center_y = (first_panel_ymax + second_panel_ymin) / 2
            else:
                bar_center_y = first_panel_ymax
            bar_y0 = bar_center_y - length / 2
            bar_y1 = bar_center_y + length / 2
            if single_unit:
                # original placement: in the left margin (matches single-unit fig)
                bar_x = xlim_0 - 0.1
                label_x = xlim_0 - 0.95
                label_ha = "center"
            else:
                # place inside the column so it doesn't collide with neighbours
                bar_x = xlim_0 + 0.03 * (xlim_1 - xlim_0)
                label_x = bar_x + 0.05 * (xlim_1 - xlim_0)
                label_ha = "left"
            ax.plot(
                [bar_x, bar_x],
                [bar_y0, bar_y1],
                c="k",
                lw=1,
                clip_on=False,
            )
            ax.text(
                x=label_x,
                y=bar_center_y,
                s=f"{scale_bar_len} Hz",
                fontsize=6,
                ha=label_ha,
                va="center",
                color="k",
                rotation=0,
            )

        # stim onset vertical line
        ax.axvline(x=0, **line_params)

        ax.set_xlim(xlim_0, xlim_1)

        # baseline / stim / resp / ITI schematic above the rasters (see raster.png)
        y_lower = -0.5
        if show_epoch_schematic:
            schem_gap = 3.0
            schem_height = nominal_rows_per_block * 0.55
            schem_bottom = -schem_gap  # nearer the rasters
            schem_top = schem_bottom - schem_height  # further from the rasters (top after invert)
            resp_start = response_window_start_time
            resp_stop = min(response_window_stop_time, xlim_1)
            stim_stop = min(0.5, xlim_1)  # full stimulus length (0-0.5 s)
            # main row (nearer rasters): baseline / response / ITI. The stimulus
            # overlaps the response window (0.1-1 s) so it is drawn in its own row
            # above this one rather than inline.
            segments = [
                ("Baseline", xlim_0, 0.0, "white"),
                ("Resp", resp_start, resp_stop, [0.85, 0.95, 1.0]),
            ]
            if xlim_1 > resp_stop:
                segments.append(("ITI", resp_stop, xlim_1, "white"))
            for label, x0, x1, facecolor in segments:
                if x1 <= x0:
                    continue
                ax.add_patch(
                    patches.Rectangle(
                        xy=(x0, schem_top),
                        width=x1 - x0,
                        height=schem_bottom - schem_top,
                        linewidth=0.5,
                        edgecolor="k",
                        facecolor=facecolor,
                        clip_on=False,
                        zorder=200,
                    )
                )
                ax.text(
                    x=(x0 + x1) / 2,
                    y=(schem_top + schem_bottom) / 2,
                    s=label,
                    fontsize=6,
                    ha="center",
                    va="center",
                    color="k",
                    clip_on=False,
                    zorder=201,
                )

            # stimulus row, placed above the main row so its full 0-0.5 s extent
            # can be shown without overlapping the response window
            stim_row_gap = 0.75
            stim_row_height = schem_height * 0.9
            stim_row_bottom = schem_top - stim_row_gap
            stim_row_top = stim_row_bottom - stim_row_height
            y_lower = stim_row_top - 1.0
            if stim_stop > 0.0:
                ax.add_patch(
                    patches.Rectangle(
                        xy=(0.0, stim_row_top),
                        width=stim_stop,
                        height=stim_row_bottom - stim_row_top,
                        linewidth=0.5,
                        edgecolor="k",
                        facecolor=[1.0, 0.9, 0.75],
                        clip_on=False,
                        zorder=200,
                    )
                )
                ax.text(
                    x=stim_stop / 2,
                    y=stim_row_top - 0.5,
                    s="Stim",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    color='k',
                    clip_on=False,
                    zorder=201,
                )

        ax.set_ylim(y_lower, max(ypos, *last_ypos))
        ax.set_xticks(sorted({min(xlim_0, 0), 0, xlim_1}))
        # ax.set_xticklabels("" if v % 2 else str(v) for v in ax.get_xticks())
        ax.xaxis.set_tick_params(labelsize=6)
        ax.set_yticks([])
        if is_first_column:
            ax.set_ylabel("← trials")
            ax.yaxis.set_label_coords(x=-0.5, y=0.5)
            ax.text(
                x=xlim_0 - 0.9,
                y=-0,
                s="block #" if display_block_index else "context",
                fontsize=8,
                ha="center",
                va="center",
                color="k",
                rotation=0,
            )
        ax.set_xlabel("time from\nstim onset (s)")
        ax.invert_yaxis()
        ax.set_aspect(0.1 * (xlim_1 - xlim_0) / 3)
        stim_to_label = {
            "vis1": "VIS+",
            "vis2": "VIS-",
            "sound1": "AUD+",
            "sound2": "AUD-",
        }
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_zorder(199)


    if unit.is_empty():
        location = "not in units df"
    else:
        if "location" in unit.columns:
            location = unit["location"][0]
        else:
            location = "unannotated"
    return location


if __name__ == "__main__":

    plot(
        unit_id="742903_2024-10-21_E-179",
        unit_spike_times=np.load(
            r"C:\Users\ben.hardcastle\Downloads\05392fb2-557a-4185-990b-8b94026d7eae.npy"
        ),
    ).savefig("ks4_742903_2024-10-21_E-179.png")
    exit()

    all_stim_names = ("sound1", "vis1", "sound2", "vis2")
    target_stim_names = ("sound1", "vis1")
    pyfile_path = pathlib.Path(__file__)

    raise_on_error = True
    skip_existing = False
    print(f"{skip_existing=}, {raise_on_error=}")
    get_unit_id_func = get_specific_unit_ids
    for unit_id in get_unit_id_func():

        figsave_path = pyfile_path.with_name(f"{pyfile_path.stem}_{unit_id}")
        if get_unit_id_func in (
            get_unit_ids_shawn_session_list,
            get_unit_ids_baseline_psth_parquet,
        ):
            materials_path = pathlib.Path(
                "C:/Users/ben.hardcastle/OneDrive - Allen Institute/Shared Documents - Dynamic Routing/DR Manuscripts/DR Paper 2 - context representations/Figures/Figure 3/materials"
            )
            if get_unit_id_func is get_unit_ids_shawn_session_list:
                requested_path = (
                    materials_path / "top_context_units_by_session_v0.0.235"
                )
            elif get_unit_id_func is get_unit_ids_baseline_psth_parquet:
                area = get_grouped_baseline_psth_parquet().filter(
                    pl.col("unit_id") == unit_id
                )["structure"][0]
                requested_path = (
                    materials_path / "top_context_units_by_area_v0.0.235" / area
                )
            requested_path.mkdir(exist_ok=True, parents=True)
            figsave_path = requested_path / figsave_path.name
        if skip_existing and figsave_path.with_suffix(".png").exists():
            print(f"skipping {pyfile_path.stem} for {unit_id} - already exists")
            continue

        print(f"plotting {pyfile_path.stem} for {unit_id}")
        try:
            fig = plot(
                unit_id=unit_id,
                stim_names=all_stim_names,
                with_instruction_trial_whitespace=False,
                use_session_obj=False,
            )
        except Exception as exc:
            if raise_on_error:
                raise
            print(f"failed: {exc!r}")
            continue
        fig.savefig(f"{figsave_path}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{figsave_path}.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)


# %%
