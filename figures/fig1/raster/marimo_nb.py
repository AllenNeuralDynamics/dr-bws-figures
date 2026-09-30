# /// script
# dependencies = [
#     "dr-datacube",
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pydantic-settings",
#     "polars==1.43.2",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///

"""Plot example-unit rasters and context-split PSTHs for Figure 1."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import logging
    from pathlib import Path

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from dr_datacube import config as datacube_config
    from dr_datacube import get_lf, on_codeocean
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class RasterSettings(BaseSettings):
        """CLI settings for the raster figure, with the paper run as defaults."""

        model_config = SettingsConfigDict(
            cli_parse_args=True,
            cli_kebab_case=False,
            cli_ignore_unknown_args=True,
        )

        unit_id: tuple[str, ...] | str = (
            "666986_2023-08-15_B-123",
            "667252_2023-09-26_F-628",
            "666986_2023-08-15_D-234",
        )
        logging_level: str = "INFO"
        xlim_min: float = -1.5
        xlim_max: float = 1.5
        max_psth_spike_rate: float = -1.0
        display_block_index: bool = False
        plot_blockwise_psths: bool = False
        show_epoch_schematic: bool = False
        fig_width_per_unit: float = 2.0
        fig_height: float = 5.0
        figscale: float = 1.0
        plot_blockwise_rate: bool = True
        psth_blockwise_gap: float = 25.0

        @property
        def unit_ids(self) -> tuple[str, ...]:
            raw_unit_ids = (
                (self.unit_id,)
                if isinstance(self.unit_id, str)
                else self.unit_id
            )
            return tuple(
                unit_id.strip()
                for unit_id in raw_unit_ids
                for unit_id in unit_id.split(",")
                if unit_id.strip()
            )

        @property
        def max_psth_spike_rate_value(self) -> float | None:
            return (
                None
                if self.max_psth_spike_rate <= 0
                else self.max_psth_spike_rate
            )

        @property
        def fig_height_value(self) -> float | None:
            return self.fig_height if self.fig_height > 0 else None

    settings = RasterSettings()

    datacube_config.use_cache = True
    logging.basicConfig(level=getattr(logging, settings.logging_level.upper()))

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 8
    plt.rcParams["pdf.fonttype"] = 42

    results_dir = (
        Path(__file__).resolve().parent
        if not on_codeocean()
        else Path("/root/capsule/results")
    )
    return get_lf, np, patches, pl, plt, results_dir, settings


@app.cell
def _(get_lf, pl, settings):
    unit_ids = settings.unit_ids
    session_ids = tuple(dict.fromkeys("_".join(unit_id.split("_")[:2]) for unit_id in unit_ids))

    units = pl.concat(
        [
            get_lf("units", session_id=session_id)
            .filter(pl.col("unit_id").is_in(unit_ids))
            .select("unit_id", "session_id", "location", "structure", "spike_times")
            .collect()
            for session_id in session_ids
        ]
    )
    trials = (
        get_lf("trials")
        .filter(pl.col("session_id").is_in(session_ids))
        .collect()
    )

    missing_unit_ids = sorted(set(unit_ids) - set(units["unit_id"]))
    if missing_unit_ids:
        raise ValueError(f"Units not found: {missing_unit_ids}")
    return trials, unit_ids, units


@app.cell
def _(np):
    def make_psth(
        spike_times,
        start_times,
        window_duration,
        bin_size=0.025,
        smoothing_window=0.05,
    ):
        """Return a trial-averaged boxcar-smoothed PSTH using only NumPy."""
        spike_times = np.sort(np.asarray(spike_times, dtype=float).reshape(-1))
        start_times = np.asarray(start_times, dtype=float).reshape(-1)
        if start_times.size == 0:
            raise ValueError("Cannot compute a PSTH without trial start times")

        shifted_starts = start_times - smoothing_window / 2
        padded_duration = window_duration + smoothing_window
        bins = np.arange(0, padded_duration + bin_size, bin_size)
        counts = np.zeros(bins.size - 1, dtype=float)
        for start_time in shifted_starts:
            start_index, stop_index = np.searchsorted(
                spike_times, (start_time, start_time + padded_duration)
            )
            counts += np.histogram(
                spike_times[start_index:stop_index] - start_time,
                bins=bins,
            )[0]

        kernel_size = max(1, round(smoothing_window / bin_size))
        kernel = np.ones(kernel_size, dtype=float)
        rates = np.convolve(counts / start_times.size, kernel)
        rates /= bin_size * kernel.size
        return rates[kernel.size - 1 : -kernel.size], bins[: -kernel.size - 1]

    return (make_psth,)


@app.cell
def _(make_psth, np, patches, pl, plt):
    VIS_CONTEXT_COLOR = "#4258A7"
    AUD_CONTEXT_COLOR = "#F36B10"
    VIS_TARGET_COLOR = "#BF00BF"
    AUD_TARGET_COLOR = "#2ca02c"
    INSTRUCTION_COLOR = "#D0B9DB"

    def add_unit_spikes(trials_for_session, spike_times, x_min):
        pad_start = -x_min + 0.5
        bounds = trials_for_session.select(
            pl.col("start_time") - pad_start,
            "stop_time",
        ).to_numpy()
        spike_ranges = np.searchsorted(spike_times, bounds)
        spikes_by_trial = tuple(
            spike_times[start:stop]
            for start, stop in spike_ranges
        )
        return (
            trials_for_session.with_columns(
                pl.Series("spike_times", spikes_by_trial, dtype=pl.List(pl.Float64))
            )
            .with_row_index()
            .explode("spike_times", empty_as_null=True)
            .with_columns(
                stim_centered_spike_times=pl.col("spike_times")
                - pl.col("stim_start_time")
            )
            .group_by(
                pl.all().exclude("spike_times", "stim_centered_spike_times"),
                maintain_order=True,
            )
            .all()
            .filter(pl.col("stim_name").is_in(("vis1", "sound1", "vis2", "sound2")))
        )

    def plot_unit(ax, unit_id, units, all_trials, settings, *, is_first_column):
        x_min = settings.xlim_min
        x_max = settings.xlim_max
        unit = units.filter(pl.col("unit_id") == unit_id)
        session_id = unit["session_id"][0]
        location = unit["location"][0]
        spike_times = np.sort(np.asarray(unit["spike_times"][0], dtype=float))
        session_trials = all_trials.filter(pl.col("session_id") == session_id)
        trials = add_unit_spikes(session_trials, spike_times, x_min)
        if trials.is_empty():
            raise ValueError(f"No plotted trials found for {unit_id}")

        n_blocks = trials["block_index"].n_unique()
        nominal_rows_per_block = 12
        block_height = 8 * nominal_rows_per_block / n_blocks
        last_y = 0.0

        for block_index in sorted(trials["block_index"].unique()):
            block = trials.filter(pl.col("block_index") == block_index).sort("trial_index")
            y_positions = (
                np.linspace(0, block_height, len(block), endpoint=False)
                + block_index * block_height
            )
            half_row = 0.5 * np.diff(y_positions).mean()
            y_positions += half_row
            is_vis_context = bool(block["is_vis_rewarded"][0])
            n_instruction = max(
                block.filter(
                    pl.col(f"is_{modality}_rewarded"),
                    pl.col("is_reward_scheduled"),
                    pl.col("trial_index_in_block") < 14,
                ).height
                for modality in ("aud", "vis")
            )

            if n_instruction:
                ax.axhspan(
                    y_positions[0] - half_row,
                    y_positions[n_instruction - 1] + half_row,
                    color=INSTRUCTION_COLOR,
                    alpha=0.5,
                    linewidth=0,
                    zorder=-1,
                )
            if is_vis_context and len(block) > n_instruction:
                ax.axhspan(
                    y_positions[n_instruction] - half_row,
                    y_positions[-1] + half_row,
                    color="#E6E7E8",
                    linewidth=0,
                    zorder=-1,
                )

            if is_first_column:
                ax.text(
                    x_min - 0.6,
                    y_positions[0] + block_height / 2,
                    str(block_index + 1)
                    if settings.display_block_index
                    else ("VIS" if is_vis_context else "AUD"),
                    color=VIS_CONTEXT_COLOR if is_vis_context else AUD_CONTEXT_COLOR,
                    ha="center",
                    va="center",
                )

            for trial, y_position in zip(block.iter_rows(named=True), y_positions):
                if trial["is_reward_scheduled"] and trial["trial_index_in_block"] > 10:
                    ax.axhspan(
                        y_position - half_row,
                        y_position + half_row,
                        color=INSTRUCTION_COLOR,
                        alpha=0.5,
                        linewidth=0,
                        zorder=-1,
                    )
                centered_spikes = np.asarray(trial["stim_centered_spike_times"])
                if centered_spikes.size and centered_spikes[0] is not None:
                    ax.eventplot(
                        centered_spikes,
                        lineoffsets=y_position,
                        linewidths=0.2,
                        linelengths=0.4,
                        color="black",
                        zorder=99,
                    )
            last_y = y_positions[-1]

        bin_size = 0.025
        pad_start = -x_min + 0.5
        panel_height = nominal_rows_per_block
        panel_gap = 5
        panel_top = last_y + panel_gap
        panel_definitions = (("vis1", "V+", VIS_TARGET_COLOR), ("sound1", "A+", AUD_TARGET_COLOR))
        panel_traces = {}

        for stimulus, label, _ in panel_definitions:
            traces = []
            for context, color in (("vis", VIS_CONTEXT_COLOR), ("aud", AUD_CONTEXT_COLOR)):
                block_rates = []
                bin_edges = None
                for block in trials.partition_by("block_index", maintain_order=True):
                    selected = block.filter(
                        pl.col(f"is_{context}_rewarded"),
                        pl.col("stim_name") == stimulus,
                    )
                    if selected.is_empty():
                        continue
                    rates, bin_edges = make_psth(
                        spike_times,
                        selected["stim_start_time"].to_numpy() - pad_start,
                        pad_start + x_max,
                        bin_size,
                    )
                    block_rates.append(rates)
                if block_rates:
                    if settings.plot_blockwise_psths:
                        traces.extend(
                            (rates, bin_edges - pad_start, color, False)
                            for rates in block_rates
                        )
                    traces.append(
                        (np.mean(block_rates, axis=0), bin_edges - pad_start, color, True)
                    )
            panel_traces[label] = traces

        peak_rate = max(
            (
                float(np.max(rates))
                for traces in panel_traces.values()
                for rates, _, _, _ in traces
            ),
            default=1.0,
        )
        max_rate = (
            settings.max_psth_spike_rate_value
            if settings.max_psth_spike_rate_value is not None
            else max(1.0, peak_rate / 0.95)
        )
        panel_bounds = []
        for _, label, label_color in panel_definitions:
            panel_min, panel_max = panel_top, panel_top + panel_height
            panel_bounds.append((panel_min, panel_max))
            for rates, bin_edges, color, is_mean in panel_traces[label]:
                scaled_rates = rates / max_rate * panel_height
                ax.plot(
                    bin_edges + np.diff(bin_edges)[0] / 2,
                    panel_max - scaled_rates,
                    color=color,
                    linewidth=1.125 if is_mean else 0.45,
                    alpha=1.0 if is_mean else 0.3,
                )
            if is_first_column:
                ax.text(
                    x_min - 0.6,
                    (panel_min + panel_max) / 2,
                    label,
                    color=label_color,
                    fontsize=8,
                    ha="center",
                    va="center",
                )
            panel_top = panel_max + panel_gap

        scale_hz = max(1, int(max_rate / 4))
        scale_length = panel_height * scale_hz / max_rate
        scale_center = (panel_bounds[0][1] + panel_bounds[1][0]) / 2
        scale_x = x_min + 0.03 * (x_max - x_min)
        ax.plot(
            [scale_x, scale_x],
            [scale_center - scale_length / 2, scale_center + scale_length / 2],
            color="black",
            linewidth=1.5,
            clip_on=False,
        )
        ax.text(
            scale_x + 0.05 * (x_max - x_min),
            scale_center,
            f"{scale_hz} Hz",
            fontsize=6,
            ha="left",
            va="center",
        )

        ax.set_xlim(x_min, x_max)

        y_min = -0.5
        if settings.show_epoch_schematic:
            schematic_height = nominal_rows_per_block * 0.55
            schematic_bottom = -3.0
            schematic_top = schematic_bottom - schematic_height
            for label, start, stop, facecolor in (
                ("Baseline", x_min, 0.0, "white"),
                ("Resp", 0.1, min(1.0, x_max), [0.85, 0.95, 1.0]),
            ):
                ax.add_patch(
                    patches.Rectangle(
                        (start, schematic_top),
                        stop - start,
                        schematic_bottom - schematic_top,
                        linewidth=0.5,
                        edgecolor="black",
                        facecolor=facecolor,
                        clip_on=False,
                        zorder=200,
                    )
                )
                ax.text(
                    (start + stop) / 2,
                    (schematic_top + schematic_bottom) / 2,
                    label,
                    fontsize=6,
                    ha="center",
                    va="center",
                    zorder=201,
                )
            stimulus_bottom = schematic_top - 0.75
            stimulus_top = stimulus_bottom - schematic_height * 0.9
            ax.add_patch(
                patches.Rectangle(
                    (0.0, stimulus_top),
                    min(0.5, x_max),
                    stimulus_bottom - stimulus_top,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor=[1.0, 0.9, 0.75],
                    clip_on=False,
                    zorder=200,
                )
            )
            ax.text(0.25, stimulus_top - 0.5, "stim", fontsize=6, ha="center", va="bottom")
            y_min = stimulus_top - 1.0

        data_bottom = panel_bounds[-1][1]
        axis_bottom = data_bottom + 3.5

        if settings.plot_blockwise_rate:
            sorted_spikes = np.sort(spike_times)

            def block_rate(block, start, stop):
                trial_starts = block["stim_start_time"].to_numpy()
                if trial_starts.size == 0:
                    return np.nan
                counts = np.searchsorted(
                    sorted_spikes, trial_starts + stop
                ) - np.searchsorted(sorted_spikes, trial_starts + start)
                return float(counts.mean()) / (stop - start)

            block_ids = sorted(trials["block_index"].unique())
            baseline_rates = []
            vis_rates = []
            aud_rates = []
            block_is_vis = []
            for block_index in block_ids:
                block = trials.filter(pl.col("block_index") == block_index)
                block_is_vis.append(bool(block["is_vis_rewarded"][0]))
                baseline_rates.append(block_rate(block, -0.5, 0.0))
                vis_rates.append(
                    block_rate(block.filter(pl.col("stim_name") == "vis1"), 0.0, 0.5)
                )
                aud_rates.append(
                    block_rate(block.filter(pl.col("stim_name") == "sound1"), 0.0, 0.5)
                )

            block_edges = np.linspace(x_min, x_max, len(block_ids) + 1)
            block_centers = (block_edges[:-1] + block_edges[1:]) / 2
            rate_panel_height = nominal_rows_per_block
            rate_panel_gap = panel_gap
            rate_region_top = axis_bottom + settings.psth_blockwise_gap
            rate_panels = (
                ("baseline", baseline_rates, "black"),
                ("V+", vis_rates, VIS_TARGET_COLOR),
                ("A+", aud_rates, AUD_TARGET_COLOR),
            )
            rate_bounds = tuple(
                (
                    rate_region_top + index * (rate_panel_height + rate_panel_gap),
                    rate_region_top + index * (rate_panel_height + rate_panel_gap) + rate_panel_height,
                )
                for index in range(len(rate_panels))
            )
            rate_region_bottom = rate_bounds[-1][1]

            for index, is_vis_context in enumerate(block_is_vis):
                if is_vis_context:
                    ax.add_patch(
                        patches.Rectangle(
                            (block_edges[index], rate_region_top),
                            block_edges[index + 1] - block_edges[index],
                            rate_region_bottom - rate_region_top,
                            facecolor="#E6E7E8",
                            edgecolor="none",
                            zorder=-1,
                            clip_on=False,
                        )
                    )
                ax.text(
                    block_centers[index],
                    rate_region_bottom + 1.0,
                    "VIS" if is_vis_context else "AUD",
                    color=VIS_CONTEXT_COLOR if is_vis_context else AUD_CONTEXT_COLOR,
                    fontsize=7,
                    ha="center",
                    va="top",
                    clip_on=False,
                )

            all_rates = np.concatenate(
                [np.asarray(rates, dtype=float) for _, rates, _ in rate_panels]
            )
            max_block_rate = max(1.0, float(np.nanmax(all_rates)))
            for (label, rates, color), (panel_top, panel_bottom) in zip(
                rate_panels, rate_bounds
            ):
                y_values = panel_bottom - np.asarray(rates) / max_block_rate * rate_panel_height
                ax.plot(
                    block_centers,
                    y_values,
                    color=color,
                    marker="o",
                    markersize=2.0,
                    linewidth=0.75,
                    clip_on=False,
                )
                if is_first_column:
                    ax.text(
                        x_min - 0.6,
                        (panel_top + panel_bottom) / 2,
                        label,
                        color=color,
                        fontsize=8,
                        ha="center",
                        va="center",
                        clip_on=False,
                    )

            scale_hz_block = max(1, int(max_block_rate / 2))
            block_scale_length = rate_panel_height * scale_hz_block / max_block_rate
            block_scale_center = (rate_region_top + rate_region_bottom) / 2
            block_scale_x = x_min + 0.02 * (x_max - x_min)
            ax.plot(
                [block_scale_x, block_scale_x],
                [
                    block_scale_center - block_scale_length / 2,
                    block_scale_center + block_scale_length / 2,
                ],
                color="black",
                linewidth=1.5,
                clip_on=False,
            )
            ax.text(
                block_scale_x + 0.03 * (x_max - x_min),
                block_scale_center,
                f"{scale_hz_block} Hz",
                fontsize=6,
                ha="left",
                va="center",
            )
            axis_bottom = rate_region_bottom + 2.0

        ax.plot(
            [0, 0],
            [y_min, data_bottom],
            color="grey",
            linewidth=0.3,
            zorder=1,
        )
        ax.set_ylim(y_min, axis_bottom)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.invert_yaxis()
        ax.set_aspect("auto")
        ax.set_title(location, fontsize=8)

        if is_first_column:
            ax.set_ylabel("← trials")
            ax.yaxis.set_label_coords(
                x_min - 0.9,
                last_y / 2,
                transform=ax.transData,
            )

        if is_first_column:
            time_bar_y = data_bottom + 1.5
            if x_min < 0:
                ax.plot(
                    [x_min, 0.0],
                    [time_bar_y] * 2,
                    color="black",
                    linewidth=1.5,
                    clip_on=False,
                )
                ax.text(
                    x_min,
                    time_bar_y + 1.2,
                    f"{abs(x_min):g} s\nbaseline",
                    fontsize=6,
                    ha="left",
                    va="top",
                )
            ax.plot(
                [0.03, 0.5],
                [time_bar_y] * 2,
                color="grey",
                linewidth=1.5,
                clip_on=False,
            )
            ax.text(
                0.03,
                time_bar_y + 1.2,
                "0.5 s\nstim",
                fontsize=6,
                ha="left",
                va="top",
                color="grey",
            )

    def plot_raster(unit_ids, units, trials, settings):
        figure, axes = plt.subplots(
            1,
            len(unit_ids),
            figsize=(
                settings.fig_width_per_unit * len(unit_ids) * settings.figscale,
                settings.fig_height_value * settings.figscale,
            ),
            sharey=True,
        )
        for index, (axis, unit_id) in enumerate(zip(axes, unit_ids)):
            plot_unit(
                axis,
                unit_id,
                units,
                trials,
                settings,
                is_first_column=index == 0,
            )
        figure.set_dpi(300)
        return figure

    return (plot_raster,)


@app.cell
def _(plot_raster, plt, results_dir, settings, trials, unit_ids, units):
    figure = plot_raster(unit_ids, units, trials, settings)
    figure.savefig(results_dir / "raster.png", dpi=300, bbox_inches="tight")
    figure.savefig(results_dir / "raster.svg", bbox_inches="tight")
    plt.close(figure)
    return (figure,)


if __name__ == "__main__":
    app.run()
