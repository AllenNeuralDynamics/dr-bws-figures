# /// script
# dependencies = [
#     "altair==6.2.2",
#     "dr-bws",
#     "matplotlib",
#     "marimo",
#     "polars==1.43.2",
# ]
# requires-python = "<3.12"
#
# [tool.uv.sources]
# dr-bws = { git = "https://github.com/AllenNeuralDynamics/dr-bws-figures" }
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import polars as pl
    import matplotlib.style
    import matplotlib.pyplot as plt
    import numpy as np

    from dr_bws.datacube import (
        datacube_config,
        get_lf,
        get_session_ids_from_github,
        on_codeocean,
    )

    matplotlib.style.use("default")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 8
    plt.rcParams["pdf.fonttype"] = 42

    datacube_config.use_cache = True

    results_dir = (
        pathlib.Path(__file__).resolve().parent
        if not on_codeocean()
        else pathlib.Path("/root/capsule/results")
    )

    first_block_aud = True
    subject_traces = True
    error_bars = "ci95"
    return (
        error_bars,
        first_block_aud,
        get_lf,
        get_session_ids_from_github,
        np,
        pl,
        plt,
        results_dir,
        subject_traces,
    )


@app.cell
def _(get_lf, get_session_ids_from_github, pl):
    sessions = (
        get_lf("session")
        .filter(pl.col("session_id").is_in(get_session_ids_from_github("brainwide")))
        .select(
            "session_id",
            pl.col("keywords").list.contains("first_block_aud").alias("is_first_block_aud"),
        )
    )
    return (sessions,)


@app.cell
def _(get_lf, pl, results_dir, sessions):
    target_response_rate = (
        get_lf("performance")
        .join(sessions.lazy(), on="session_id", how="inner")
        # pivot so we have block index, rew modal, response rate, and 'target [str]' as cols
        .unpivot(
            on=["vis_target_response_rate", "aud_target_response_rate"],
            value_name="response_rate",
            variable_name="target",
            index=[
                "subject_id",
                "session_id",
                "block_index",
                "rewarded_modality",
                "vis_dprime",
                "aud_dprime",
                "cross_modality_dprime",
                "signed_cross_modality_dprime",
                "is_first_block_aud",
            ],
        )
        .with_columns(pl.col("target").str.split("_").list.first())
        .sort("session_id", "is_first_block_aud", "block_index", "rewarded_modality")
        .collect()
    )
    target_response_rate.write_csv(results_dir / "block_perf.csv")
    target_response_rate
    return (target_response_rate,)


@app.cell
def _(pl, results_dir, target_response_rate):
    target_response_rate_agg = (
        target_response_rate.group_by(
            "subject_id", "block_index", "rewarded_modality", "target", "is_first_block_aud"
        )
        .agg(
            pl.all(),
            pl.col(
                "response_rate",
                "cross_modality_dprime",
                "signed_cross_modality_dprime",
                "aud_dprime",
                "vis_dprime",
            )
            .mean()
            .name.suffix("_mean"),
            pl.col("session_id").n_unique().alias("n_sessions"),
        )
        .group_by("block_index", "rewarded_modality", "target", "is_first_block_aud")
        .agg(
            pl.selectors.ends_with("_mean").mean(),
            pl.col("subject_id").n_unique().alias("n_subjects"),
            pl.col("n_sessions").sum(),
        )
        .sort("is_first_block_aud", "block_index", "rewarded_modality")
    )
    target_response_rate_agg.write_csv(results_dir / "block_perf_agg.csv")
    target_response_rate_agg
    return (target_response_rate_agg,)


@app.cell
def _(target_response_rate_agg):
    (
        target_response_rate_agg.plot.line(
            x="block_index:N",
            y="response_rate_mean",
            color="target",
            column="is_first_block_aud",
            tooltip=["response_rate_mean", "rewarded_modality", "n_subjects"],
        ).properties(width=200)
    )
    return


@app.cell
def _(target_response_rate_agg):
    (
        target_response_rate_agg.plot.line(
            x="block_index:N",
            y="signed_cross_modality_dprime_mean",
            color="target",
            column="is_first_block_aud",
            tooltip=["signed_cross_modality_dprime_mean", "rewarded_modality", "n_subjects"],
        ).properties(width=200)
    )
    return


@app.cell
def _(pl, target_response_rate_agg):
    (
        target_response_rate_agg.unpivot(
            on=["aud_dprime_mean", "vis_dprime_mean"],
            index=["block_index", "rewarded_modality", "is_first_block_aud", "n_subjects"],
            value_name="dprime",
            variable_name="modality",
        )
        .with_columns(pl.col("modality").str.split("_").list.first())
        .plot.line(
            x="block_index:N",
            y="dprime",
            color="modality",
            column="is_first_block_aud",
            tooltip=["dprime", "rewarded_modality", "n_subjects"],
        )
        .properties(width=200)
    )
    return


@app.cell
def _():
    colors = {"vis": "#0072B2", "aud": "#D55E00"}
    figure_kwargs = {"figsize": (1.5, 2)}

    def format_ax(ax, data, targets):
        block_modalities = (
            data.select("block_index", "rewarded_modality")
            .unique()
            .sort("block_index")
            .rows()
        )
        block_ticks = [block_index for block_index, _ in block_modalities]
        ax.set(
            xlabel="Block #",
            xticks=block_ticks,
            xlim=(block_ticks[0] - 0.5, block_ticks[-1] + 0.5),
        )
        for block_index, rewarded_modality in block_modalities:
            ax.axvspan(
                block_index - 0.5,
                block_index + 0.5,
                ymin=1,
                ymax=1.08,
                facecolor="0.85" if rewarded_modality == "aud" else "white",
                edgecolor="0.6",
                linewidth=0.5,
                clip_on=False,
            )
            ax.text(
                block_index,
                1.04,
                "A" if rewarded_modality == "aud" else "V",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=6,
                clip_on=False,
            )
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False, labelsize=8)
        # if len(targets) == 2:
        #     ax.legend(frameon=False)
    return colors, figure_kwargs, format_ax


@app.cell
def _(
    colors,
    error_bars,
    figure_kwargs,
    first_block_aud,
    format_ax,
    np,
    pl,
    plt,
    results_dir,
    subject_traces,
    target_response_rate,
):
    _targets = ["vis", "aud"]

    _subject_data = (
        target_response_rate.filter(
            pl.col("is_first_block_aud") == first_block_aud,
        )
        .group_by("subject_id", "block_index", "rewarded_modality", "target")
        .agg(pl.col("response_rate").mean())
        .sort("subject_id", "target", "block_index")
    )
    if len(_targets) == 1:
        _subject_data = _subject_data.filter(pl.col("target") == _targets[0])

    _summary = (
        _subject_data.group_by("block_index", "target")
        .agg(
            pl.col("response_rate").mean().alias("mean"),
            pl.col("response_rate").std().fill_null(0).alias("std"),
            pl.len().alias("n_subjects"),
        )
        .sort("target", "block_index")
    )

    if error_bars not in {"none", "sem", "ci95"}:
        raise ValueError("error_bars must be one of: 'none', 'sem', or 'ci95'")

    _fig, _ax = plt.subplots(**figure_kwargs)

    for _target in _targets:
        _subject_target = _subject_data.filter(pl.col("target") == _target)
        if subject_traces:
            for _trace in _subject_target.partition_by("subject_id", as_dict=False):
                _ax.plot(
                    _trace["block_index"].to_numpy(),
                    _trace["response_rate"].to_numpy(),
                    color=colors[_target],
                    # alpha=0.15,
                    linewidth=0.05,
                    zorder=1,
                    clip_on=False,
                )

        _target_summary = _summary.filter(pl.col("target") == _target)
        if _target_summary.is_empty():
            continue

        _meanlinewidth = 0.8 
        _x = _target_summary["block_index"].to_numpy()
        _mean = _target_summary["mean"].to_numpy()
        _ax.plot(
            _x,
            _mean,
            color=colors[_target],
            # marker=".",
            markersize=2,
            linewidth=_meanlinewidth,
            label=f"{_target} target",
            zorder=3,
            clip_on=False,
        )

        if error_bars != "none":
            _sem = _target_summary["std"].to_numpy() / np.sqrt(
                _target_summary["n_subjects"].to_numpy()
            )
            if error_bars == "ci95":
                _sem = 1.96 * _sem
            _lower = np.clip(_mean - _sem, 0, 1)
            _upper = np.clip(_mean + _sem, 0, 1)
            _ax.errorbar(
                _x,
                _mean,
                yerr=np.vstack((_mean - _lower, _upper - _mean)),
                fmt="none",
                ecolor=colors[_target],
                elinewidth=_meanlinewidth,
                capsize=0,
                zorder=2,
                clip_on=False,
            )

    format_ax(_ax, _subject_data, _targets)
    _ax.set(
        ylabel="Response probability"
        if len(_targets) == 2
        else f"{_targets[0].capitalize()} target response probability",
        ylim=(0, 1.05),
    )
    _fig.tight_layout()
    _fig.savefig(results_dir / "response-probability.svg")
    _fig
    return


@app.cell
def _(
    error_bars,
    figure_kwargs,
    first_block_aud,
    format_ax,
    np,
    pl,
    plt,
    results_dir,
    subject_traces,
    target_response_rate,
):
    _subject_data = (
        target_response_rate.filter(
            pl.col("is_first_block_aud") == first_block_aud,
            pl.col("target")
            == "aud",  # doesn't matter which we choose here - signed dprime is the same
        )
        .group_by("subject_id", "block_index", "rewarded_modality", "target")
        .agg(pl.col("signed_cross_modality_dprime").mean())
        .sort("subject_id", "target", "block_index")
    )
    _summary = (
        _subject_data.group_by("block_index", "target")
        .agg(
            pl.col("signed_cross_modality_dprime").mean().alias("mean"),
            pl.col("signed_cross_modality_dprime").std().fill_null(0).alias("std"),
            pl.len().alias("n_subjects"),
        )
        .sort("target", "block_index")
    )

    if error_bars not in {"none", "sem", "ci95"}:
        raise ValueError("error_bars must be one of: 'none', 'sem', or 'ci95'")

    _fig, _ax = plt.subplots(**figure_kwargs)

    if subject_traces:
        for _trace in _subject_data.partition_by("subject_id", as_dict=False):
            _ax.plot(
                _trace["block_index"].to_numpy(),
                _trace["signed_cross_modality_dprime"].to_numpy(),
                color="k",
                # alpha=0.15,
                linewidth=0.05,
                zorder=1,
                clip_on=False,
            )

    _x = _summary["block_index"].to_numpy()
    _mean = _summary["mean"].to_numpy()
    _meanlinewidth = 0.8

    _ax.plot(
        _x,
        _mean,
        color="k",
        # marker=".",
        markersize=2,
        linewidth=_meanlinewidth,
        zorder=3,
        clip_on=False,
    )

    if error_bars != "none":
        _sem = _summary["std"].to_numpy() / np.sqrt(_summary["n_subjects"].to_numpy())
        if error_bars == "ci95":
            _sem = 1.96 * _sem
        _lower = _mean - _sem
        _upper = _mean + _sem
        _ax.errorbar(
            _x,
            _mean,
            yerr=np.vstack((_mean - _lower, _upper - _mean)),
            fmt="none",
            ecolor="k",
            elinewidth=_meanlinewidth,
            capsize=0,
            zorder=2,
            clip_on=False,
        )

    format_ax(_ax, _subject_data, [])
    _ax.set(
        ylabel="Cross-modality d'",
        ylim=(-3.5, 3.5),
    )
    _ax.axhline(0, lw=0.5, zorder=0, c="grey")
    _fig.tight_layout()
    _fig.savefig(results_dir / "cross-modality-dprime.svg")
    _fig
    return


@app.cell
def _(
    colors,
    error_bars,
    figure_kwargs,
    first_block_aud,
    format_ax,
    np,
    pl,
    plt,
    results_dir,
    subject_traces,
    target_response_rate,
):
    _modalities = ["vis", "aud"]

    _subject_data = (
        target_response_rate.filter(
            pl.col("is_first_block_aud") == first_block_aud,
        )
        .unpivot(
            on=["aud_dprime", "vis_dprime"],
            index=["subject_id", "block_index", "rewarded_modality"],
            value_name="dprime",
            variable_name="modality",
        )
        .with_columns(pl.col("modality").str.split("_").list.get(0))
        .group_by("subject_id", "block_index", "modality", "rewarded_modality")
        .agg(pl.col("dprime").mean())
        .sort("subject_id", "modality", "block_index")
    )
    if len(_modalities) == 1:
        _subject_data = _subject_data.filter(pl.col("modality") == _modalities[0])

    _summary = (
        _subject_data.group_by("block_index", "modality")
        .agg(
            pl.col("dprime").mean().alias("mean"),
            pl.col("dprime").std().fill_null(0).alias("std"),
            pl.len().alias("n_subjects"),
        )
        .sort("modality", "block_index")
    )

    if error_bars not in {"none", "sem", "ci95"}:
        raise ValueError("error_bars must be one of: 'none', 'sem', or 'ci95'")

    _fig, _ax = plt.subplots(**figure_kwargs)

    for _modality in _modalities:
        _subject_target = _subject_data.filter(pl.col("modality") == _modality)
        if subject_traces:
            for _trace in _subject_target.partition_by("subject_id", as_dict=False):
                _ax.plot(
                    _trace["block_index"].to_numpy(),
                    _trace["dprime"].to_numpy(),
                    color=colors[_modality],
                    # alpha=0.15,
                    linewidth=0.05,
                    zorder=1,
                    clip_on=False,
                )

        _target_summary = _summary.filter(pl.col("modality") == _modality)
        if _target_summary.is_empty():
            continue

        _x = _target_summary["block_index"].to_numpy()
        _mean = _target_summary["mean"].to_numpy()
        _meanlinewidth = 0.8
        _ax.plot(
            _x,
            _mean,
            color=colors[_modality],
            # marker=".",
            markersize=2,
            linewidth=_meanlinewidth,
            label=f"{_modality}",
            zorder=3,
            clip_on=False,
        )

        if error_bars != "none":
            _sem = _target_summary["std"].to_numpy() / np.sqrt(
                _target_summary["n_subjects"].to_numpy()
            )
            if error_bars == "ci95":
                _sem = 1.96 * _sem
            _lower = _mean - _sem
            _upper = _mean + _sem
            _ax.errorbar(
                _x,
                _mean,
                yerr=np.vstack((_mean - _lower, _upper - _mean)),
                fmt="none",
                ecolor=colors[_modality],
                elinewidth=_meanlinewidth,
                capsize=0,
                zorder=2,
                clip_on=False,
            )

    format_ax(_ax, _subject_data, _modalities)
    _ax.set(
        ylabel="d'" if len(_modalities) == 2 else f"{_modalities[0].capitalize()} d'",
        ylim=(-0.99, 4),
    )
    _ax.axhline(0, lw=0.5, zorder=0, c="grey")
    _fig.tight_layout()
    _fig.savefig(results_dir / "intramodal-dprime.svg")
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
