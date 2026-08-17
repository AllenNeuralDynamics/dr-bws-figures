# /// script
# dependencies = [
#     "altair==6.2.2",
#     "dr-bws",
#     "matplotlib",
#     "marimo",
#     "polars==1.43.2",
# ]
# requires-python = "3.11"
#
# [tool.uv.sources]
# dr-bws = { git = "https://github.com/AllenNeuralDynamics/dr-bws-figures" }
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import pathlib

    import polars as pl

    from dr_bws.datacube import get_lf, get_session_ids_from_github, on_codeocean

    asset_dir = (
        pathlib.Path(__file__).resolve().parent
        if not on_codeocean()
        else pathlib.Path("/root/capsule/results")
    )
    return asset_dir, get_lf, get_session_ids_from_github, pl


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
def _(asset_dir, get_lf, pl, sessions):
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
                "is_first_block_aud",
            ],
        )
        .with_columns(pl.col("target").str.split("_").list.first())
        .sort("session_id", "is_first_block_aud", "block_index", "rewarded_modality")
        .collect()
    )
    target_response_rate.write_csv(asset_dir / "target_response_rate.csv")
    target_response_rate
    return (target_response_rate,)


@app.cell
def _(asset_dir, pl, target_response_rate):
    target_response_rate_agg = (
        target_response_rate.group_by(
            "subject_id", "block_index", "rewarded_modality", "target", "is_first_block_aud"
        )
        .agg(
            pl.all(),
            pl.col("response_rate", "cross_modality_dprime", "aud_dprime", "vis_dprime")
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
    target_response_rate_agg.write_csv(asset_dir / "target_response_rate_agg.csv")
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


@app.cell
def _():
    first_block_aud = True
    subject_traces = True
    targets = ["vis"]
    error_bars = "ci95"
    return error_bars, first_block_aud, subject_traces, targets


@app.cell
def _(
    error_bars,
    first_block_aud,
    pl,
    subject_traces,
    target_response_rate,
    targets,
):
    import matplotlib.pyplot as plt
    import numpy as np

    _data = target_response_rate.filter(
        pl.col("is_first_block_aud") == first_block_aud,
    )
    if len(targets) == 1:
        target = targets[0]
        _data = _data.filter(pl.col("target") == target)

    _subject_data = (
        _data.group_by("subject_id", "block_index", "target")
        .agg(pl.col("response_rate").mean())
        .sort("subject_id", "target", "block_index")
    )
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

    _colors = {"vis": "#0072B2", "aud": "#D55E00"}
    _fig, _ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)

    for _target in ("vis", "aud"):
        _subject_target = _subject_data.filter(pl.col("target") == _target)
        if subject_traces:
            for _trace in _subject_target.partition_by("subject_id", as_dict=False):
                _ax.plot(
                    _trace["block_index"].to_numpy(),
                    _trace["response_rate"].to_numpy(),
                    color=_colors[_target],
                    alpha=0.15,
                    linewidth=0.7,
                    zorder=1,
                )

        _target_summary = _summary.filter(pl.col("target") == _target)
        if _target_summary.is_empty():
            continue

        _x = _target_summary["block_index"].to_numpy()
        _mean = _target_summary["mean"].to_numpy()
        _ax.plot(
            _x,
            _mean,
            color=_colors[_target],
            marker="o",
            linewidth=2,
            label=f"{_target} target",
            zorder=3,
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
                ecolor=_colors[_target],
                elinewidth=1,
                capsize=0,
                zorder=2,
            )

    _ax.set(
        xlabel="Block #",
        ylabel="Response rate" if len(targets) == 2 else f"{targets[0].capitalize()} target response rate",
        xticks=sorted(_data["block_index"].unique().to_list()),
        ylim=(0, 1.05),
    )
    _ax.grid(axis="y", alpha=0.25)
    if len(targets) == 2:
        _ax.legend(frameon=False)
    _fig


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
