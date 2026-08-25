# /// script
# dependencies = [
#     "altair==6.2.2",
#     "dr-datacube",
#     "matplotlib",
#     "marimo",
#     "polars==1.43.2",
#     "scipy",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib
    from collections.abc import Iterable

    import matplotlib.pyplot as plt
    import matplotlib.style
    import numpy as np
    import polars as pl
    from dr_datacube.datacube import (
        datacube_config,
        get_lf,
        get_session_ids_from_github,
        on_codeocean,
    )
    from scipy.stats import wilcoxon

    matplotlib.style.use("default")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 8
    plt.rcParams["pdf.fonttype"] = 42

    datacube_config.use_cache = True

    results_dir = (
        pathlib.Path(__file__).resolve().parent if not on_codeocean() else pathlib.Path("/root/capsule/results")
    )
    return (
        Iterable,
        get_lf,
        get_session_ids_from_github,
        np,
        pl,
        plt,
        results_dir,
        wilcoxon,
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
        .collect()
    )
    trials = (
        get_lf("trials").filter(
            pl.col("session_id").is_in(sessions["session_id"].implode()),
            ~(
                pl.col("is_reward_scheduled") & pl.col("trial_index_in_block").gt(14)
            ),  # don't use trials with autorewards after consecutive misses
        )
    ).collect()
    return (trials,)


@app.cell
def _(Iterable, np, pl, plt, wilcoxon):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def format_ax(
        ax,
        ax_idx: int | None,
        is_switch_to_rewarded: bool,
        preTrials: int,
        postTrials: int,
        annotate_rewarded: bool,
        annotate_context: tuple[str, str] = (),
    ) -> None:
        ax.axvline(x=0, color="grey", lw=0.5)
        # green patch for instruction trials
        # if is_switch_to_rewarded:
        #     ax.axvspan(xmin=0, xmax=5, color=[0.9, 0.95, 0.9], lw=0, zorder=-1)
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False)
        xticks = np.arange(-preTrials, postTrials + 1, 1)
        xticks = xticks[xticks != 0]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) if abs(x) == 1 else "" for x in xticks], fontsize=8)
        ax.set_aspect(10)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1], fontsize=8)
        ax.set_xlim([-preTrials - 0.5, postTrials + 0.5])
        ax.set_ylim([0, 1.01])
        ax.set_ylabel("Response probability")
        ax.tick_params(direction="out", top=False, right=False)

        # ax.legend(bbox_to_anchor=(1,1),loc='upper left')
        if ax_idx == 1:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)
        if annotate_rewarded:
            states = ("unrewarded", "rewarded")
            if not is_switch_to_rewarded:
                states = states[::-1]
            transition_x = (0 - ax.get_xlim()[0]) / np.diff(ax.get_xlim())[0]
            marker_half_widths = []
            for i, state in enumerate(states):
                x = transition_x + (-0.13 if i == 0 else 0.13)
                marker_half_widths.append(0.08 if state == "unrewarded" else 0.06)
                if state == "unrewarded":
                    ax.plot(
                        [x - 0.08, x + 0.08],
                        [1.1, 1.1],
                        color="#d62728",
                        lw=2,
                        solid_capstyle="butt",
                        transform=ax.transAxes,
                        clip_on=False,
                        zorder=200,
                    )
                else:
                    drop = Path(
                        [
                            (x, 1.145),
                            (x - 0.014, 1.12),
                            (x - 0.06, 1.10),
                            (x - 0.06, 1.085),
                            (x - 0.06, 1.06),
                            (x - 0.03, 1.045),
                            (x, 1.045),
                            (x + 0.03, 1.045),
                            (x + 0.06, 1.06),
                            (x + 0.06, 1.085),
                            (x + 0.06, 1.10),
                            (x + 0.014, 1.12),
                            (x, 1.145),
                            (x, 1.145),
                        ],
                        [
                            Path.MOVETO,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CURVE4,
                            Path.CLOSEPOLY,
                        ],
                    )
                    ax.add_patch(
                        PathPatch(
                            drop,
                            facecolor="#6ebbdc",
                            edgecolor="#6ebbdc",
                            lw=0.5,
                            transform=ax.transAxes,
                            clip_on=False,
                            zorder=200,
                        )
                    )
            arrow_start = transition_x - 0.13 + marker_half_widths[0] + 0.02
            arrow_end = transition_x + 0.13 - marker_half_widths[1] - 0.02
            ax.annotate(
                "",
                xy=(arrow_end, 1.1),
                xytext=(arrow_start, 1.1),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "black",
                    "lw": 0.8,
                    "mutation_scale": 4,
                },
                annotation_clip=False,
                zorder=201,
            )
        if annotate_context:
            for i, (annotation, color, x) in enumerate(zip(annotate_context, "kk", (-preTrials / 2, postTrials / 2))):
                ax.text(
                    x,
                    1.1,
                    annotation,
                    color=color,
                    fontsize=8,
                    va="center",
                    ha="center",
                )

    def plot(trials: pl.DataFrame, late_autorewards: bool | None = None):
        trials_df = trials.clone()
        fig, axes = plt.subplots(1, 2, figsize=(3, 2), sharex=True, sharey=True)
        transition_stats_rows = []
        for ax_idx, (ax, stimLbl, clr) in enumerate(
            zip(axes, ("rewarded target stim", "unrewarded target stim"), "kk")
        ):
            is_switch_to_rewarded = "unrewarded" not in stimLbl
            preTrials = 2
            postTrials = 2
            x = np.arange(-preTrials, postTrials + 1)
            y = []
            for subject_id, subject_df in trials_df.group_by(["subject_id"]):
                y.append([])
                for session_id, session_df in subject_df.group_by(["session_id"]):
                    d = session_df
                    trialBlock = np.array(d["block_index"])
                    trialResp = np.array(d["is_response"])
                    trialStim = np.array(d["stim_name"])
                    goStim = np.array(d["is_go"])
                    nogoStim = np.array(d["is_nogo"])
                    targetStim = np.array(d["is_vis_target"] | d["is_aud_target"])
                    autoReward = np.array(d["is_reward_scheduled"])
                    for blockInd in np.unique(trialBlock):  # range(1,6):
                        rewStim = trialStim[(trialBlock == blockInd) & goStim][0]
                        nonRewStim = trialStim[(trialBlock == blockInd) & nogoStim & targetStim][0]
                        if (
                            blockInd > 0
                        ):  # and rewStim == blockRewardStim: #! blockRewardStim is defined in the previous cell
                            stim = nonRewStim if "unrewarded" in stimLbl else rewStim
                            trials = trialStim == stim  # & ~autoReward
                            y[-1].append(np.full(preTrials + postTrials + 1, np.nan))
                            pre = trialResp[
                                (trialBlock == blockInd - 1) & trials & ~autoReward
                            ]  # & ~autoReward makes no difference
                            i = min(preTrials, pre.size)
                            y[-1][-1][preTrials - i : preTrials] = pre[-i:]
                            post = trialResp[(trialBlock == blockInd) & trials]
                            i = min(postTrials, post.size)
                            y[-1][-1][preTrials + 1 : preTrials + 1 + i] = post[:i]
                    if np.all(np.isnan(y[-1][-1])):
                        y[-1].pop()
                if len(y[-1]) == 0 or np.all(np.isnan(y[-1])):
                    y.pop()
                    continue
                y[-1] = np.nanmean(y[-1], axis=0)
            y = np.asarray(y, dtype=float)
            last_before = y[:, preTrials - 1]
            first_after = y[:, preTrials + 1]
            valid_pairs = ~(np.isnan(last_before) | np.isnan(first_after))
            last_before = last_before[valid_pairs]
            first_after = first_after[valid_pairs]
            differences = first_after - last_before
            if np.all(differences == 0):
                statistic, p_value = 0.0, 1.0
            else:
                test_result = wilcoxon(first_after, last_before, alternative="two-sided")
                statistic, p_value = float(test_result.statistic), float(test_result.pvalue)
            transition_stats_rows.append(
                {
                    "ax": ax_idx,
                    "transition": "to_rewarded" if is_switch_to_rewarded else "to_unrewarded",
                    "stimulus": stimLbl,
                    "test": "two-sided Wilcoxon signed-rank",
                    "unit": "mouse",
                    "n": len(differences),
                    "last_before_mean": float(np.mean(last_before)),
                    "first_after_mean": float(np.mean(first_after)),
                    "mean_difference": float(np.mean(differences)),
                    "median_difference": float(np.median(differences)),
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
            m = np.nanmean(y, axis=0)
            # There is no trial at x=0; omit that placeholder so matplotlib
            # connects the last pre-switch point to the first post-switch point.
            line_mask = ~np.isnan(m)
            pre_line_mask = line_mask & (x < 0)
            post_line_mask = line_mask & (x > 0)
            ax.plot(
                x[pre_line_mask],
                m[pre_line_mask],
                color=clr,
                label=stimLbl,
                lw=0.3,
                zorder=99,
            )
            ax.plot(
                x[post_line_mask],
                m[post_line_mask],
                color=clr,
                lw=0.3,
                zorder=99,
            )
            ax.plot(
                [x[pre_line_mask][-1], x[post_line_mask][0]],
                [m[pre_line_mask][-1], m[post_line_mask][0]],
                color=clr,
                ls="--",
                lw=0.3,
                zorder=99,
            )
            # Match point colors to the rewarded/unrewarded annotation colors.
            pre_color, post_color = ("r", "c") if is_switch_to_rewarded else ("c", "r")
            for point_mask, point_color in ((x < 0, pre_color), (x > 0, post_color)):
                ax.plot(
                    x[point_mask],
                    m[point_mask],
                    ".",
                    color=point_color,
                    ms=4,
                    zorder=99,
                    clip_on=False,
                )
            is_sem = False
            if is_sem:
                s = np.nanstd(y, axis=0) / (len(y) ** 0.5)
                ax.fill_between(x, m + s, m - s, color=clr, alpha=0.1, edgecolor="none", zorder=50)
            else:
                y = np.array(y)
                lower = np.full(len(m), np.nan)
                upper = np.full(len(m), np.nan)
                for i in range(len(m)):
                    ys = y[~np.isnan(y[:, i]), i]
                    # all nans at i=0 will raise a warning
                    lower[i], upper[i] = np.percentile(
                        [np.nanmean(np.random.choice(ys, size=ys.size, replace=True)) for _ in range(1000)],
                        (5, 95),
                    )
                ax.fill_between(x, upper, lower, color=clr, alpha=0.1, edgecolor="none", zorder=50)
            format_ax(ax, ax_idx, is_switch_to_rewarded, preTrials, postTrials, True)
            print(len(y), "mice")
            ax.set_zorder(199)

            plt.tight_layout()

            if late_autorewards is not None:
                autorewards_name = "late-autorewards" if late_autorewards else "early-autorewards"
            else:
                autorewards_name = "all-autorewards"
            # utils.savefig(__file__, fig, suffix=autorewards_name)
        fig.supxlabel("N stim presentations\nfrom context change", fontsize=8, y=0.08)
        fig.tight_layout()
        return fig, pl.DataFrame(transition_stats_rows)

    return (plot,)


@app.cell
def _(plot, results_dir, trials):
    fig, transition_stats = plot(trials, late_autorewards=None)  # both targets
    fig.savefig(results_dir / "block-switch.svg")
    transition_stats.write_csv(results_dir / "block-switch-stats.csv")


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
