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

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import matplotlib.pyplot as plt
    import matplotlib.style
    import numpy as np
    import polars as pl
    from dr_datacube import config as datacube_config
    from dr_datacube import (
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
def _(np, pl, plt, wilcoxon):
    from matplotlib.path import Path

    instruction_color = "#D0B9DB"
    transition_colors = {"to_unrewarded": "#d62728", "to_rewarded": "black"}

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
        # Patch the first five post-switch trials.
        ax.axvspan(xmin=0, xmax=4, color=instruction_color, alpha=0.4, lw=0, zorder=-1)
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False)
        xticks = sorted(set(np.arange(-preTrials, postTrials + 1, 5)) - {0} | {-1})
        xticks = [-15, -1, 5, 10]
        if is_switch_to_rewarded:
            xticks.remove(10)
        ax.set_xticks(xticks)
        if is_switch_to_rewarded:
            xticklabels = [str(x + 1) if x in (5, ) else str(x) if x in (-15, -1,) else "" for x in xticks]
        else:
            xticklabels = [str(x - 4) if x in (5, 10, 15,) else str(x) if x in (-15, -1,) else "" for x in xticks]
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1], fontsize=8)
        ax.set_xlim([-preTrials - 0.5, postTrials + 0.5])
        ax.set_ylim([0, 1.05])
        ax.tick_params(direction="out", top=False, right=False)
        if not is_switch_to_rewarded:
            # Hide the x-axis baseline across the highlighted gap for switches to unrewarded.
            ax.plot([-0.89, 4.89], [0, 0], color="white", lw=1.0, zorder=250, solid_capstyle="butt", clip_on=False)

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
                marker_half_widths.append(0.06 if state == "unrewarded" else 0.03375)
                if state == "unrewarded":
                    ax.plot(
                        [x - 0.06, x + 0.06],
                        [1.1, 1.1],
                        color="#d62728",
                        lw=2,
                        solid_capstyle="butt",
                        transform=ax.transAxes,
                        clip_on=False,
                        zorder=200,
                    )
                else:
                    # A scatter marker is sized in points, so the drop keeps its
                    # aspect ratio even when the axes are made narrower.
                    drop = Path(
                        [
                            (0.0, 1.0),
                            (-0.2, 0.5),
                            (-1.0, 0.1),
                            (-1.0, -0.2),
                            (-1.0, -0.7),
                            (-0.5, -1.0),
                            (0.0, -1.0),
                            (0.5, -1.0),
                            (1.0, -0.7),
                            (1.0, -0.2),
                            (1.0, 0.1),
                            (0.2, 0.5),
                            (0.0, 1.0),
                            (0.0, 1.0),
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
                    ax.scatter(
                        [x],
                        [1.095],
                        marker=drop,
                        s=42,
                        facecolor="#6ebbdc",
                        edgecolor="#6ebbdc",
                        linewidth=0.5,
                        transform=ax.transAxes,
                        clip_on=False,
                        zorder=200,
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
                    "arrowstyle": "->",
                    "color": "black",
                    "lw": 0.4,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "mutation_scale": 5,
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
                    fontsize=6,
                    va="center",
                    ha="center",
                )

    def plot(trials: pl.DataFrame, late_autorewards: bool | None = None):
        trials_df = trials.clone()
        fig, axes = plt.subplots(1, 2, figsize=(2.4, 2), sharey=True)
        transition_stats_rows = []
        transition_data = {}
        individual_transition_rows = []
        for ax_idx, (ax, stimLbl, clr) in enumerate(
            zip(axes, ("rewarded target stim", "unrewarded target stim"), "kk")
        ):
            is_switch_to_rewarded = "unrewarded" not in stimLbl
            preTrials = 15
            postTrials = 15
            x = np.arange(-preTrials, postTrials + 1)
            y = []
            subject_ids = []
            subject_n_sessions = []
            subject_n_transitions = []
            for subject_id, subject_df in trials_df.group_by(["subject_id"]):
                y.append([])
                n_sessions = 0
                for session_id, session_df in subject_df.group_by(["session_id"]):
                    n_session_transitions_before = len(y[-1])
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
                    if len(y[-1]) > n_session_transitions_before and np.all(np.isnan(y[-1][-1])):
                        y[-1].pop()
                    if len(y[-1]) > n_session_transitions_before:
                        n_sessions += 1
                if len(y[-1]) == 0 or np.all(np.isnan(y[-1])):
                    y.pop()
                    continue
                subject_ids.append(str(subject_id[0]))
                subject_n_sessions.append(n_sessions)
                subject_n_transitions.append(len(y[-1]))
                y[-1] = np.nanmean(y[-1], axis=0)
            y = np.asarray(y, dtype=float)
            transition_name = "to_rewarded" if is_switch_to_rewarded else "to_unrewarded"
            transition_data[transition_name] = y.copy()
            for subject_idx, subject_id in enumerate(subject_ids):
                last_before_value = y[subject_idx, preTrials - 1]
                first_after_value = y[subject_idx, preTrials + 1]
                individual_transition_rows.append(
                    {
                        "subject_id": subject_id,
                        "transition": transition_name,
                        "stimulus": stimLbl,
                        "n_sessions": subject_n_sessions[subject_idx],
                        "n_transitions": subject_n_transitions[subject_idx],
                        "last_before": (
                            None if np.isnan(last_before_value) else float(last_before_value)
                        ),
                        "first_after": (
                            None if np.isnan(first_after_value) else float(first_after_value)
                        ),
                        "change": (
                            None
                            if np.isnan(last_before_value) or np.isnan(first_after_value)
                            else float(first_after_value - last_before_value)
                        ),
                    }
                )
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
                    "transition": transition_name,
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
            # Position the first post-switch point at x=0 for switches to
            # rewarded and x=5 for switches to unrewarded, leaving the
            # highlighted five-trial window at x=0 through x=4.
            pre_x = x[:preTrials]
            post_x = x[preTrials + 1 :] + (4 if not is_switch_to_rewarded else -1)
            _meanlinewidth = 0.8
            ax.plot(
                pre_x,
                m[:preTrials],
                color=clr,
                label=stimLbl,
                linewidth=_meanlinewidth,
                zorder=99,
            )
            ax.plot(
                post_x,
                m[preTrials + 1 :],
                color=clr,
                linewidth=_meanlinewidth,
                zorder=99,
            )
            # Match point colors to the rewarded/unrewarded annotation colors.
            pre_color, post_color = ("r", "k") if is_switch_to_rewarded else ("k", "r")
            for point_x, point_y, point_color in (
                (pre_x[-1], m[preTrials - 1], pre_color),
                (post_x[0], m[preTrials + 1], post_color),
            ):
                ax.plot(
                    [point_x],
                    [point_y],
                    ".",
                    color=point_color,
                    ms=4,
                    zorder=99,
                    clip_on=False,
                )
            if is_switch_to_rewarded:
                ax.plot(
                    post_x[1:5],
                    m[preTrials + 2 : preTrials + 6],
                    ".",
                    color="black",
                    ms=4,
                    zorder=100,
                )
            is_sem = False
            if is_sem:
                s = np.nanstd(y, axis=0) / (len(y) ** 0.5)
                ax.fill_between(
                    pre_x,
                    (m + s)[:preTrials],
                    (m - s)[:preTrials],
                    color=clr,
                    alpha=0.1,
                    edgecolor="none",
                    zorder=50,
                )
                ax.fill_between(
                    post_x,
                    (m + s)[preTrials + 1 :],
                    (m - s)[preTrials + 1 :],
                    color=clr,
                    alpha=0.1,
                    edgecolor="none",
                    zorder=50,
                )
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
                ax.fill_between(
                    pre_x,
                    upper[:preTrials],
                    lower[:preTrials],
                    color=clr,
                    alpha=0.1,
                    edgecolor="none",
                    zorder=50,
                )
                ax.fill_between(
                    post_x,
                    upper[preTrials + 1 :],
                    lower[preTrials + 1 :],
                    color=clr,
                    alpha=0.1,
                    edgecolor="none",
                    zorder=50,
                )
            format_ax(ax, ax_idx, is_switch_to_rewarded, preTrials, postTrials, True)
            print(len(y), "mice")
            ax.set_zorder(199)

            if late_autorewards is not None:
                pass
            else:
                pass
            # utils.savefig(__file__, fig, suffix=autorewards_name)
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.24, top=0.80, wspace=0.1)
        fig.supylabel("Response probability", fontsize=8, x=0.02, y=0.52, va="center")
        fig.supxlabel("N target trials relative to context switch", fontsize=8, x=0.56, y=0.04)
        return (
            fig,
            pl.DataFrame(transition_stats_rows),
            transition_data,
            pl.DataFrame(individual_transition_rows),
        )

    def plot_combined(transition_data):
        fig, ax = plt.subplots(figsize=(1.65, 2))
        preTrials = postTrials = 15
        x = np.arange(-preTrials, postTrials + 1)
        labels = {"to_unrewarded": "to unrewarded", "to_rewarded": "to rewarded"}
        show_instruction_dots = False  # Set True to restore dots at trials 0:5.

        ax.axvline(x=0, color="grey", lw=0.5)
        ax.axvspan(xmin=0, xmax=4, color=instruction_color, alpha=0.4, lw=0, zorder=-1)
        for transition_name in ("to_unrewarded", "to_rewarded"):
            y = transition_data[transition_name]
            m = np.nanmean(y, axis=0)
            is_switch_to_rewarded = transition_name == "to_rewarded"
            pre_x = x[:preTrials]
            post_x = x[preTrials + 1 :] + (4 if not is_switch_to_rewarded else -1)

            lower = np.full(len(m), np.nan)
            upper = np.full(len(m), np.nan)
            for i in range(len(m)):
                ys = y[~np.isnan(y[:, i]), i]
                if len(ys):
                    lower[i], upper[i] = np.percentile(
                        [np.nanmean(np.random.choice(ys, size=ys.size, replace=True)) for _ in range(1000)],
                        (5, 95),
                    )
            ax.fill_between(
                pre_x,
                upper[:preTrials],
                lower[:preTrials],
                color=transition_colors[transition_name],
                alpha=0.1,
                edgecolor="none",
                zorder=50,
            )
            ax.fill_between(
                post_x,
                upper[preTrials + 1 :],
                lower[preTrials + 1 :],
                color=transition_colors[transition_name],
                alpha=0.1,
                edgecolor="none",
                zorder=50,
            )
            ax.plot(
                pre_x,
                m[:preTrials],
                color=transition_colors[transition_name],
                linewidth=0.8,
                label=labels[transition_name],
                zorder=99,
            )
            ax.plot(
                post_x,
                m[preTrials + 1 :],
                color=transition_colors[transition_name],
                linewidth=0.8,
                zorder=99,
            )
            if not is_switch_to_rewarded:
                ax.plot(
                    [pre_x[-1], post_x[0]],
                    [m[preTrials - 1], m[preTrials + 1]],
                    marker="o",
                    color=transition_colors[transition_name],
                    markerfacecolor="none",
                    markeredgecolor=transition_colors[transition_name],
                    markeredgewidth=1.0,
                    linestyle="None",
                    ms=4,
                    zorder=100,
                )
            elif show_instruction_dots:
                ax.plot(
                    [post_x[0]],
                    [m[preTrials + 1]],
                    ".",
                    color="black",
                    ms=4,
                    zorder=100,
                )
                ax.plot(
                    post_x[1:5],
                    m[preTrials + 2 : preTrials + 6],
                    ".",
                    color="black",
                    ms=4,
                    zorder=100,
                )

        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False)
        # Post-switch x positions encode different trial numbers for the two
        # transitions, so only label the shared pre-switch scale.
        ax.set_xticks([-15, -1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim([-preTrials - 0.5, postTrials + 0.5])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("N target trials relative\nto context switch", fontsize=8)
        ax.set_ylabel("Response probability", fontsize=8)
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=1,
            fontsize=6,
            handlelength=1.5,
            loc="lower left",
        )
        fig.subplots_adjust(left=0.29, right=0.97, bottom=0.24, top=0.97)
        return fig

    def plot_individual_transition(transition_data):
        fig, ax = plt.subplots(figsize=(1.35, 2))
        preTrials = 15
        y = transition_data["to_unrewarded"]
        last_rewarded = y[:, preTrials - 1]
        first_unrewarded = y[:, preTrials + 1]
        valid = ~(np.isnan(last_rewarded) | np.isnan(first_unrewarded))
        last_rewarded = last_rewarded[valid]
        first_unrewarded = first_unrewarded[valid]
        point_jitter = np.linspace(-0.08, 0.08, len(last_rewarded))

        for last_value, first_value in zip(last_rewarded, first_unrewarded):
            ax.plot([0, 1], [last_value, first_value], color="0.35", linewidth=0.8, alpha=0.55)
        ax.scatter(
            point_jitter,
            last_rewarded,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.0,
            s=12,
            zorder=2,
        )
        ax.scatter(
            1 + point_jitter,
            first_unrewarded,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.0,
            s=12,
            zorder=2,
        )

        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction="out", top=False, right=False)
        ax.set_xlim([-0.35, 1.35])
        ax.set_ylim([0, 1.05])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["last\nrewarded\ntrial", "first\nunrewarded\ntrial"], fontsize=7)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Response probability", fontsize=8)
        fig.subplots_adjust(left=0.36, right=0.97, bottom=0.22, top=0.97)
        return fig

    return plot, plot_combined, plot_individual_transition


@app.cell
def _(plot, plot_combined, plot_individual_transition, results_dir, trials):
    (
        fig,
        transition_stats,
        transition_data,
        individual_transition_data,
    ) = plot(trials, late_autorewards=None)  # both targets
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(results_dir / "block-switch.svg", **save_kwargs)
    fig.savefig(
        results_dir / "block-switch.png",
        dpi=300,
        transparent=True,
        **save_kwargs,
    )
    transition_stats.write_csv(results_dir / "block-switch-stats.csv")

    combined_fig = plot_combined(transition_data)
    combined_fig.savefig(results_dir / "block-switch-combined.svg", **save_kwargs)
    combined_fig.savefig(
        results_dir / "block-switch-combined.png",
        dpi=300,
        transparent=True,
        **save_kwargs,
    )

    individual_fig = plot_individual_transition(transition_data)
    individual_fig.savefig(results_dir / "block-switch-individual-mice.svg", **save_kwargs)
    individual_fig.savefig(
        results_dir / "block-switch-individual-mice.png",
        dpi=300,
        transparent=True,
        **save_kwargs,
    )
    individual_transition_data.write_csv(
        results_dir / "block-switch-individual-mice.csv"
    )

if __name__ == "__main__":
    app.run()
