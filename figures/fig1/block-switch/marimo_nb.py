# /// script
# dependencies = [
#     "altair==6.2.2",
#     "dr-bws",
#     "matplotlib",
#     "marimo",
#     "polars==1.43.2",
#     "scipy",
# ]
# requires-python = "<3.12"
#
# [tool.uv.sources]
# dr-bws = { git = "https://github.com/AllenNeuralDynamics/dr-bws-figures" }
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
    from scipy.stats import wilcoxon

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
        ax.set_xticks(np.arange(-preTrials, postTrials + 1, 1))
        xticklabels = ax.get_xticklabels()
        for i in range(len(xticklabels)):
            if i in [0, len(xticklabels) - 1] or xticklabels[i].get_text() == "0":
                continue
            xticklabels[i] = ""
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_aspect(10)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([0, 0.5, 1], fontsize=8)
        ax.set_xlim([-preTrials - 0.5, postTrials + 0.5])
        ax.set_ylim([0, 1.01])
        ax.set_xlabel(
            "N stim presentations\nfrom context change",
        )
        ax.set_ylabel("Response probability")
        ax.tick_params(direction="out", top=False, right=False)

        # ax.legend(bbox_to_anchor=(1,1),loc='upper left')
        if ax_idx == 1:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)
        annotations = ("unrewarded", "rewarded")
        colors = "rc"
        if annotate_rewarded:
            if not is_switch_to_rewarded:
                annotations = annotations[::-1]
                colors = colors[::-1]
            for i, (annotation, color, x) in enumerate(zip(annotations, colors, (0, 0))):
                ax.text(
                    x,
                    1.1,
                    annotation,
                    color=color,
                    fontsize=6,
                    va="center",
                    ha=("right" if i == 0 else "left"),
                )
        if annotate_context:
            for i, (annotation, color, x) in enumerate(
                zip(annotate_context, "kk", (-preTrials / 2, postTrials / 2))
            ):
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
                        nonRewStim = trialStim[(trialBlock == blockInd) & nogoStim & targetStim][
                            0
                        ]
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
            ax.plot(x, m, color=clr, label=stimLbl, lw=0.3, zorder=99)
            # add points
            ax.plot(
                x,
                m,
                ".",
                color=clr,
                # label=(
                #     "instruction trials" if late_autorewards == False else "scheduled reward"
                # ),
                ms=2,
                zorder=99,
                clip_on=False,
            )
            is_sem = False
            if is_sem:
                s = np.nanstd(y, axis=0) / (len(y) ** 0.5)
                ax.fill_between(
                    x, m + s, m - s, color=clr, alpha=0.1, edgecolor="none", zorder=50
                )
            else:
                y = np.array(y)
                lower = np.full(len(m), np.nan)
                upper = np.full(len(m), np.nan)
                for i in range(len(m)):
                    ys = y[~np.isnan(y[:, i]), i]
                    # all nans at i=0 will raise a warning
                    lower[i], upper[i] = np.percentile(
                        [
                            np.nanmean(np.random.choice(ys, size=ys.size, replace=True))
                            for _ in range(1000)
                        ],
                        (5, 95),
                    )
                ax.fill_between(
                    x, upper, lower, color=clr, alpha=0.1, edgecolor="none", zorder=50
                )
            format_ax(ax, ax_idx, is_switch_to_rewarded, preTrials, postTrials, True)
            print(len(y), "mice")
            ax.set_zorder(199)

            plt.tight_layout()

            if late_autorewards is not None:
                autorewards_name = (
                    "late-autorewards" if late_autorewards else "early-autorewards"
                )
            else:
                autorewards_name = "all-autorewards"
            # utils.savefig(__file__, fig, suffix=autorewards_name)
        return fig, pl.DataFrame(transition_stats_rows)

    def plot_suppl(
        trials: pl.DataFrame, combined: bool = False, late_autorewards: bool | None = None
    ):
        trials_df = trials.clone()
        stimNames = ("vis1", "vis2", "sound1", "sound2")
        stimLabels = (
            "visual target",
            "visual non-target",
            "auditory target",
            "auditory non-target",
        )
        preTrials = 2
        postTrials = 1
        x = np.arange(-preTrials, postTrials + 1)
        if combined:
            figStimNames = (None,)
        else:
            figStimNames = stimNames
        figs = []
        for rewardStim, blockLabel in zip(
            ("vis1", "sound1"), ("visual rewarded blocks", "auditory rewarded blocks")
        ):
            for figStimName in figStimNames:
                fig = plt.figure(figsize=(3, 2))
                ax = fig.add_subplot(1, 1, 1)
                for stim, stimLbl, clr, ls in zip(
                    stimNames, stimLabels, "ggmm", ("-", "--", "-", "--")
                ):
                    if not combined and figStimName != stim:
                        continue
                    y = []
                    for subject_id, subject_df in trials_df.group_by("subject_id"):
                        y.append([])
                        for session_id, session_df in subject_df.group_by("session_id"):
                            d = session_df
                            trialBlock = np.array(d["block_index"])
                            trialResp = np.array(d["is_response"])
                            trialStim = np.array(d["stim_name"])
                            goStim = np.array(d["is_go"])
                            autoReward = np.array(d["is_reward_scheduled"])
                            for blockInd in np.unique(trialBlock):  # range(1,6)
                                rewStim = trialStim[(trialBlock == blockInd) & goStim][0]
                                if blockInd > 0 and rewStim == rewardStim:
                                    trials = trialStim == stim  # & ~autoReward
                                    y[-1].append(np.full(preTrials + postTrials + 1, np.nan))
                                    pre = trialResp[(trialBlock == blockInd - 1) & trials]
                                    i = min(preTrials, pre.size)
                                    y[-1][-1][preTrials - i : preTrials] = pre[-i:]
                                    post = trialResp[(trialBlock == blockInd) & trials]
                                    i = min(postTrials, post.size)
                                    y[-1][-1][preTrials + 1 : preTrials + 1 + i] = post[:i]
                        y[-1] = np.nanmean(y[-1], axis=0)
                    m = np.nanmean(y, axis=0)
                    s = np.nanstd(y, axis=0) / (len(y) ** 0.5)
                    # ax.plot(x,m,color=clr,ls=ls,label=stimLbl)
                    # ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
                    ax.plot(x, m, color=clr, label=stimLbl, ls=ls, lw=0.3, zorder=99)
                    is_sem = True
                    if is_sem:
                        s = np.nanstd(y, axis=0) / (len(y) ** 0.5)
                        ax.fill_between(
                            x,
                            m + s,
                            m - s,
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
                                [
                                    np.nanmean(np.random.choice(ys, size=ys.size, replace=True))
                                    for _ in range(1000)
                                ],
                                (5, 95),
                            )
                        ax.fill_between(
                            x,
                            upper,
                            lower,
                            color=clr,
                            alpha=0.1,
                            edgecolor="none",
                            zorder=50,
                        )
                    is_switch_to_rewarded = figStimName == rewardStim and not combined
                    if is_switch_to_rewarded or stim == rewardStim:
                        ax.plot(
                            x[preTrials + 1 : preTrials + 6],
                            m[preTrials + 1 : preTrials + 6],
                            ".",
                            color=clr,
                            label=(
                                "instruction trials"
                                if late_autorewards == False
                                else "scheduled reward"
                            ),
                            ms=1,
                            zorder=99,
                        )

                is_switch_to_vis = "visual" in blockLabel
                annotate_context = ("A", "V") if is_switch_to_vis else ("V", "A")
                format_ax(
                    ax,
                    0,
                    False if combined else is_switch_to_rewarded,
                    preTrials,
                    postTrials,
                    False,
                    annotate_context,
                )
                if "visual" in blockLabel:
                    xmin, xmax = 0, postTrials
                else:
                    xmin, xmax = -preTrials, 0
                ax.axvspan(xmin, xmax, color=[0.95] * 3, lw=0, zorder=-99, clip_on=False)

                # ax.legend(bbox_to_anchor=(1, 1), fontsize=4)
                ax.set_title("")
                plt.tight_layout()

                stim_name = "combined" if combined else figStimName
                if late_autorewards is not None:
                    autorewards_name = (
                        "late-autorewards" if late_autorewards else "early-autorewards"
                    )
                else:
                    autorewards_name = "all-autorewards"
                # utils.savefig(__file__, fig, suffix=f"suppl_{stim_name}_{'to-vis' if 'visual' in blockLabel else 'to-aud'}_{autorewards_name}")
                figs.append(fig)
        return figs

    def plot_last_rewarded_vs_first_unrewarded(
        trials: pl.DataFrame, late_autorewards: bool | None = None, scatter: bool = True
    ):

        trials_df = trials.clone()

        fig, axes = plt.subplots(1, 1, figsize=(2, 2))
        if not isinstance(axes, Iterable):
            axes = [axes]
        for ax_idx, (ax, stimLbl, clr) in enumerate(zip(axes, ("unrewarded target stim",), "k")):
            is_switch_to_rewarded = "unrewarded" not in stimLbl
            preTrials = 2
            postTrials = 1
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
                        nonRewStim = trialStim[(trialBlock == blockInd) & nogoStim & targetStim][
                            0
                        ]
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
            y = np.array(y)
            m = np.nanmean(y, axis=0)
            if scatter:
                ax.scatter(
                    y[:, -1],
                    y[:, 0],
                    facecolor="k",
                    edgecolor="none",
                    lw=0,
                    label=stimLbl,
                    zorder=99,
                    s=0.8,
                    clip_on=False,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                for side in ("right", "top"):
                    ax.spines[side].set_visible(False)
                ax.set_aspect(1)
                ax.set_yticks([0, 0.5, 1])
                ax.set_xticks([0, 0.5, 1])
                ax.set_ylabel("Response probability\nlast rewarded")
                ax.set_xlabel("Response probability\nfirst unrewarded")
            else:
                xpos = (0, 1)

                common_line_params = dict(alpha=1, lw=0.3)
                line_params = {
                    "target": common_line_params.copy(),
                    "nontarget": common_line_params.copy(),
                }
                line_params["target"] |= dict(c=[0.8] * 3)
                ax.plot(
                    xpos, [y[:, 0], y[:, -1]], color=[0.8] * 3, label=stimLbl, lw=0.3, zorder=10
                )
                # format_ax(ax, ax_idx, is_switch_to_rewarded, preTrials, postTrials, True)
                for side in ("right", "top"):
                    ax.spines[side].set_visible(False)
                ax.set_xlim(-0.2, 1.2)
                ax.set_ylim(-0, 1)
                ax.set_aspect(3)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["last\nrewarded", "first\nunrewarded"], fontsize=6)
                ax.plot(
                    xpos,
                    [np.nanmedian(y[:, 0]), np.nanmedian(y[:, -1])],
                    "k.",
                    label=stimLbl,
                    zorder=99,
                    clip_on=False,
                )
                # ax.plot(xpos, [np.median(y[:, 0]), np.median(y[:, -1])], 'k', label=stimLbl, zorder=99, clip_on=False)
                ax.set_ylabel("Response probability")
                lower = np.full(len(x), np.nan)
                upper = np.full(len(x), np.nan)
                for i in range(len(x)):
                    ys = y[~np.isnan(y[:, i]), i]
                    lower[i], upper[i] = np.percentile(
                        [
                            np.nanmedian(np.random.choice(ys, size=ys.size, replace=True))
                            for _ in range(1000)
                        ],
                        (5, 95),
                    )
                # add vertical lines as error bars
                ax.vlines(
                    x=xpos,
                    ymin=lower[::2],
                    ymax=upper[::2],
                    color=[0.5] * 3,
                    lw=1,
                    clip_on=False,
                    zorder=50,
                )
                is_sem = False
                if is_sem:
                    s = np.nanstd(y, axis=0) / (len(y) ** 0.5)
                    ax.fill_between(
                        x, m + s, m - s, color=clr, alpha=0.1, edgecolor="none", zorder=50
                    )
                else:
                    y = np.array(y)
                    lower = np.full(len(m), np.nan)
                    upper = np.full(len(m), np.nan)
                    for i in range(len(m)):
                        ys = y[~np.isnan(y[:, i]), i]
                        # all nans at i=0 will raise a warning
                        lower[i], upper[i] = np.percentile(
                            [
                                np.nanmean(np.random.choice(ys, size=ys.size, replace=True))
                                for _ in range(1000)
                            ],
                            (5, 95),
                        )
                    ax.fill_between(
                        x, upper, lower, color=clr, alpha=0.1, edgecolor="none", zorder=50
                    )
            print(len(y), "mice")
            ax.set_zorder(199)

            plt.tight_layout()

            suffix = "last-vs-first"
            if late_autorewards is not None:
                suffix += "_late-autorewards" if late_autorewards else "_early-autorewards"
            else:
                suffix += "_all-autorewards"
            if scatter:
                suffix += "_scatter"
            else:
                suffix += "_lines"
            # utils.savefig(__file__, fig, suffix=suffix)
        return fig

    return (plot,)


@app.cell
def _(plot, results_dir, trials):
    fig, transition_stats = plot(trials, late_autorewards=None)  # both targets
    fig.savefig(results_dir / "block-switch.svg")
    transition_stats.write_csv(results_dir / "block-switch-stats.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
