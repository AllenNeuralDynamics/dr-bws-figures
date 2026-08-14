# /// script
# dependencies = [
#     "altair==6.2.2",
#     "dr-bws",
#     "marimo",
#     "polars==1.43.2",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# dr-bws = { git = "https://github.com/AllenNeuralDynamics/dr-bws-figures" }
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib

    from dr_bws.sessions import get_lf, get_sessions
    import polars as pl

    asset_dir = (p := pathlib.Path(__file__)).parent
    return asset_dir, get_lf, get_sessions, pl


@app.cell
def _(get_sessions, pl):
    sessions = get_sessions("brainwide").select(
        "session_id",
        pl.col("keywords").list.contains("first_block_aud").alias("is_first_block_aud"),
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
            pl.col("response_rate").mean().alias("response_rate_mean"),
            pl.col("cross_modality_dprime").mean().alias("cross_modality_dprime_mean"),
            pl.col("session_id").n_unique().alias("n_sessions"),
        )
        .group_by("block_index", "rewarded_modality", "target", "is_first_block_aud")
        .agg(
            pl.col("cross_modality_dprime_mean").mean(),
            pl.col("cross_modality_dprime_mean")
            .std()
            .truediv(pl.col("subject_id").n_unique().sqrt())
            .alias("cross_modality_dprime_sem"),
            pl.col("response_rate_mean").mean(),
            pl.col("response_rate_mean")
            .std()
            .truediv(pl.col("subject_id").n_unique().sqrt())
            .alias("response_rate_sem"),
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
    return


@app.cell
def _(target_response_rate_agg):
    (
        target_response_rate_agg.plot.line(
            x="block_index:N",
            y="cross_modality_dprime_mean",
            color="is_first_block_aud",
            tooltip=["cross_modality_dprime_mean", "rewarded_modality", "n_subjects"],
        ).properties(width=200)
    )
    return


if __name__ == "__main__":
    app.run()
