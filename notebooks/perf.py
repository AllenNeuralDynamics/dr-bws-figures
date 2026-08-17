import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl

    from dr_bws.datacube import behavior_summary, datacube_config, get_lf, get_sessions

    datacube_config.use_cache = False
    performance = get_lf('performance')
    return behavior_summary, get_lf, get_sessions, performance, pl


@app.cell
def _(behavior_summary, pl):
    def naive_ephys_filter() -> pl.Expr:
        required = ("dynamic_routing", "task", "ephys", "context naive") #TODO switch to "context_naive" when fixed in v0.0.290
        excluded = ("issues", "templeton",) # TODO add "brainwide_survey" when fixed in v0.0.290
        engaged_session_ids = (
            behavior_summary()
            .filter(
                pl.col("n_engaged_blocks").ge(4),
            )
        )["session_id"].to_list()
        return pl.all_horizontal(
            *[pl.col("keywords").list.contains(keyword) for keyword in required],
            *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
            pl.col("session_id").is_in(engaged_session_ids),
        )

    return (naive_ephys_filter,)


@app.cell
def _(get_lf, naive_ephys_filter):
    get_lf("session").filter(naive_ephys_filter()).collect().sort('session_id')


@app.cell
def _(behavior_summary, get_lf, naive_ephys_filter):
    naive = get_lf("session").filter(naive_ephys_filter()).collect().join(behavior_summary(), on='session_id', how='left')
    naive
    return (naive,)


@app.cell
def _(get_sessions):
    get_sessions('naive', only_in_data_asset=False)


@app.cell
def _(get_lf, naive, pl):
    get_lf('units').filter(pl.col('session_id').is_in(naive['session_id'].implode())).collect().group_by('session_id').agg(pl.col('ccf_ap').is_not_null().any().alias('has_ccf'))


@app.cell
def _(get_lf):
    get_lf('performance').collect()['session_id'].n_unique(), get_lf('session').collect()['session_id'].n_unique()


@app.cell
def _(get_lf, pl):
    get_lf('session').select(
        pl.col('keywords').list.contains('context_naive').alias('naive'),
        pl.col('keywords').list.contains('context naive').alias('context naive'),
        pl.col('keywords').list.contains('brainwide_survey').alias('bws'),

    ).select(pl.col('bws') | pl.col('context naive')).collect()


@app.cell
def _(performance, pl):
    session_behavior = (
        performance
        .with_columns(
            pl.col("n_contingent_rewards").gt(10).alias("is_engaged_block"),
            pl.col("cross_modality_dprime").ge(1.0).alias("is_good_block"),
        )
        .group_by("session_id", "rewarded_modality")
        .agg(
            pl.col("is_engaged_block", "is_good_block").sum()
        )
        .group_by("session_id")
        .agg(
            pl.col("is_good_block").filter(pl.col("rewarded_modality") == "vis").first().alias("n_good_vis"),
            pl.col("is_good_block").filter(pl.col("rewarded_modality") == "aud").first().alias("n_good_aud"),
            pl.col("is_engaged_block").sum().alias("n_engaged"),
        )
        .filter(
            pl.col("n_good_aud").ge(2),
            pl.col("n_good_vis").ge(2),
            pl.col("n_engaged").ge(4),
        )
    ).collect().sort('session_id', descending=True)
    return (session_behavior,)


@app.cell
def _(session_behavior):
    session_behavior


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
