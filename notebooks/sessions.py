# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.16",
#     "npc-lims==0.1.198",
#     "polars==1.43.2",
#     "dr-datacube",
#     "altair==6.2.2",
# ]
#
# [tool.uv.sources]
# dr-bws = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():

    import altair as alt
    import dr_datacube as s
    import lazynwb
    import marimo as mo
    import npc_lims
    import polars as pl

    lazynwb.config.anon = True
    return alt, lazynwb, mo, npc_lims, pl, s


@app.cell(hide_code=True)
def _():
    session_ids_spreadsheet = [
        "668759_2023-07-11",
        "668759_2023-07-12",
        "668759_2023-07-13",
        "670181_2023-07-18",
        "670181_2023-07-19",
        "670181_2023-07-20",
        "670180_2023-07-25",
        "670180_2023-07-26",
        "670180_2023-07-27",
        "670248_2023-08-01",
        "670248_2023-08-02",
        "670248_2023-08-03",
        "660023_2023-08-08",
        "660023_2023-08-09",
        "666986_2023-08-14",
        "666986_2023-08-15",
        "666986_2023-08-16",
        "666986_2023-08-17",
        "662892_2023-08-21",
        "662892_2023-08-22",
        "662892_2023-08-23",
        "662892_2023-08-24",
        "668755_2023-08-28",
        "668755_2023-08-29",
        "668755_2023-08-30",
        "668755_2023-08-31",
        "667252_2023-09-25",
        "667252_2023-09-26",
        "667252_2023-09-27",
        "667252_2023-09-28",
        "674562_2023-10-02",
        "674562_2023-10-03",
        "674562_2023-10-04",
        "674562_2023-10-05",
        "681532_2023-10-16",
        "681532_2023-10-17",
        "681532_2023-10-18",
        "681532_2023-10-19",
        "686740_2023-10-23",
        "686740_2023-10-24",
        "686740_2023-10-25",
        "686740_2023-10-26",
        "664851_2023-11-13",
        "664851_2023-11-14",
        "664851_2023-11-15",
        "664851_2023-11-16",
        "690706_2023-11-27",
        "690706_2023-11-28",
        "690706_2023-11-29",
        "690706_2023-11-30",
        "686176_2023-12-04",
        "686176_2023-12-05",
        "686176_2023-12-06",
        "686176_2023-12-07",
        "676909_2023-12-11",
        "676909_2023-12-12",
        "676909_2023-12-13",
        "676909_2023-12-14",
        "702131_2024-02-26",
        "702131_2024-02-27",
        "702136_2024-03-04",
        "702136_2024-03-05",
        "702136_2024-03-06",
        "702136_2024-03-07",
        "703333_2024-04-08",
        "703333_2024-04-09",
        "703333_2024-04-10",
        "703333_2024-04-11",
        "699847_2024-04-15",
        "699847_2024-04-16",
        "699847_2024-04-17",
        "699847_2024-04-18",
        "703880_2024-04-15",
        "703880_2024-04-16",
        "703880_2024-04-17",
        "703880_2024-04-18",
        "703882_2024-04-22",
        "703882_2024-04-23",
        "703882_2024-04-24",
        "703882_2024-04-25",
        "706401_2024-04-22",
        "706401_2024-04-23",
        "706401_2024-04-24",
        "706401_2024-04-25",
        "708016_2024-04-29",
        "708016_2024-04-30",
        "708016_2024-05-01",
        "708016_2024-05-02",
        "712815_2024-05-20",
        "712815_2024-05-21",
        "712815_2024-05-22",
        "712815_2024-05-23",
        "726088_2024-06-17",
        "726088_2024-06-18",
        "726088_2024-06-20",
        "726088_2024-06-21",
        "714748_2024-06-24",
        "714748_2024-06-25",
        "714748_2024-06-26",
        "714748_2024-06-27",
        "714748_2024-06-28",
        "714753_2024-07-01",
        "714753_2024-07-02",
        "714753_2024-07-03",
        "714753_2024-07-05",
        "715710_2024-07-15",
        "715710_2024-07-16",
        "715710_2024-07-17",
        "715710_2024-07-18",
        "715710_2024-07-19",
        "713655_2024-08-05",
        "713655_2024-08-06",
        "713655_2024-08-07",
        "713655_2024-08-08",
        "713655_2024-08-09",
        "733780_2024-08-26",
        "733780_2024-08-27",
        "733780_2024-08-29",
        "733780_2024-08-30",
        "733780_2024-09-03",
        "733780_2024-09-04",
        "733780_2024-09-05",
        "733780_2024-09-06",
        "733891_2024-09-16",
        "733891_2024-09-17",
        "733891_2024-09-18",
        "733891_2024-09-19",
        "733891_2024-09-20",
        "737403_2024-09-24",
        "737403_2024-09-25",
        "737403_2024-09-26",
        "737403_2024-09-27",
        "741137_2024-10-08",
        "741137_2024-10-09",
        "741137_2024-10-10",
        "741137_2024-10-11",
        "741148_2024-10-15",
        "741148_2024-10-16",
        "741148_2024-10-17",
        "741148_2024-10-18",
        "742903_2024-10-21",
        "742903_2024-10-22",
        "742903_2024-10-23",
        "742903_2024-10-24",
        "724903_2024-10-25",
        "744740_2024-11-11",
        "744740_2024-11-12",
        "744740_2024-11-13",
        "744740_2024-11-14",
        "750329_2024-11-25",
        "750329_2024-11-26",
        "750329_2024-11-27",
        "743199_2024-12-03",
        "743199_2024-12-04",
        "743199_2024-12-05",
        "743199_2024-12-06",
        "761583_2024-12-16",
        "761583_2024-12-17",
        "761583_2024-12-18",
        "761583_2024-12-19",
        "761583_2024-12-20",
        "744279_2025-01-13",
        "744279_2025-01-14",
        "744279_2025-01-15",
        "744279_2025-01-16",
        "746439_2025-01-27",
        "746439_2025-01-28",
        "746439_2025-01-29",
        "746439_2025-01-30",
        "746439_2025-01-31",
        "759434_2025-02-03",
        "759434_2025-02-04",
        "759434_2025-02-05",
        "759434_2025-02-06",
        "759434_2025-02-07",
        "796012_2025-07-10",
        "796012_2025-07-11",
        "796012_2025-07-14",
        "796012_2025-07-15",
        "796848_2025-07-14",
        "796848_2025-07-15",
        "796848_2025-07-16",
        "796848_2025-07-17",
        "796847_2025-07-29",
        "796847_2025-07-30",
        "796847_2025-07-31",
        "796847_2025-08-01",
        "795555_2025-08-21",
        "795555_2025-08-22",
        "795555_2025-08-25",
        "795555_2025-08-26",
        "798632_2025-08-29",
        "798632_2025-09-02",
        "798632_2025-09-03",
        "798632_2025-09-04",
        "807082_2025-09-29",
        "807082_2025-09-30",
        "807082_2025-10-01",
        "807082_2025-10-02",
        "810752_2025-10-07",
        "810752_2025-10-08",
        "810752_2025-10-09",
        "810752_2025-10-10",
        "805752_2025-10-20",
        "805752_2025-10-21",
        "805752_2025-10-22",
        "805752_2025-10-23",
        "805752_2025-10-24",
        "810754_2025-10-27",
        "810754_2025-10-28",
        "810754_2025-10-29",
        "810754_2025-10-30",
        "810754_2025-10-31",
        "814666_2025-11-06",
        "814666_2025-11-07",
        "814666_2025-11-10",
        "814666_2025-11-11",
        "814669_2025-11-10",
        "814669_2025-11-11",
        "814669_2025-11-12",
        "814669_2025-11-13",
        "813586_2025-12-02",
        "813586_2025-12-03",
        "813586_2025-12-04",
        "813586_2025-12-05",
        "831994_2025-12-15",
        "831994_2025-12-16",
        "831994_2025-12-17",
        "834408_2025-12-17",
        "831994_2025-12-18",
        "834408_2025-12-18",
        "818720_2025-12-19",
        "834408_2025-12-19",
        "818720_2025-12-22",
        "834408_2025-12-22",
        "822061_2026-01-12",
        "822061_2026-01-13",
        "822061_2026-01-14",
        "822061_2026-01-15",
        "822061_2026-01-16",
        "836726_2026-02-11",
        "836726_2026-02-12",
        "836726_2026-02-13",
        "836729_2026-02-16",
        "836729_2026-02-17",
        "836729_2026-02-18",
        "835598_2026-02-23",
        "835598_2026-02-24",
        "835598_2026-02-25",
        "835598_2026-02-26",
        "835598_2026-02-27",
        "850215_2026-03-18",
        "850215_2026-03-19",
        "850215_2026-03-20",
        "850215_2026-03-23",
        "837959_2026-03-30",
        "837959_2026-03-31",
        "837959_2026-04-01",
        "837959_2026-04-02",
        "850217_2026-04-02",
        "837959_2026-04-03",
        "850217_2026-04-03",
        "850217_2026-04-06",
        "844567_2026-04-08",
        "847958_2026-04-13",
        "847958_2026-04-14",
        "847958_2026-04-15",
        "847958_2026-04-16",
        "847958_2026-04-17",
        "837230_2026-04-17",
        "837230_2026-04-20",
        "837230_2026-04-21",
        "837230_2026-04-22",
        "854390_2026-04-22",
        "854390_2026-04-23",
        "854390_2026-04-24",
        "854390_2026-04-27",
        "840160_2026-04-28",
        "836629_2026-04-28",
        "840160_2026-04-29",
        "836629_2026-04-29",
        "840160_2026-04-30",
        "836629_2026-04-30",
        "840160_2026-05-01",
        "836629_2026-05-01",
        "840160_2026-05-04",
        "841308_2026-05-05",
        "841308_2026-05-06",
        "841308_2026-05-07",
        "841308_2026-05-08",
        "841308_2026-05-11",
        "839146_2026-05-12",
        "839146_2026-05-13",
        "839146_2026-05-14",
        "859583_2026-05-14",
        "859583_2026-05-15",
        "839146_2026-05-15",
        "839146_2026-05-18",
        "859583_2026-05-19",
        "859583_2026-05-20",
        "856845_2026-05-27",
        "856845_2026-05-28",
        "856845_2026-05-29",
        "856845_2026-06-01",
        "856847_2026-06-02",
        "856847_2026-06-03",
        "856847_2026-06-04",
        "856847_2026-06-05",
        "852255_2026-06-16",
        "852255_2026-06-17",
        "852255_2026-06-18",
        "852255_2026-06-22",
        "852252_2026-06-24",
        "852252_2026-06-25",
        "852252_2026-06-26",
        "859582_2026-06-29",
        "859582_2026-06-30",
        "859582_2026-07-01",
        "859582_2026-07-02",
        "859584_2026-07-06",
        "859584_2026-07-07",
        "859584_2026-07-08",
        "859584_2026-07-09",
        "862025_2026-08-04",
        "862025_2026-08-05",
        "862025_2026-08-06",
        "857681_2026-08-13",
    ]
    return (session_ids_spreadsheet,)


@app.cell
def _(pl, session_ids_spreadsheet):
    spreadsheet_sessions = pl.DataFrame({"session_id": session_ids_spreadsheet})
    return (spreadsheet_sessions,)


@app.cell
def _(npc_lims):
    npc_lims_session_info = npc_lims.get_session_info(is_ephys=True)
    return (npc_lims_session_info,)


@app.cell
def _(npc_lims_session_info, pl, spreadsheet_sessions):
    _records = []
    for i in npc_lims_session_info:
        if i.project == "TempletonPilotSession":
            session_type = "templeton"
        elif i.session_kwargs.get("is_context_naive"):
            session_type = "naive"
        else:
            session_type = "brainwide"
        _records.append(
            dict(
                subject_id=i.subject.id,
                session_id=i.id.removesuffix("_0"),
                session_type=session_type,
                is_prod=i.session_kwargs.get("is_production") != False,
                is_ephys=True,
            )
        )
    npc_lims_sessions = spreadsheet_sessions.join(
        pl.DataFrame(_records), on="session_id", how="left"
    )
    npc_lims_sessions.group_by("session_type").len()
    return (npc_lims_sessions,)


@app.cell
def _(mo, npc_lims_sessions):
    mo.callout(
        value=f"{npc_lims_sessions['session_type'].is_null().sum()} sessions in spreadsheet but not in npc_lims",
        kind="warn",
    )


@app.cell
def _(npc_lims_sessions):
    npc_lims_sessions


@app.cell
def _(npc_lims_sessions, pl, s, session_ids_spreadsheet):
    session_ids_in_cache_v288 = list(
        p.stem
        for p in s.DatacubeConfig(use_cache=True, version="v0.0.288").nwb_dir.glob("*.nwb*")
        # 288 is naive-only
    )
    session_ids_in_cache_v289 = list(
        p.stem
        for p in s.DatacubeConfig(use_cache=True, version="v0.0.289").nwb_dir.glob("*.nwb*")
    )
    session_ids_in_datacube_v289 = (
        pl.read_parquet(
            (
                s.DatacubeConfig(use_cache=False, version="v0.0.289").asset_dir
                / "session_table.parquet"
            ).as_posix()
        )["session_id"]
        .sort()
        .to_list()
    )
    session_ids_in_datacube_v288 = (
        pl.read_parquet(
            (
                s.DatacubeConfig(use_cache=False, version="v0.0.288").asset_dir
                / "session_table.parquet"
            ).as_posix()
        )["session_id"]
        .sort()
        .to_list()
    )

    sessions = npc_lims_sessions.with_columns(
        in_cache_v288=pl.col("session_id").is_in(session_ids_in_cache_v288),
        in_cache_v289=pl.col("session_id").is_in(session_ids_in_cache_v289),
        in_datacube_v288=pl.col("session_id").is_in(session_ids_in_datacube_v288),
        in_datacube_v289=pl.col("session_id").is_in(session_ids_in_datacube_v289),
        in_npc_lims=pl.when(pl.col("session_type").is_not_null()).then(True).otherwise(False),
        in_spreadsheet=pl.col("session_id").is_in(session_ids_spreadsheet),
    )
    sessions
    return session_ids_in_cache_v289, session_ids_in_datacube_v289, sessions


@app.cell
def _(mo, pl, sessions):
    assert (
        sessions.filter(
            (pl.col("in_datacube_v288") & ~pl.col("in_datacube_v289")),
        )
    ).is_empty()
    mo.callout(value="all sessions in v288 datacube are in v289 datacube", kind="success")


@app.cell
def _(alt, sessions):
    _df = (
        sessions
        # .filter('is_prod')
        .unpivot(
            on=["in_npc_lims", "in_cache_v289", "in_datacube_v289", "in_spreadsheet"],
            index=["session_id", "subject_id", "session_type", "is_prod"],
            value_name="in_source",
            variable_name="source_type",
        )
    )

    _chart = (
        alt.Chart(_df)
        .mark_arc()
        .encode(
            color=alt.Color(field="in_source", type="nominal"),
            theta=alt.Theta(aggregate="count", type="quantitative"),
            row="session_type",
            column="source_type",
            tooltip=[
                alt.Tooltip(aggregate="count"),
                alt.Tooltip(aggregate="count"),
                alt.Tooltip(field="in_source"),
            ],
        )
        .properties(
            width=100,
            height=100,
        )
    )
    _chart


@app.cell
def _(
    lazynwb,
    mo,
    pl,
    s,
    session_ids_in_cache_v289,
    session_ids_in_datacube_v289,
):
    missing_from_datacube_v289 = set(session_ids_in_cache_v289) - set(
        session_ids_in_datacube_v289
    )
    _records = []
    for session_id in missing_from_datacube_v289:
        _record = {"session_id": session_id}
        try:
            units = lazynwb.scan_nwb(
                s.DatacubeConfig(use_cache=True, version="v0.0.289").nwb_dir
                / f"{session_id}.nwb",
                "units",
            )
        except Exception:
            _record["has_units"] = False
        else:
            _record["has_units"] = True
        _records.append(_record)
        # break
    assert not pl.DataFrame(_records)["has_units"].any()
    mo.callout(
        value=f"all {len(_records)} NWBs in cache but not in datacube are missing units (as expected)",
        kind="info",
    )


@app.cell
def _(mo):
    n_errors_during_caching = 21
    n_prod = 6
    mo.callout(
        value=f"there were {n_errors_during_caching} errors during caching, of which {n_prod} were production sessions (see https://codeocean.allenneuraldynamics.org/capsule/6743659/tree?results=8d2d3fe6-cb9e-4830-9362-dd54899eb9ba_computation)",
        kind="info",
    )


@app.cell
def _(pl, s):
    def session_table() -> pl.DataFrame:
        dfs = []
        for session_type in s.filter_functions():
            good = s.get_sessions(session_type).with_columns(pl.lit(True).alias("is_behavior_pass"))
            all = s.get_sessions(session_type, with_behavior_filter=False)
            dfs.extend([df.with_columns(session_type=pl.lit(session_type)) for df in (good, all)])
        df= (
            pl.concat(dfs, how='diagonal')
            .drop('keywords')
            .with_columns(
                pl.col("is_behavior_pass").fill_null(pl.lit(False)),
            )
            .sort("session_id")
        )
        assert df["session_type"].null_count() == 0
        assert df["is_behavior_pass"].null_count() == 0
        return df
    session_table()


@app.cell
def _(pl, s):
    (
        s.get_lf("performance")
        .join(s.get_sessions("templeton").lazy(), on="session_id", how="semi")
        .filter(
            pl.col("cross_modality_dprime").is_null(),
            pl.col("aud_dprime").gt(1.0) | pl.col("vis_dprime").gt(1.0),
        )
    ).collect()


if __name__ == "__main__":
    app.run()
