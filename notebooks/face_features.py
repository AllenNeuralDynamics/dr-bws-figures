# /// script
# dependencies = [
#     "altair==6.2.2",
#     "dr-datacube",
#     "marimo",
#     "polars==1.43.2",
# ]
# requires-python = "<3.12"
#
# [tool.uv.sources]
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib
    import pprint

    import lazynwb
    import matplotlib.pyplot as plt
    import polars as pl
    from dr_datacube import (
        datacube_config,
        get_session_ids_from_github,
        list_nwb_sources,
        on_codeocean,
    )

    lazynwb.config.anon = True
    datacube_config.use_cache = not on_codeocean()
    return (
        get_session_ids_from_github,
        lazynwb,
        list_nwb_sources,
        pathlib,
        pl,
        plt,
        pprint,
    )


@app.cell
def _(get_session_ids_from_github, list_nwb_sources, pathlib):
    session_ids = get_session_ids_from_github("brainwide", with_behavior_filter=True)
    nwb_sources = [p for p in list_nwb_sources() if pathlib.Path(p).stem in session_ids]
    print(len(session_ids))
    return (nwb_sources,)


@app.cell
def _(pl):
    # polars expressions to derive ids from lazynwb tables
    session_id = pl.col("_nwb_path").str.split("/").list.get(-1).str.strip_suffix(".nwb").alias("session_id")
    subject_id = session_id.str.split("_").list.get(0).alias("subject_id")
    return session_id, subject_id


@app.cell
def _(lazynwb, nwb_sources, pprint, session_id, subject_id):
    eye_lf = (
        lazynwb.scan_nwb(nwb_sources, table_path="processing/behavior/eye_tracking")
        .with_columns(session_id, subject_id)
    )
    pprint.pprint(eye_lf.collect_schema())
    return (eye_lf,)


@app.cell
def _(eye_lf):
    eye_df = (
        eye_lf
        .select(
            "session_id", "subject_id", "pupil_area", "timestamps", "pupil_is_bad_frame"
        )
        .collect()
    )
    return (eye_df,)


@app.cell
def _(eye_df, pl, plt):
    med_filter_size = 10  # frames

    _df = (
        eye_df
        .filter(
            pl.col("session_id") == pl.col("session_id").first(), # or use .unique().sample() to get random session
            ~pl.col("pupil_is_bad_frame"),
        )
        .with_columns(
            pl.col("pupil_area").rolling_median(med_filter_size).alias("filtered_pupil_area"),
        )
    )
    fig, ax = plt.subplots()
    ax.plot(_df["timestamps"], _df["pupil_area"], lw=.5)
    ax.plot(_df["timestamps"], _df["filtered_pupil_area"], lw=.5)
    ax.set_ylim(0, 20_000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pupil area (pixels)")


@app.cell
def _(lazynwb, nwb_sources, pprint):
    face_lf = lazynwb.scan_nwb(nwb_sources, table_path="processing/behavior/lp_front_camera")
    pprint.pprint(face_lf.collect_schema())


if __name__ == "__main__":
    app.run()
