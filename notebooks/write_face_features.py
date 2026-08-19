# /// script
# dependencies = [
#     "altair==6.2.2",
#     "dr-bws",
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
    import pprint

    import lazynwb
    import polars as pl
    import matplotlib.pyplot as plt
    from dr_bws.datacube import (
        on_codeocean,
        datacube_config,
        list_nwb_sources,
        get_session_ids_from_github,
    )

    lazynwb.config.anon = True
    datacube_config.use_cache = not on_codeocean()
    return get_session_ids_from_github, lazynwb, list_nwb_sources, pathlib, pl


@app.cell
def _(get_session_ids_from_github, list_nwb_sources, pathlib):
    session_ids = get_session_ids_from_github(session_type=None, with_behavior_filter=False)
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
def _(lazynwb, nwb_sources, pl, session_id, subject_id):
    eye = (
        lazynwb.scan_nwb(nwb_sources, table_path="processing/behavior/eye_tracking")
        .with_columns(session_id, subject_id)
        .drop(pl.selectors.starts_with("_"))
        .collect()
    )
    eye.write_parquet("s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.289/consolidated/eye_tracking.parquet")
    return


@app.cell
def _(lazynwb, nwb_sources, pl, session_id, subject_id):
    face = (
        lazynwb.scan_nwb(nwb_sources, table_path="processing/behavior/lp_front_camera")
        .with_columns(session_id, subject_id)
        .drop(pl.selectors.starts_with("_"))
        .collect()
    )
    face.write_parquet("s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.289/consolidated/lp_front_camera.parquet")
    return


if __name__ == "__main__":
    app.run()
