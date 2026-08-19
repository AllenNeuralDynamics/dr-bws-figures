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

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import lazynwb
    from dr_bws.datacube import (
        datacube_config,
        list_nwb_sources,
        get_session_ids_from_github,
    )

    datacube_config.use_cache = True
    lazynwb.config.anon = True
    return get_session_ids_from_github, lazynwb, list_nwb_sources, pathlib


@app.cell
def _(get_session_ids_from_github, list_nwb_sources, pathlib):
    nwb_sources = [
        p
        for p in list_nwb_sources()
        if pathlib.Path(p).stem
        in get_session_ids_from_github("brainwide", with_behavior_filter=True)
    ]
    return (nwb_sources,)


@app.cell
def _(lazynwb, nwb_sources):
    eye = (
        lazynwb.scan_nwb(nwb_sources, table_path="processing/behavior/eye_tracking")
        .select("pupil_area", "timestamps", "pupil_is_bad_framepupil_is_bad_frame")
        .collect()
    )
    eye
    return


@app.cell
def _(lazynwb, nwb_sources):
    face = lazynwb.scan_nwb(
        nwb_sources, table_path="processing/behavior/lp_front_camera"
    ).collect_schema()
    face
    return


if __name__ == "__main__":
    app.run()
