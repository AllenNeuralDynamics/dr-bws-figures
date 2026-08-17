# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.16",
#     "numpy==2.4.6",
#     "oursin==1.0.1",
#     "polars==1.43.2",
#     "dr-bws",
# ]
#
# [tool.uv.sources]
# dr-bws = { git = "https://github.com/AllenNeuralDynamics/dr-bws-figures" }
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _(points):
    import contextlib
    import pathlib
    import time

    import numpy as np
    import oursin as urchin
    import polars as pl

    from dr_bws.datacube import get_lf, get_sessions

    urchin.setup()
    urchin.ccf25.load()

    def clear_all():
        with contextlib.suppress(NameError):
            points.delete()
            urchin.particles.clear()
        with contextlib.suppress(NameError):
            for p in globals().get("probes", ()):
                p.delete()
            urchin.probes.clear()

    ccf_df = pl.read_csv(
        "https://raw.githubusercontent.com/cortex-lab/allenCCF/refs/heads/master/structure_tree_safe_2017.csv"
    ).with_columns(
        color_hex_triplet=pl.concat_str(
            pl.lit("#"), pl.col("color_hex_triplet").str.to_lowercase()
        )
    )

    # must wait for session to open in browser before continuing
    time.sleep(6)

    urchin.ccf25.grey.set_color("#000000")
    urchin.ccf25.grey.set_material("transparent-lit")
    urchin.ccf25.grey.set_alpha(0.15)
    urchin.ccf25.grey.set_visibility(True)
    return get_lf, get_sessions, np, pathlib, pl, urchin


@app.cell
def _(get_sessions):
    sessions = get_sessions("brainwide")
    return (sessions,)


@app.cell
def _(get_lf, pl, sessions):
    probes = (
        get_lf("electrodes")
        .join(sessions.lazy(), on="session_id", how="semi")
        .collect()
        .drop_nulls(["x", "y", "z"])
        .with_columns(
            pl.when(
                pl.col("channel").eq(pl.col("channel").min().over("session_id", "group_name"))
            )
            .then(pl.lit("tip"))
            .when(pl.col("channel").eq(pl.col("channel").max().over("session_id", "group_name")))
            .then(pl.lit("base"))
            .alias("point")
        )
        .drop_nulls("point")
        .select("point", "x", "y", "z", "session_id", "group_name")
    )
    return (probes,)


@app.cell
def draw_probe_lines(np, pl, probes, urchin):
    # Render CCF probe trajectories with Urchin's thin probe-line primitive.
    # Electrode CCF columns are (x=AP, y=DV, z=ML); Urchin expects (AP, ML, DV).
    for _existing_probe in list(urchin.probes.probes):
        _existing_probe.delete()
    urchin.probes.probes.clear()

    _probe_geometry = []
    for _probe in probes.partition_by("session_id", "group_name", maintain_order=True):
        _base_row = _probe.filter(pl.col("point") == "base").row(0, named=True)
        _tip_row = _probe.filter(pl.col("point") == "tip").row(0, named=True)

        _base_ccf = np.array([_base_row["x"], _base_row["z"], _base_row["y"]])
        _tip_ccf = np.array([_tip_row["x"], _tip_row["z"], _tip_row["y"]])
        _shaft = _base_ccf - _tip_ccf
        _horizontal = np.hypot(_shaft[0], _shaft[1])

        _probe_geometry.append(
            (
                _tip_ccf.tolist(),
                [
                    np.degrees(np.arctan2(_shaft[1], _shaft[0])),
                    np.degrees(np.arctan2(-_shaft[2], _horizontal)),
                    0.0,
                ],
                [0.01, np.linalg.norm(_shaft) / 1000, 0.01],
            )
        )

    probe_lines = urchin.probes.create(len(_probe_geometry))
    urchin.probes.set_positions(probe_lines, [_item[0] for _item in _probe_geometry])
    urchin.probes.set_angles(probe_lines, [_item[1] for _item in _probe_geometry])
    urchin.probes.set_scales(probe_lines, [_item[2] for _item in _probe_geometry])
    urchin.probes.set_colors(probe_lines, ["#000000"] * len(probe_lines))


@app.cell
def _():
    return


@app.cell
async def save_snapshot(pathlib, urchin):
    # Save the current Urchin view as a PNG snapshot.
    snapshot_path = pathlib.Path(__file__).resolve().parent / "probe_tracks_snapshot.png"
    await urchin.camera.main.screenshot(
        size=[1600, 1200],
        filename=str(snapshot_path),
    )
    snapshot_path


if __name__ == "__main__":
    app.run()
