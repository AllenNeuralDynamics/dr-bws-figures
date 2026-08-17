# /// script
# requires-python = "<3.12"
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
app = marimo.App(width="full")


@app.cell
def _():
    import contextlib
    import pathlib
    import time

    import oursin as urchin
    import polars as pl

    from dr_bws.datacube import datacube_config, get_lf, get_session_ids_from_github, on_codeocean

    datacube_config.use_cache = True

    asset_dir = (
        pathlib.Path(__file__).resolve().parent
        if not on_codeocean()
        else pathlib.Path("/root/capsule/results")
    )
    return (
        asset_dir,
        contextlib,
        get_lf,
        get_session_ids_from_github,
        pl,
        time,
        urchin,
    )


@app.cell
def _(contextlib, pl, time, urchin):
    urchin.setup()
    urchin.ccf25.load()

    def clear_all():
        with contextlib.suppress(NameError):
            urchin.particles.clear()
        with contextlib.suppress(NameError):
            for p in globals().get("probes", ()):
                p.delete()
            urchin.probes.clear()
            
    # must wait for session to open in browser before continuing
    time.sleep(6)

    urchin.ccf25.grey.set_color("#000000")
    urchin.ccf25.grey.set_material("transparent-lit")
    urchin.ccf25.grey.set_alpha(0.15)
    urchin.ccf25.grey.set_visibility(True)


@app.cell
def _(get_lf, get_session_ids_from_github, pl):
    electrodes_df = (
        get_lf("electrodes")
        .filter(
            pl.col("session_id").is_in(get_session_ids_from_github("brainwide")),
            ~pl.col('structure').is_in(['out of brain', 'undefined', 'root']),
        )
        .join(
            get_lf("units").filter(pl.col("decoder_label").ne("noise")), 
            left_on=["session_id", "group_name", "channel"],
            right_on=["session_id", "electrode_group_name", "peak_channel"], 
            how="semi",
        )
        .drop_nulls(["x", "y", "z"])
        .collect()
        # label the min and max channels as tip/base:
        .with_columns(
            pl.when(
                pl.col("channel").eq(pl.col("channel").min().over("session_id", "group_name"))
            )
            .then(pl.lit("tip"))
            .when(pl.col("channel").eq(pl.col("channel").max().over("session_id", "group_name")))
            .then(pl.lit("base"))
            .alias("loc")
        )
        # keep only tip/base
        .drop_nulls("loc")
        # find distance between tip/base in each dimension, for each probe
        .sort('loc')
        .group_by('session_id', 'group_name', maintain_order=True)
        .agg(
            pl.col('loc', 'x', 'y', 'z'),
            dx=pl.col('x').get(1) - pl.col('x').get(0),
            dy=pl.col('y').get(1) - pl.col('y').get(0),
            dz=pl.col('z').get(1) - pl.col('z').get(0),
        )
        .with_columns(
            # phi is azimuth, cw rotation from AP axis (looking top down on AP/ML plane) 0 deg is
            # pointing posterior
            phi=pl.arctan2(-pl.col('dz'), 'dx').degrees()
        )
        .with_columns(
            # theta is clockwise elevation from horizontal in the azimuthal plane (0 deg is horizontal,
            # +90 straight up)
            theta=pl.arctan2(-pl.col('dy'), (pl.col('dx') ** 2 + pl.col('dz') ** 2) ** 0.5).degrees()
        )
        # .select("loc", "x", "y", "z", "session_id", "group_name")
        .explode(['loc', 'x', 'y', 'z'])
        .sort('session_id', 'group_name', 'loc')
        .rename({'x': 'ap', 'y': 'dv', 'z': 'ml'})
    )
    electrodes_df
    return (electrodes_df,)


@app.cell
def _(electrodes_df, pl):
    _df = (
        electrodes_df
        .with_columns(pl.col('session_id').str.split('_').list.get(0).alias('subject_id'))
    )
    print(f"n subjects: {_df['subject_id'].n_unique()}")
    print(f"n sessions: {_df['session_id'].n_unique()}")
    print(f"n insertions: {_df.unique(['session_id', 'group_name']).height}")


@app.cell
def _(electrodes_df, urchin):
    particles = urchin.particles.ParticleSystem(n=len(electrodes_df))
    return (particles,)


@app.cell
def _(electrodes_df, particles, pl):
    # configure appearance of points here:
    points_df = (
        electrodes_df
        .with_columns(
            (
                # size:
                pl.when(pl.col('loc') == 'surface').then(pl.lit(0))
                .otherwise(pl.lit(0))
                .alias('size')
            ),
            (
                # color:
                pl.when(pl.col('loc') == 'tip').then(pl.lit('#ff0000'))
                .otherwise(pl.lit("#00ffff"))
                .alias('color')
            ),
        )
    )
    particles.set_positions(points_df.select(['ap', 'ml', 'dv']).to_numpy().tolist())
    particles.set_colors(points_df['color'].to_list())
    particles.set_sizes(points_df['size'].to_list())
    points_df


@app.cell
def _(electrodes_df, pl, urchin):
    probes: list[urchin.probes.Probe] = urchin.probes.create(len(electrodes_df.filter(pl.col('loc') == 'tip')))
    return (probes,)


@app.cell
def _(electrodes_df, pl, probes: "list[urchin.probes.Probe]", urchin):
    tip_df = electrodes_df.filter(pl.col('loc') == 'tip')
    urchin.probes.set_colors(probes, ["#424242"] * len(tip_df))
    # urchin.probes.set_colors(probes, ["#A0A0A0"] * len(tip_df))
    urchin.probes.set_positions(probes, tip_df['ap', 'ml', 'dv'].to_numpy().tolist()) #setting the positions within the renderer
    urchin.probes.set_scales(probes, [(0.01, -3.840, 0.01)] * len(tip_df))
    urchin.probes.set_angles(
        probes, 
        (
            tip_df
            .with_columns(
                pl.lit(0).alias('roll'),
            )   
            .select(['phi', 'theta', 'roll'])
            .to_numpy().tolist()
        )
    )


@app.cell
async def _(asset_dir, urchin):
    # more dorsal:
    urchin.camera.main.set_rotation([20,39, 225])
    urchin.camera.main.set_zoom(40)
    # more medial:
    urchin.camera.main.set_rotation([30, 55, 225])
    urchin.camera.main.set_zoom(45)

    urchin.camera.main.set_mode('perspective')
    snapshot_path = asset_dir / "urchin.png"
    await urchin.camera.main.screenshot(
        size=[2200, 1800],
        filename=str(snapshot_path),
    )
    await urchin.camera.main.screenshot()


if __name__ == "__main__":
    app.run()
