# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "aind-low-point",
#     "boto3>=1.35,<2",
#     "dr-datacube",
#     "dracopy>=2,<3",
#     "marimo>=0.23.16",
#     "numpy>=1.20,<3",
#     "polars==1.43.2",
# ]
#
# [tool.uv.sources]
# aind-low-point = { git = "https://github.com/AllenNeuralDynamics/aind-low-point" }
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///

"""Marimo notebook for the brainwide electrode summary.

The notebook is fully programmatic: its cells load the datacube, build a
low-point scene, and save a PNG through PyVista's off-screen renderer. Running
the file directly with ``uv run --script`` executes ``app.run()`` without
starting a browser. Opening it with ``marimo edit`` gives the normal Marimo
notebook UI.

"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import io
    import json
    from pathlib import Path

    # The render cell explicitly selects off-screen rendering, so a global
    # PYVISTA_OFF_SCREEN setting is unnecessary here. Keeping that choice
    # local lets the notebook display the resulting PNG in interactive mode.
    import boto3
    import DracoPy
    import marimo as mo
    import numpy as np
    import polars as pl
    import pyvista as pv
    import trimesh
    from botocore import UNSIGNED
    from botocore.config import Config

    from aind_low_point.assets import AssetCatalog, AssetSpec
    from aind_low_point.core import (
        AffineTransform,
        Material,
        MeshTransformable,
        PointsTransformable,
        TransformChain,
    )
    from aind_low_point.planning import Kinematics, PlanningState
    from aind_low_point.pyvista_backend import PyVistaBackend
    from aind_low_point.rendering import RendererAdapter
    from aind_low_point.runtime.loaders import load_geometry
    from aind_low_point.scene import NodeInstance, Scene

    return (
        AffineTransform,
        AssetCatalog,
        AssetSpec,
        Config,
        DracoPy,
        Kinematics,
        Material,
        MeshTransformable,
        NodeInstance,
        Path,
        PlanningState,
        PointsTransformable,
        PyVistaBackend,
        RendererAdapter,
        Scene,
        TransformChain,
        UNSIGNED,
        boto3,
        csv,
        io,
        json,
        load_geometry,
        mo,
        np,
        pl,
        pv,
        trimesh,
    )


@app.cell
def _(
    AffineTransform,
    AssetCatalog,
    AssetSpec,
    Kinematics,
    Material,
    MeshTransformable,
    NodeInstance,
    Path,
    PlanningState,
    PointsTransformable,
    PyVistaBackend,
    RendererAdapter,
    Scene,
    TransformChain,
    load_geometry,
    mo,
    np,
    pl,
    pv,
    trimesh,
):
    PROBE_LENGTH_MM = 3.840
    PROBE_RADIUS_MM = 0.0035
    PROBE_SURFACE_CLEARANCE_MM = 0.6

    def load_electrodes(
        *,
        get_lf,
        get_session_ids_from_github,
        session_type: str = "brainwide",
    ) -> pl.DataFrame:
        """Load and reduce electrode rows to probe tips and bases."""
        session_ids = get_session_ids_from_github(session_type)
        return (
            get_lf("electrodes")
            .filter(
                pl.col("session_id").is_in(session_ids),
                ~pl.col("structure").is_in(["out of brain", "undefined", "root"]),
            )
            .join(
                get_lf("units").filter(pl.col("decoder_label").ne("noise")),
                left_on=["session_id", "group_name", "channel"],
                right_on=["session_id", "electrode_group_name", "peak_channel"],
                how="semi",
            )
            .drop_nulls(["x", "y", "z"])
            .collect()
            .with_columns(
                pl.when(
                    pl.col("channel").eq(
                        pl.col("channel").min().over("session_id", "group_name")
                    )
                )
                .then(pl.lit("tip"))
                .when(
                    pl.col("channel").eq(
                        pl.col("channel").max().over("session_id", "group_name")
                    )
                )
                .then(pl.lit("base"))
                .alias("loc")
            )
            .drop_nulls("loc")
            .sort("loc")
            .group_by("session_id", "group_name", maintain_order=True)
            .agg(
                pl.col("loc", "x", "y", "z"),
                dx=pl.col("x").get(1) - pl.col("x").get(0),
                dy=pl.col("y").get(1) - pl.col("y").get(0),
                dz=pl.col("z").get(1) - pl.col("z").get(0),
            )
            .with_columns(
                phi=pl.arctan2(-pl.col("dz"), "dx").degrees(),
                theta=pl.arctan2(
                    -pl.col("dy"),
                    (pl.col("dx") ** 2 + pl.col("dz") ** 2) ** 0.5,
                ).degrees(),
            )
            .explode(["loc", "x", "y", "z"], empty_as_null=True)
            .sort("session_id", "group_name", "loc")
            # Datacube electrode coordinates arrive as x=AP, y=DV, z=ML.
            .rename({"x": "ap", "y": "dv", "z": "ml"})
        )

    def points_in_scene_mm(
        df: pl.DataFrame,
        *,
        ml_midline_mm: float,
    ) -> np.ndarray:
        """Convert electrode AP/DV/ML micrometres to atlas ML/DV/AP mm."""
        # Datacube/Pinpoint ML starts on the anatomical left, whereas the raw
        # BrainGlobe ASR atlas starts on the right. Reflect ML about the atlas
        # midline while swapping the source AP and ML axes.
        points = df.select(["ml", "dv", "ap"]).to_numpy().astype(np.float64) / 1000.0
        points[:, 0] = 2.0 * ml_midline_mm - points[:, 0]
        return points

    def probe_rotation(tip_to_base: np.ndarray) -> np.ndarray:
        """Rotate local probe -Z onto a measured tip-to-base vector."""
        direction = np.asarray(tip_to_base, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("Cannot orient a probe with coincident tip and base")
        matrix = trimesh.geometry.align_vectors([0.0, 0.0, -1.0], direction / norm)
        return matrix[:3, :3]

    def probe_length_to_surface(
        tip: np.ndarray,
        tip_to_base: np.ndarray,
        brain: trimesh.Trimesh,
        *,
        clearance_mm: float = PROBE_SURFACE_CLEARANCE_MM,
    ) -> float:
        """Find the probe length needed to end at a fixed surface clearance."""
        direction = np.asarray(tip_to_base, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        locations, _, _ = brain.ray.intersects_location(
            ray_origins=np.asarray([tip]),
            ray_directions=np.asarray([direction]),
            multiple_hits=True,
        )
        distances = np.dot(locations - tip, direction)
        positive_distances = distances[distances > 1e-6]
        if positive_distances.size == 0:
            raise ValueError("Could not find the brain surface along a probe direction")
        return float(positive_distances.min() + clearance_mm)

    def make_probe_mesh(length_mm: float = PROBE_LENGTH_MM) -> trimesh.Trimesh:
        """Make a thin mesh whose local tip is at the origin."""
        mesh = trimesh.creation.cylinder(
            radius=PROBE_RADIUS_MM,
            height=length_mm,
            sections=8,
        )
        mesh.apply_translation([0.0, 0.0, -length_mm / 2.0])
        return mesh

    def build_scene(
        electrodes_df: pl.DataFrame,
        brain: Path | trimesh.Trimesh | None = None,
        *,
        normalize_probe_lengths: bool = True,
        surface_clearance_mm: float = PROBE_SURFACE_CLEARANCE_MM,
    ):
        """Build a low-point scene from electrode coordinates.

        When ``normalize_probe_lengths`` is enabled, each probe ends at the
        same clearance from the brain surface along its measured insertion
        direction.
        """
        tip_df = electrodes_df.filter(pl.col("loc") == "tip")
        base_df = electrodes_df.filter(pl.col("loc") == "base")
        if tip_df.height == 0 or base_df.height == 0:
            raise ValueError("The electrode table did not contain tip/base pairs")

        if brain is None:
            raise ValueError("Brain geometry is required to resolve the atlas midline")
        if isinstance(brain, trimesh.Trimesh):
            brain_geometry = brain
        else:
            brain_loader = (
                "trimesh"
                if brain.suffix.lower() in {".obj", ".ply", ".stl", ".glb", ".gltf"}
                else "sitk_volume"
            )
            brain_geometry = load_geometry(brain, loader=brain_loader)
        if not isinstance(brain_geometry, trimesh.Trimesh):
            raise TypeError(
                f"Brain loader returned {type(brain_geometry).__name__}, not a mesh"
            )

        ml_midline_mm = float(
            brain_geometry.metadata.get(
                "pinpoint_ml_midline_mm",
                brain_geometry.bounds[:, 0].mean(),
            )
        )

        assets: dict[str, AssetSpec] = {}
        scene = Scene()

        # PyVista point actors have one material per actor. Render only the
        # magenta bases; tip coordinates are still used to position the probes.
        for loc, df, color in (("base", base_df, "#000000"),):
            asset_key = f"electrodes:{loc}"
            assets[asset_key] = AssetSpec(
                key=asset_key,
                kind="points",
                default_material=Material(
                    name=f"electrodes-{loc}",
                    color_hex_str=color,
                    point_size=10.0,
                ),
                points=PointsTransformable(
                    points_in_scene_mm(df, ml_midline_mm=ml_midline_mm)
                ),
            )
            scene.upsert(
                NodeInstance(
                    key=asset_key,
                    asset_key=asset_key,
                    tags={"electrodes", loc},
                )
            )

        # The measured tip-to-base direction avoids Urchin's azimuth/elevation
        # convention and keeps coordinates in the scene's ML/DV/AP order.
        grouped = electrodes_df.partition_by(
            ["session_id", "group_name"], maintain_order=True
        )
        for insertion in grouped:
            tip = points_in_scene_mm(
                insertion.filter(pl.col("loc") == "tip"),
                ml_midline_mm=ml_midline_mm,
            )[0]
            base = points_in_scene_mm(
                insertion.filter(pl.col("loc") == "base"),
                ml_midline_mm=ml_midline_mm,
            )[0]
            session_id = insertion["session_id"][0]
            group_name = insertion["group_name"][0]
            tip_to_base = base - tip
            probe_length = (
                probe_length_to_surface(
                    tip,
                    tip_to_base,
                    brain_geometry,
                    clearance_mm=surface_clearance_mm,
                )
                if normalize_probe_lengths
                else PROBE_LENGTH_MM
            )
            probe_asset_key = f"probe:{session_id}:{group_name}"
            assets[probe_asset_key] = AssetSpec(
                key=probe_asset_key,
                kind="mesh",
                default_material=Material(
                    name="probe",
                    color_hex_str="#424242",
                    opacity=1.0,
                ),
                mesh=MeshTransformable(make_probe_mesh(probe_length)),
            )
            node_key = f"probe:{session_id}:{group_name}"
            scene.upsert(
                NodeInstance(
                    key=node_key,
                    asset_key=probe_asset_key,
                    tags={"probe", "electrode"},
                    transform=TransformChain.new(
                        [
                            AffineTransform(
                                rotation=probe_rotation(tip_to_base),
                                translation=tip,
                            )
                        ]
                    ),
                )
            )

        brain_key = "brain"
        assets[brain_key] = AssetSpec(
            key=brain_key,
            kind="mesh",
            default_material=Material(
                name="brain",
                color_hex_str="#a5a5a5",
                opacity=0.15,
            ),
            mesh=MeshTransformable(brain_geometry),
        )
        scene.upsert(NodeInstance(key=brain_key, asset_key=brain_key, tags={"brain"}))

        return AssetCatalog(assets=assets), scene

    def render(
        electrodes_df: pl.DataFrame,
        output: Path,
        *,
        brain: Path | trimesh.Trimesh | None = None,
        width: int = 2000,
        height: int = 2000,
        normalize_probe_lengths: bool = True,
        surface_clearance_mm: float = PROBE_SURFACE_CLEARANCE_MM,
        camera_zoom: float = 1.25,
    ):
        """Render named anatomical views and save one PNG per view."""
        catalog, scene = build_scene(
            electrodes_df,
            brain=brain,
            normalize_probe_lengths=normalize_probe_lengths,
            surface_clearance_mm=surface_clearance_mm,
        )
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(width, height),
        )
        # Keep a white render background for contrast; screenshot export below
        # writes it as transparent pixels.
        plotter.set_background("white")

        adapter = RendererAdapter(
            backend=PyVistaBackend(plotter=plotter),
            scene=scene,
            assets=catalog,
        )
        plan = PlanningState(kinematics=Kinematics(), probes={})
        adapter.build(plan)

        output.parent.mkdir(parents=True, exist_ok=True)

        def set_sagittal_view():
            plotter.view_yz()
            plotter.camera.roll = 180.0

        def set_coronal_view():
            # View from the anterior (low-AP) side, looking posteriorly.
            plotter.view_xy(negative=True)
            plotter.camera.roll = 180.0

        def set_isometric_view():
            # broken
            plotter.view_isometric()
            plotter.camera.azimuth = 0
            plotter.camera.elevation = 0
            plotter.camera.roll = 0

        # These view functions are chosen from the rendered atlas orientation:
        # XZ is dorsal, YZ is sagittal, and XY is coronal. The sagittal view
        # and coronal view need a half-turn to place dorsal at the top.
        views = {
            "dorsal": lambda: plotter.view_xz(),
            "sagittal": set_sagittal_view,
            "coronal": set_coronal_view,
            # "isometric": set_isometric_view,
        }
        snapshots = []
        for view_name, set_view in views.items():
            set_view()
            plotter.reset_camera()
            plotter.camera.zoom(camera_zoom)
            plotter.render()
            view_output = output.with_name(f"{output.stem}_{view_name}{output.suffix}")
            plotter.screenshot(str(view_output), transparent_background=True)
            snapshots.append(view_output)

        plotter.close()

        if mo.running_in_notebook():
            return mo.vstack(
                [
                    mo.image(path, alt=f"Low-point {name} view")
                    for name, path in zip(views, snapshots)
                ]
            )
        return snapshots

    return load_electrodes, render


@app.cell
def _(Config, DracoPy, UNSIGNED, boto3, csv, io, json, np, trimesh):
    PINPOINT_ATLAS_SOURCES = {
        "brainglobe": ("brainglobe", "atlas"),
        "allenInstitute": ("aind-scratch-data", "pinpoint-atlases"),
    }
    DEFAULT_ATLAS_SOURCE_BY_NAME = {"qiu2018_mouse": "allenInstitute"}

    def load_pinpoint_atlas_mesh(
        atlas_name: str = "allen_mouse",
        *,
        source_name: str | None = None,
        resolution_um: str = "25",
        version: str = "3_0",
    ) -> trimesh.Trimesh:
        """Load a Pinpoint root mesh from a public S3 atlas anonymously.

        Pinpoint publishes BrainGlobe-compatible manifests and Draco meshes in
        public S3 buckets. The mesh coordinates are nanometres in scene order
        (ML/DV/AP); low-point stores scene coordinates in millimetres in the
        same order.
        """
        source_name = source_name or DEFAULT_ATLAS_SOURCE_BY_NAME.get(
            atlas_name, "allenInstitute"
        )
        source_name = {
            "allen_institute": "allenInstitute",
            "allen": "allenInstitute",
        }.get(source_name, source_name)
        try:
            bucket, prefix = PINPOINT_ATLAS_SOURCES[source_name]
        except KeyError as exc:
            valid_sources = ", ".join(PINPOINT_ATLAS_SOURCES)
            raise ValueError(
                f"Unknown Pinpoint atlas source {source_name!r}; "
                f"choose one of: {valid_sources}"
            ) from exc

        resolution_token = str(resolution_um).removesuffix(".0")
        variant = f"{atlas_name}_{resolution_token}um/{version}"
        manifest_key = f"{prefix}/atlases/{variant}/manifest.json"

        # UNSIGNED is important here: notebook users should not need AWS
        # credentials, and this matches Pinpoint's public S3 URL access.
        s3 = boto3.client(
            "s3",
            region_name="us-west-2",
            config=Config(signature_version=UNSIGNED),
        )

        def read_object(key: str) -> bytes:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

        manifest = json.loads(read_object(manifest_key))
        terminology_location = manifest["terminology"]["location"]
        annotation_location = manifest["annotation_set"]["location"]
        terminology_key = f"{prefix}{terminology_location}"
        terminology = csv.DictReader(
            io.StringIO(read_object(terminology_key + "/terminology.csv").decode())
        )
        root_rows = [row for row in terminology if not row["parent_identifier"]]
        if len(root_rows) != 1:
            raise ValueError(
                f"Expected one root structure in {terminology_key}, "
                f"found {len(root_rows)}"
            )

        root_identifier = root_rows[0]["identifier"]
        mesh_key = (
            f"{prefix}{annotation_location}"
            f"/annotations.precomputed/mesh/{root_identifier}"
        )
        decoded = DracoPy.decode(read_object(mesh_key))
        vertices_mm = np.asarray(decoded.points, dtype=np.float64) * 1e-6
        # Pinpoint flips the Draco winding when moving its right-handed atlas
        # geometry into the left-handed scene coordinate system.
        faces = np.asarray(decoded.faces, dtype=np.int64)[:, [0, 2, 1]]
        mesh = trimesh.Trimesh(vertices=vertices_mm, faces=faces, process=False)
        # BrainGlobe shape/resolution are AP/DV/ML. Preserve the exact atlas
        # midline for converting Pinpoint's opposite-origin ML coordinates.
        mesh.metadata["pinpoint_ml_midline_mm"] = (
            float(manifest["shape"][2]) * float(manifest["resolution"][2]) / 2000.0
        )
        trimesh.repair.fix_normals(mesh)
        return mesh

    return (load_pinpoint_atlas_mesh,)


@app.cell
def _(load_electrodes):
    from dr_datacube import (
        datacube_config,
        get_lf,
        get_session_ids_from_github,
        on_codeocean,
    )

    datacube_config.use_cache = True
    electrodes_df = load_electrodes(
        get_lf=get_lf,
        get_session_ids_from_github=get_session_ids_from_github,
        session_type="brainwide",
    )
    return electrodes_df, on_codeocean


@app.cell
def _(electrodes_df):
    _subject_count = electrodes_df["session_id"].str.split("_").list.get(0).n_unique()
    _session_count = electrodes_df["session_id"].n_unique()
    _insertion_count = electrodes_df.unique(["session_id", "group_name"]).height
    print(f"n subjects: {_subject_count}")
    print(f"n sessions: {_session_count}")
    print(f"n insertions: {_insertion_count}")
    print(f"n electrodes: {electrodes_df.height}")
    return


@app.cell
def _(Path, load_pinpoint_atlas_mesh, on_codeocean):
    _results_dir = (
        Path("/root/capsule/results")
        if on_codeocean()
        else Path(__file__).resolve().parent
    )
    output = _results_dir / "low_point.png"
    atlas_name = "allen_mouse"
    atlas_source = "brainglobe"
    atlas_resolution_um = "25"
    atlas_version = "3_0"
    brain = load_pinpoint_atlas_mesh(
        atlas_name,
        source_name=atlas_source,
        resolution_um=atlas_resolution_um,
        version=atlas_version,
    )
    width = 2000
    height = 2000
    normalize_probe_lengths = True
    camera_zoom = 1.25
    return (
        atlas_name,
        atlas_resolution_um,
        atlas_source,
        atlas_version,
        brain,
        height,
        camera_zoom,
        normalize_probe_lengths,
        output,
        width,
    )


@app.cell
def _(
    Path,
    atlas_name,
    atlas_resolution_um,
    atlas_source,
    atlas_version,
    brain,
    electrodes_df,
    height,
    camera_zoom,
    normalize_probe_lengths,
    output,
    render,
    width,
):
    if isinstance(brain, Path):
        print(f"Using local brain override: {brain}")
    else:
        source_label = atlas_source or (
            "allenInstitute" if atlas_name == "qiu2018_mouse" else "brainglobe"
        )
        print(
            f"Using Pinpoint atlas {atlas_name} from {source_label} "
            f"({atlas_resolution_um}um, version {atlas_version})"
        )
    rendered = render(
        electrodes_df,
        output,
        brain=brain,
        width=width,
        height=height,
        normalize_probe_lengths=normalize_probe_lengths,
        camera_zoom=camera_zoom,
    )
    if isinstance(rendered, list):
        for snapshot in rendered:
            print(f"Saved {snapshot}")
    rendered
    return


if __name__ == "__main__":
    app.run()
