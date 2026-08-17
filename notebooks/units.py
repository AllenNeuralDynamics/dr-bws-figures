import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    with open(r"C:\Users\ben.hardcastle\github\dr-bws-figures\assets\session_ids.json", "r") as f:
        session_ids = json.load(f)


@app.cell
def _():
    v = 'v0.0.289'
    return (v,)


@app.cell
def _(v):
    import polars as pl

    (
        pl.scan_parquet(
            f"s3://aind-scratch-data/dynamic-routing/cache/nwb_components/{v}/consolidated/units.parquet",
            storage_options={'skip_signature': 'true', 'region': 'us-west-2'}
        )
        .filter(
            # pl.col('session_id').is_in(session_ids["naive"]),
            pl.col('structure').str.contains("STN"),
        )
        .collect()
    )
    return (pl,)


@app.cell
def _(pl, sessions, v):
    from dr_bws.datacube import brainwide_ephys_filter

    sessions (
        pl.scan_parquet(
            f"s3://aind-scratch-data/dynamic-routing/cache/nwb_components/{v}/consolidated/session.parquet",
            storage_options={'skip_signature': 'true', 'region': 'us-west-2'}
        )
        .filter(brainwide_ephys_filter())
        .collect()
    )


if __name__ == "__main__":
    app.run()
