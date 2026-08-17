# /// script
# dependencies = [
#     "dr-bws",
#     "polars",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# dr-bws = { git = "https://github.com/AllenNeuralDynamics/dr-bws-figures" }
# ///
"""Write session IDs for each standard ephys preset to assets/session_ids.json."""

from pathlib import Path

import polars as pl

from dr_bws import datacube


def session_table() -> pl.DataFrame:
    dfs = []
    for session_type in datacube.filter_functions():
        good = datacube.get_sessions(session_type).with_columns(pl.lit(True).alias("is_behavior_pass"))
        all = datacube.get_sessions(session_type, with_behavior_filter=False)
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

def dump_session_table() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_table().write_csv(output_dir / "datacube_sessions.csv")

if __name__ == "__main__":
    dump_session_table()
