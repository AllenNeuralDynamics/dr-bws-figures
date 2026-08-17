import contextlib
import functools
import logging
import os
from typing import Literal
from collections.abc import Callable

import aind_session
import polars as pl
import pydantic
import pydantic_settings
import upath
from polars._typing import FrameType

logger = logging.getLogger(__name__)

def pipeline_data_dir() -> upath.UPath:
    return upath.UPath("/tmp/data")

def capsule_data_dir() -> upath.UPath:
    return upath.UPath("/root/capsule/data")

@functools.cache
def on_codeocean() -> bool:
    with contextlib.suppress(Exception):
        return capsule_data_dir().exists() or is_pipeline()
    return False

@functools.cache
def is_pipeline() -> bool:
    with contextlib.suppress(Exception):
        return pipeline_data_dir().exists() and bool(os.environ.get("AWS_BATCH_JOB_ID"))
    return False

class DatacubeConfig(pydantic_settings.BaseSettings):
    model_config = pydantic.ConfigDict(validate_assignment=True)

    version: str = "v0.0.289"
    use_cache: bool = False
    s3_cache_dir: upath.UPath = upath.UPath("s3://aind-scratch-data/dynamic-routing/cache", anon=True)
    
    @property
    def asset_dir(self) -> upath.UPath:
        if on_codeocean():
            data_dir = pipeline_data_dir() if is_pipeline() else capsule_data_dir() 
            try:
                datacube_dir = tuple(data_dir.glob("dynamicrouting_datacube*"))
            except StopIteration:
                raise FileNotFoundError(f"Could not find dynamicrouting_datacube data asset in {data_dir}")
            if not datacube_dir:
                raise FileNotFoundError(f"Could not find dynamicrouting_datacube data asset in {data_dir}")
            if len(datacube_dir) > 1:
                choice = next((d for d in datacube_dir if self.version in d.name), datacube_dir[0])
                logger.warning(f"Found multiple dynamicrouting_datacube data assets in {data_dir}, using: {choice} (set `datacube_config.version` to change)")
                return choice
            return datacube_dir[0]
        # get S3 dir of datacube asset from CO API
        return aind_session.get_data_asset_source_dir(
            next(d for d in reversed(aind_session.get_data_assets('dynamicrouting_datacube')) if self.version in d.name).id
        )

    @property
    def nwb_dir(self) -> upath.UPath:
        if self.use_cache:
            return self.s3_cache_dir / "nwb" / self.version 
        return self.asset_dir / "nwb"

    @property
    def parquet_dir(self) -> upath.UPath:
        if self.use_cache:
            return self.s3_cache_dir / "nwb_components" / self.version / "consolidated"
        else:
            return self.nwb_dir.parent / "consolidated"
            
datacube_config = DatacubeConfig()

def get_lf(name: str) -> pl.LazyFrame:
    storage_options = {} if not datacube_config.use_cache else {"skip_signature": "true", "region": "us-west-2"}
    return pl.scan_parquet((datacube_config.parquet_dir / f"{name}.parquet").as_posix(), storage_options=storage_options)

@functools.cache
def list_nwb_sources() -> tuple[str, ...]:
    """Get all file URIs."""
    sources = sorted(path.as_posix() for path in datacube_config.nwb_dir.glob("*.nwb*"))
    logger.info(f"Found {len(sources)} NWB sources in {datacube_config.nwb_dir}")
    return tuple(sources)

def ensure_id_cols(df: FrameType) -> FrameType:
    schema = df.lazy().collect_schema() # works if we pass a dataframe or lazyframe
    if "session_id" in schema and "subject_id" in schema:
        logger.debug("DataFrame already has a `session_id` and `subject_id` columns, skipping parsing from `_nwb_path`")
        return df
    if "_nwb_path" not in schema and "session_id" not in schema:
        raise ValueError("Attempted to parse `session_id` from `_nwb_path` column, which doesn't exist in dataframe")
    if "session_id" not in schema:
        df = df.with_columns(
            pl.col("_nwb_path").str.split("/").list.get(-1).str.split(".").list.get(0).alias("session_id")
        )
    if "subject_id" not in schema:
        df = df.with_columns(pl.col("session_id").str.split("_").list.get(0).alias("subject_id"))
    return df

def behavior_summary(block_dprime_threshold: float = 1.0) -> pl.DataFrame:
    return (
        get_lf("performance")
        .with_columns(
            pl.col("n_contingent_rewards").gt(10).alias("is_engaged_block"),
            pl.col("cross_modality_dprime").ge(block_dprime_threshold).alias("is_good_block"),
        )
        .group_by("session_id", "rewarded_modality")
        .agg(
            pl.col("is_engaged_block", "is_good_block").sum()
        )
        .group_by("session_id")
        .agg(
            pl.col("is_good_block").filter(pl.col("rewarded_modality") == "vis").first().alias("n_good_vis_blocks"),
            pl.col("is_good_block").filter(pl.col("rewarded_modality") == "aud").first().alias("n_good_aud_blocks"),
            pl.col("is_engaged_block").sum().alias("n_engaged_blocks"),
        )
    ).collect()   # Added return statement

def brainwide_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
    required = ("prod", "brainwide_survey", "task", "ephys", "ccf") # good_behavior is incorrect - will be fixed in v0.0.290
    excluded = ("issues", "context naive") # context_naive (w underscore) had bug - should be mutually exclusive with bws from >=v0.0.290
    if with_behavior_filter:
        good_behavior_session_ids = (
            behavior_summary(block_dprime_threshold=1.0)
            .filter(
                pl.col("n_good_aud_blocks").ge(2),
                pl.col("n_good_vis_blocks").ge(2),
            )
        )["session_id"].to_list()
    else:
        good_behavior_session_ids = None
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
        pl.col("session_id").is_in(good_behavior_session_ids) if with_behavior_filter else pl.lit(True)
    )

def naive_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
    required = ("dynamic_routing", "task", "ephys", "ccf", "context naive") 
    # TODO prod should be included, but is incorrect
    #TODO switch to "context_naive" (w/underscore) when fixed in v0.0.290
    excluded = ("issues", "templeton") # TODO add "brainwide_survey" when fixed in v0.0.290
    if with_behavior_filter:
        engaged_session_ids = (
            behavior_summary()
            .filter(
                pl.col("n_engaged_blocks").ge(4),
            )
        )["session_id"].to_list()
    else:
        engaged_session_ids = []
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
        pl.col("session_id").is_in(engaged_session_ids) if with_behavior_filter else pl.lit(True),
    )
    
def templeton_ephys_filter(with_behavior_filter: bool = True) -> pl.Expr:
    required = ("prod", "templeton", "task", "ephys", "ccf")
    excluded = ("issues", ) 
    if with_behavior_filter:
        good_behavior_session_ids = (
            get_lf("performance")
            .filter(
                pl.col('cross_modality_dprime').is_null(),
                pl.col('aud_dprime').ge(1.0) | pl.col('vis_dprime').ge(1.0),
            )
        ).collect()["session_id"].to_list()
    else:
        good_behavior_session_ids = []
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
        pl.col("session_id").is_in(good_behavior_session_ids) if with_behavior_filter else pl.lit(True),
    )

def filter_functions() -> dict[str, Callable[[bool], pl.Expr]]:
    """Return a dict of filter functions for each session_type. Each function has signature (with_behavior_filter: bool = True) -> pl.Expr."""
    return {
        'brainwide': brainwide_ephys_filter, 
        'naive': naive_ephys_filter, 
        'templeton': templeton_ephys_filter,
    }

@functools.cache
def get_sessions(
    session_type: Literal['brainwide', 'naive', 'templeton'] | None = 'brainwide',
    with_behavior_filter: bool = True,
    only_in_data_asset: bool = True,
    filter_expr: pl.Expr | None = None,
) -> pl.DataFrame:
    """A DataFrame with 'session_id' and 'keywords'.
 
    Options:
    'brainwide' (default) - standard brainwide survey ephys sessions
    'naive' - context naive dynamic routing ephys sessions.
    'templeton' - Templeton ephys sessions.
    None - all sessions in datacube

    If `with_behavior_filter` is True, a standard behavioral filter for each session type will be applied.

    If a custom `filter_expr` is passed, it will be applied to the NWB "session" table, which contains keywords, session_id and subject_id for filtering. The value of `session_type` will be ignored.

    If only_in_data_asset is True, a further filter will be applied to return only sessions present in the CO data asset. This requires credentials to check CO and S3.
    """
    if filter_expr is None:
        if not session_type:
            raise ValueError("If `filter_expr` is None, a valid `session_type` must be provided.")
        elif session_type not in filter_functions():
            raise ValueError(f"Unknown filter session_type. Use one of: {list(filter_functions().keys())}")
        filter_expr = filter_functions()[session_type](with_behavior_filter)
    filtered = (
        get_lf("session")
        .pipe(ensure_id_cols)
        .filter(filter_expr)
        .select('session_id', 'subject_id', 'keywords')
        .collect()
    )
    if only_in_data_asset:
        session_ids_in_data_asset = pl.read_parquet((datacube_config.asset_dir / 'session_table.parquet').as_posix(), columns=["session_id"])["session_id"].sort().to_list()
        filtered = filtered.filter(pl.col("session_id").is_in(session_ids_in_data_asset))
    return filtered

if __name__ == "__main__":
    import doctest
    doctest.testmod()
