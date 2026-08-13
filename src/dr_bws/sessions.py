from typing import Literal
import functools
import logging

import lazynwb
import polars as pl
import pydantic
import pydantic_settings
import upath

logger = logging.getLogger(__name__)

class DatacubeConfig(pydantic_settings.BaseSettings):
    model_config = pydantic.ConfigDict(validate_assignment=True)

    version: str = "v0.0.289"
    use_cache: bool = True
    s3_cache_dir: upath.UPath = upath.UPath("s3://aind-scratch-data/dynamic-routing/cache/", anon=True)
    
    @property
    def nwb_dir(self) -> upath.UPath:
        if self.use_cache:
            return self.s3_cache_dir / "nwb" / self.version 
        else:
            data_dir = upath.UPath("/root/capsule/data")
            if not data_dir.exists():
                raise FileNotFoundError("Could not find /root/capsule/data directory")
            try:
                return next(data_dir.glob("dynamicrouting_datacube*")) / "nwb"
            except StopIteration:
                raise FileNotFoundError("Could not find dynamicrouting_datacube data asset in /root/capsule/data")

    @property
    def parquet_dir(self) -> upath.UPath:
        if self.use_cache:
            return self.s3_cache_dir / "nwb_components" / self.version
        else:
            return self.nwb_dir.parent / "consolidated"
            
datacube_config = DatacubeConfig()

def get_session_ids() -> list[str]:
    """Get a list of all session IDs.
    
    Examples
    --------
    >>> get_session_ids()[:2]
    ['620263_2022-07-26', '620263_2022-07-27']
    """
    return sorted(session_id.stem.strip('.nwb').strip('.zarr') for session_id in datacube_config.nwb_dir.glob("*.nwb*"))

@functools.cache
def list_nwb_sources() -> tuple[str, ...]:
    """Get all file URIs."""
    sources = sorted(path.as_posix() for path in datacube_config.nwb_dir.glob("*.nwb*"))
    logger.info(f"Found {len(sources)} NWB sources in {datacube_config.nwb_dir}")
    return tuple(sources)

def ensure_id_cols(df: pl.DataFrame) -> pl.DataFrame:
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
        df = df.with_columns(pl.col("session_id").split("_").list.get(0))
    return df

def brainwide_ephys_filter() -> pl.Expr:
    required = ("prod", "brainwide_survey", "task", "ephys", "ccf", "good_behavior")
    excluded = ("issues", "early_autorewards", "context_naive") # context_naive had bug - should be mut-ex with bws in future
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
    )

def naive_ephys_filter() -> pl.Expr:
    required = ("prod", "dynamic_routing", "task", "ephys", "ccf", "context_naive")
    excluded = ("issues", "early_autorewards", ) 
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
    )
    
def templeton_ephys_filter() -> pl.Expr:
    required = ("prod", "templeton", "task", "ephys", "ccf", "context_naive")
    excluded = ("issues", "early_autorewards", ) 
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
    )

@functools.cache
def get_sessions(
    preset: Literal['brainwide', 'naive'] | None = 'brainwide',
    filter_expr: pl.Expr | None = None,
) -> pl.DataFrame:
    """A DataFrame with 'session_id' and 'keywords'.
 
    Options:
    'brainwide' (default) - standard brainwide survey ephys sessions, passing "good behavior" criterion.
    'naive' - context naive dynamic routing ephys sessions.
    'templeton' - Templeton ephys sessions.
    None - all sessions in datacube

    If a custom `filter_expr` is passed, it will be applied to the NWB "session" table, which contains keywords, session_id and subject_id for filtering. The value of `preset` will be ignored.
    """
    presets = {
        'brainwide': brainwide_ephys_filter(), 
        'naive': naive_ephys_filter(), 
        'templeton': templeton_ephys_filter(),
    }
    if filter_expr is None:
        if preset and preset not in presets:
            raise ValueError("Unknown filter preset. Use one of: 'brainwide', 'naive', 'templeton'")
        filter_expr = presets.get(preset, pl.lit(True))
    return (
        lazynwb.scan_nwb(list_nwb_sources(), "session")
        .pipe(ensure_id_cols)
        .select('session_id', 'keywords')
        .filter(filter_expr)
        .collect()
    )

if __name__ == "__main__":
    import doctest
    doctest.testmod()