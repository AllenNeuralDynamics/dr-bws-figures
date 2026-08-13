import logging

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

def list_nwb_sources(use_s3_cache: bool = True) -> list[str]:
    """Expand the public S3 cache glob into lazynwb-compatible URLs."""
    sources = sorted(path.as_posix() for path in datacube_config.nwb_dir.glob("*.nwb*"))
    logger.info(f"Found {len(sources)} NWB sources in {datacube_config.nwb_dir}")
    return sources


def _ensure_session_id(df: pl.DataFrame) -> pl.DataFrame:
    if "session_id" in df.columns:
        logger.debug("DataFrame already has a `session_id` column, skipping parsing from `_nwb_path`")
        return df
    if "_nwb_path" not in df.columns:
        raise ValueError("Attempted to parse `session_id` from `_nwb_path` column, which doesn't exist in dataframe")
    return df.with_columns(pl.col("_nwb_path").str.split("/").list.get(-1).str.split(".").list.get(0).alias("session_id"))

def standard_brainwide_filter() -> pl.Expr:
    required = ("prod", "brainwide_survey", "task", "ephys", "ccf", "good_behavior")
    excluded = ("issues","early_autorewards", "context_naive") # context_naive had bug - should be mut-ex with bws in future
    return pl.all_horizontal(
        *[pl.col("keywords").list.contains(keyword) for keyword in required],
        *[~pl.col("keywords").list.contains(keyword) for keyword in excluded],
    )

if __name__ == "__main__":
    import doctest
    doctest.testmod()