# /// script
# dependencies = [
#     "dr-datacube",
#     "matplotlib",
#     "marimo",
#     "polars==1.43.2",
# ]
# requires-python = ">=3.11"
#
# [tool.uv.sources]
# dr-datacube = { git = "https://github.com/AllenNeuralDynamics/dr-datacube" }
# ///

"""Plot the number of recorded units by broad anatomical region."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    from dr_datacube import config as datacube_config
    from dr_datacube import get_lf, get_session_ids_from_github, on_codeocean

    datacube_config.use_cache = True
    plt.rcParams["font.family"] = "Arial"
    results_dir = (
        Path(__file__).resolve().parent
        if not on_codeocean()
        else Path("/root/capsule/results")
    )

    return Path, get_lf, get_session_ids_from_github, pl, plt, results_dir


@app.cell
def _(get_lf, get_session_ids_from_github, pl):
    # This is the session list used by the other Fig. 1 datacube notebooks.
    # All tabular data below comes from dr_datacube.get_lf.
    session_ids = get_session_ids_from_github("brainwide")

    structure_grouping = {
        "Frontal cortex": ["ACAd", "ACAv", "FRP", "ORBl", "ORBvl", "ORBm", "PL", "ILA"],
        "Somatomotor cortex": ["MOs", "MOp", "SSp", "SSs"],
        "Lateral cortex": ["AId", "AIp", "AIv", "GU", "VISC", "TEa", "PERI", "ECT"],
        "Visual cortex": ["VISp", "VISl", "VISal", "VISli", "VISpl", "VISpor", "VISrl"],
        "Medial cortex": ["VISa", "VISam", "VISpm", "RSPagl", "RSPd", "RSPv"],
        "Auditory cortex": ["AUDp", "AUDv", "AUDd", "AUDpo"],
        "Cortical subplate": ["CLA", "EPd", "EPv", "LA", "BLA", "BMA", "PA"],
        "Hippocampal formation": ["CA1", "CA2", "CA3", "DG", "IG", "ENTl", "ENTm", "PAR", "POST", "PRE", "SUB", "ProS", "HPF", "FC", "APr"],
        "Olfactory areas": ["OLF", "AON", "AOB", "MOB", "TT", "TTd", "DP", "PIR", "NLOT1", "NLOT2", "AOBmi", "AOBgl"],
        "Thalamus - sensorimotor": ["VAL", "VM", "VPL", "VPLpc", "VPM", "VPMpc", "MGd", "MGv", "MGm", "LGd", "PP", "PoT", "SPF", "SPFp", "SPFm"],
        "Thalamus - association": ["LP", "PO", "POL", "SGN", "Eth", "AV", "AMd", "AMv", "AD", "IAM", "IAD", "LD", "IMD", "MD", "SMT", "PR", "PVT", "PT", "RE", "Xi", "RH", "PCN", "CM", "CL", "PF", "PIL", "RT", "IGL", "IntG", "LGv", "SubG", "MH", "LH"],
        "Basal ganglia": ["CP", "ACB", "OT", "LSc", "LSr", "LSv", "CEAm", "MEA", "SF", "SH", "SFO", "GPe", "GPi", "BST", "MS", "TRS", "NDB", "SI"],
        "Hypothalamus": ["LHA", "ZI", "FF", "PSTN", "MPO", "PVH", "PVHd", "PH", "SUM", "Mml", "MPN", "STN", "PeF"],
        "Midbrain": ["SCs", "SCm", "ICd", "ICe", "SAG", "NB", "PBG", "SCO", "MRN", "RN", "APN", "MPT", "NOT", "OP", "PAG", "PPT", "VTA", "SNr", "SNc", "PPN", "PRC", "RR", "DT", "NPC", "CUN", "INC", "DR", "III", "Su3", "AT", "CLI", "IPC", "IPR", "PN", "LT", "MT", "ND", "Pa4"],
        "Hindbrain": ["PCG", "CS", "DTN", "PRNc", "PRNr", "PRNv", "NI", "P", "LDT"],
        "Medulla": ["GRN"],
    }
    structure_to_group = {
        structure: group
        for group, structures in structure_grouping.items()
        for structure in structures
    }
    group_order = {group: order for order, group in enumerate(structure_grouping, start=1)}
    group_colors = {
        "Frontal cortex": "#2B7A3E", "Somatomotor cortex": "#5EBA47", "Lateral cortex": "#98C13D",
        "Visual cortex": "#1E7B7B", "Medial cortex": "#45A87E", "Auditory cortex": "#3E9B9B",
        "Cortical subplate": "#2EC4B6", "Hippocampal formation": "#7B68AE", "Olfactory areas": "#A7A844",
        "Thalamus - sensorimotor": "#E05B5B", "Thalamus - association": "#F49D6E", "Basal ganglia": "#4A6FA5",
        "Hypothalamus": "#C93C2B", "Midbrain": "#9B2D9B", "Hindbrain": "#D4A82E", "Medulla": "#8B5E2B",
    }

    units = (
        get_lf("unit_metrics")
        .filter(
            pl.col("session_id").is_in(session_ids),
            pl.col("presence_ratio") >= 0.7,
            pl.col("isi_violations_ratio") <= 0.5,
            pl.col("amplitude_cutoff") <= 0.1,
            pl.col("activity_drift") <= 0.1,
            pl.col("decoder_label").ne("noise"),
            pl.col("structure").is_not_null(),
            pl.col("location").is_not_null(),
        )
        .select("unit_id", "session_id", "structure")
        .with_columns(
            pl.col("structure").replace_strict(structure_to_group, default="").alias("structure_group")
        )
        .filter(pl.col("structure_group").ne(""))
        .collect()
    )

    return group_colors, group_order, pl, units


@app.cell
def _(group_colors, group_order, pl, results_dir, units, plt):
    min_n_units = 100
    min_n_sessions = 3

    units_per_region = (
        units.group_by("structure_group")
        .agg(
            pl.col("unit_id").n_unique().alias("n_units"),
            pl.col("session_id").n_unique().alias("n_sessions"),
        )
        .join(
            units.group_by(["structure_group", "session_id"])
            .agg(pl.col("unit_id").n_unique().alias("n_units_per_session"))
            .group_by("structure_group")
            .agg(pl.col("n_units_per_session").median().alias("median_units_per_session")),
            on="structure_group",
        )
        .filter(
            (pl.col("n_units") > min_n_units)
            & (pl.col("n_sessions") >= min_n_sessions)
            & (pl.col("structure_group") != "Cortical subplate")
        )
        .with_columns(
            pl.col("structure_group")
            .replace(group_order)
            .alias("group_order")
        )
        .sort("n_units")
    )

    fig, ax = plt.subplots(figsize=(4, 2.8))
    labels = units_per_region["structure_group"].to_list()
    ax.barh(labels, units_per_region["n_units"], color="white", edgecolor="black")
    ax.plot(
        units_per_region["median_units_per_session"],
        labels,
        "|",
        color="black",
        markersize=8,
        markeredgewidth=1,
    )
    ax.set_xlabel("N units per region")
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1000", "10000"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    top_bar_y = len(labels) - 1
    ax.set_ylim(-0.8, top_bar_y + 1.1)
    median_x = units_per_region["median_units_per_session"][-1]
    label_x = 1100
    label_y = top_bar_y + 0.92
    ax.plot(
        [median_x, label_x],
        [top_bar_y, label_y],
        color="#808080",
        lw=0.8,
        clip_on=False,
    )
    ax.text(
        label_x,
        label_y,
        "median per session",
        ha="left",
        va="center",
        fontsize=7,
        clip_on=False,
    )
    fig.tight_layout()
    fig.savefig(results_dir / "units-per-region.svg", bbox_inches="tight")
    fig.savefig(
        results_dir / "units-per-region.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
    units_per_region.write_csv(results_dir / "units-per-region.csv")
    fig

    return fig, units_per_region


if __name__ == "__main__":
    app.run()
