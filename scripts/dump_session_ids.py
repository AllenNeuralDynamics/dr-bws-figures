"""Write session IDs for each standard ephys preset to assets/session_ids.json."""

import json
from pathlib import Path

import lazynwb

from dr_bws.sessions import get_sessions, filter_presets


OUTPUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "session_ids.json"


def dump_session_ids() -> None:
    lazynwb.config.anon = True
    session_ids = {
        preset: sorted(get_sessions(preset)["session_id"].unique().to_list())
        for preset in filter_presets().keys()
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(session_ids, indent=2) + "\n")


if __name__ == "__main__":
    dump_session_ids()
