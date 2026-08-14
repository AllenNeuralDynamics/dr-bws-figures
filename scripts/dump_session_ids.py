"""Write session IDs for each standard ephys preset to assets/session_ids.json."""

import json
from pathlib import Path

from dr_bws.sessions import filter_presets, get_sessions

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "session_ids.json"


def dump_session_ids() -> None:
    session_ids = {
        preset: sorted(get_sessions(preset)["session_id"].unique().to_list())
        for preset in filter_presets().keys()
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(session_ids, indent=2) + "\n")


if __name__ == "__main__":
    dump_session_ids()
