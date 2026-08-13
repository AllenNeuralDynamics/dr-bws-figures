Session IDs for the standard ephys sets are in [`assets/session_ids.json`](assets/session_ids.json). Read them locally with:

```python
import json
from urllib.request import urlopen

url = "https://raw.githubusercontent.com/allenneuraldynamics/dr-bws-figures/main/assets/session_ids.json"
session_ids = json.load(urlopen(url))
brainwide_ids = session_ids["brainwide"]
```

To apply the standard filters to the local session table and compare the
results with the checked-in lists:

```sh
uv run python scripts/compare_session_ids.py --verbose
```

The command reports missing and extra IDs and exits nonzero when a preset does
not match. Use `--preset brainwide` (repeatable) to check only selected sets.
