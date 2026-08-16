Session metadata for the standard ephys sets is published in
[`assets/sessions.csv`](assets/sessions.csv). The table contains the session and
subject IDs, the standard session type, and whether the session passes the
behavior filter. The available session types are `brainwide`, `naive`, and
`templeton`.

Read it directly from GitHub with pandas:

```python
import pandas as pd

url = "https://raw.githubusercontent.com/allenneuraldynamics/dr-bws-figures/main/assets/sessions.csv"
sessions = pd.read_csv(url)
brainwide_pass = sessions.query("is_behavior_pass and session_type == 'brainwide'")
```

Or with Polars:

```python
import polars as pl

url = "https://raw.githubusercontent.com/allenneuraldynamics/dr-bws-figures/main/assets/sessions.csv"
sessions = pl.read_csv(url)
brainwide_pass = sessions.filter(
    pl.col("is_behavior_pass"),
    pl.col("session_type") == "brainwide",
)
```
