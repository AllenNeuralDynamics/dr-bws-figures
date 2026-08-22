## Writing figure notebooks/scripts


## Generating figures and tables
- as a self-contained script that writes figures and tables: `uv run --script figures/fig1/block-switch/marimo_nb.py` (or use the GitHub URL to run without cloning the repo, e.g. `https://raw.githubusercontent.com/AllenNeuralDynamics/dr-bws-figures/refs/heads/main/figures/fig1/block-performance/marimo_nb.py`)
- as an interactive notebook in a browser for the user to explore and modify: `uvx marimo edit --sandbox https://raw.githubusercontent.com/AllenNeuralDynamics/dr-bws-figures/refs/heads/main/figures/fig1/block-performance/marimo_nb.py`
- `--sandbox` runs in a temporary environment according to the dependencies specified in the notebook.