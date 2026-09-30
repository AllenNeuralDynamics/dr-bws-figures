## Generating figures
- figures should be created in self-contained scripts with PEP723 dependencies, that write figures and tables of stats, so we can run with end-to-end with `uv run`. For example: `uv run --script figures/fig1/block-switch/marimo_nb.py` (or use the GitHub URL to run without cloning the repo, e.g. `https://raw.githubusercontent.com/AllenNeuralDynamics/dr-bws-figures/refs/heads/main/figures/fig1/block-performance/marimo_nb.py`)
- preferably, the scripts will be written as marimo notebooks for interactive development and exploration. For example: `uvx marimo edit --sandbox https://raw.githubusercontent.com/AllenNeuralDynamics/dr-bws-figures/refs/heads/main/figures/fig1/block-performance/marimo_nb.py`
- `--sandbox` runs in a temporary environment according to the dependencies specified in the notebook.
- run scripts after making modifications

## Colors and naming conventions

### Context (block)
- should be referred to as "context"
- color text, not color blocks or box outlines 
- "VIS":  "#4258A7" (light gray background #E6E7E8 where applicable)
- "AUD":  "#F36B10" (white background)

### Stimulus
- "A+":  #BF00BF ('m' in matplotlib, not 'magenta')
- "V+":  #2ca02c ('tab:g' in matplotlib, not 'g')
- "V-"/"A-": #000000

### Instruction trial patch
- #D0B9DB 

## Figure text
- Arial
- axis labels and titles:sentence case (first letter and proper nounscapitalized)
- tick labels typically lower case

