# studies/

Each study repeatedly launches a tutorial's single-trial CLI as an isolated subprocess,
varying one thing (seed, initial-dataset size, ...), each trial writing its own `summary.bin`
and a `step_NNN_repMM/experiment.json` per step.

## Examples

```
python -m studies.variability_study \
    --target tutorials.multi_objective.branin_currin.main --n-replicates 20

python -m studies.variability_study \
    --target tutorials.multi_objective.branin_currin.main --n-initial 5,10,20,40 --n-replicates 5

python -m studies.variability_study --target tutorials.multi_objective.branin_currin.main \
    --strategy bo    --n-replicates 10 --output-dir data/branin_currin
    
python -m studies.variability_study --target tutorials.multi_objective.branin_currin.main \
    --strategy sobol --n-replicates 10 --output-dir data/branin_currin
```