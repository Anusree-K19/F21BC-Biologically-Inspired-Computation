# Coursework verification

The reference snapshot is commit [`e7edc18`](https://github.com/Anusree-K19/F21BC-Biologically-Inspired-Computation/commit/e7edc18). The original dataset, 13 Python files, experiment settings and all notebook output entries are preserved. The portfolio notebook edits two headings and two progress labels, and adds explanatory text beside the archived labels.

## Environment and checks

Verification used an isolated Windows virtual environment with Python 3.12.10, NumPy 2.3.4, pandas 2.3.3, Matplotlib 3.10.7 and IPyKernel 7.1.0. All original package pins installed successfully. `pip check` reported no broken requirements. The notebook metadata records Python 3.11.0.

| Check | Result |
| --- | --- |
| Source and notebook validation | All 13 Python files compiled. The notebook passed schema validation. |
| `python run_checks.py` | Output shape `(1, 1)`, zero-input output 0, invalid input shape rejected, 161 parameters round-tripped, and random-weight test MAE of 34.86211828854752 MPa. |
| ANN and coupling | A hand-calculated forward pass matched. Batch predictions, the training fitness and MAE agreed. Single- and two-hidden-layer parameter vectors round-tripped exactly. |
| Data pipeline | 721 training rows and 309 test rows. Training features standardised to approximately zero mean and unit standard deviation. |
| PSO | Fixed informants included each particle itself. The best objective decreased monotonically in the sphere test. |
| Boundaries | Passed lower and upper reflections, velocity reversal, clipping of large overshoots and per-coordinate bounds. The notebook boundary demo matched its saved output. |

## Comparison with the archived notebook

A direct Jupyter run with the original row-by-row predictor matched the random-weight baseline, all seven single-run PSO regression experiments, the early-stopping trial and the ten-seed ReLU baseline. The direct run ended after those checks. A second run completed all 25 code cells in about two minutes. For this full run, a scratch-only batch evaluator used NumPy's stacked matrix-vector multiplication to remove the per-row Python loop. It gave bit-identical predictions to the original for 512 full training batches covering the four tested architectures and four hidden activations. Both runs used the same PSO, inputs, experiment settings and evaluation metrics.

The full comparison matched every printed concrete MAE at its saved precision, including the convergence logs, architecture and activation sweeps, activation-optimising PSO and hyperparameter grid. All 11 plotted PNGs were byte-identical to the archive. The two edited labels changed text only. The full comparison establishes agreement for the printed numbers and saved plots.

| Output | Archived | Full rerun |
| --- | ---: | ---: |
| Random-weight test MAE (MPa) | 34.86211828854752 | 34.86211828854752 |
| 10 × 450 single-run test MAE (MPa) | 7.407 | 7.407 |
| Fixed ReLU ten-seed test MAE (MPa) | 7.426 ± 0.572 | 7.426 ± 0.572 |
| Activation-optimising PSO ten-seed test MAE (MPa) | 7.575 ± 0.453 | 7.575 ± 0.453 |
| Sphere objective at initialisation | 39.65255357793565 | 39.65255357793565 |
| Sphere objective after 20 steps | 0.2840603311935848 | 0.7786059792022677 |

The illustrative ANN example draws an unseeded input, so its scalar output and activation ranges vary. The diagnostic working-directory line also varies by machine.

## Sphere benchmark investigation

The earliest PSO commit already contains the archived value, `0.2840603311935848`, in the sphere cell's saved output. The executable source of that cell and `pso/pso.py` is identical across the first PSO commit, the activation-extension commit and the final coursework snapshot, as verified by comparing their Python syntax trees. The archived output also stayed identical. [First PSO commit](https://github.com/Anusree-K19/F21BC-Biologically-Inspired-Computation/commit/f9aeeb6) · [Activation extension](https://github.com/Anusree-K19/F21BC-Biologically-Inspired-Computation/commit/26e04ca) · [Final comment edits](https://github.com/Anusree-K19/F21BC-Biologically-Inspired-Computation/commit/83baaa5)

The cell creates a fresh swarm with seed 0, 10 particles, five dimensions, bounds of −5 to 5 and three informants per particle. It uses the default inertia 0.729 and cognitive and social coefficients of 1.494, then takes 20 steps on the sum-of-squares objective. Each swarm creates its own NumPy `default_rng` (PCG64 in the tested environments). Changing the global NumPy seed and consuming up to 8,000 draws from the global generator before running the cell gave the same result. The earlier unseeded ANN demo therefore has no effect on this benchmark.

The unchanged historical cell and module were run together in fresh Windows processes under the following configurations.

| Python | NumPy | Best objective after 20 steps |
| --- | --- | ---: |
| 3.11.0 (notebook metadata) | 2.3.4 (repository pin) | 0.7786059792022677 |
| 3.11.0 | 1.23.5, 1.26.4, 2.0.2, 2.2.6, 2.3.3 | 0.7786059792022677 |
| 3.12.10 | 2.3.4 | 0.7786059792022677 |
| 3.14.3 | 2.4.4 | 0.7786059792022677 |

The particle positions, velocities, best values, informant sets and generator state matched exactly from initialisation through step 20 in each test. The best value reached 0.7786059792022677 at step 19 and still held at step 20. Continuing the same run first passed below the archived value at step 33 (`0.20254078844418322`). The best value never equalled the archived one in the first 500 steps. [NumPy random stream compatibility](https://numpy.org/doc/stable/reference/random/compatibility.html)

**The cause of the archived value remains unresolved.** The original notebook saved the sphere cell at execution count 10, before an explicit `importlib.reload(pso.pso)` at count 17. A module already loaded in the original kernel or output retained after an edit could explain the discrepancy. The notebook preserves the printed values and execution counts, but lacks a record of the live module source or the exact runtime build for that cell. The sphere test is independent of the concrete dataset and the ANN. Its archived result remains intact.

## Dataset and history review

The CSV was compared with UCI's public spreadsheet. It contains all 1,030 rows in the same apparent order, with simplified headers and rounded numeric values. The age column matched exactly, ingredient values differed by at most 0.05 kg/m³ and target values differed by less than 0.005 MPa. The coursework CSV is the input for reproducing the saved results. [UCI dataset and licence](https://doi.org/10.24432/C5PK67)

A pattern scan of tracked text across the available Git history found no credential candidates after checking matches embedded in PNG data. The saved notebook contains an old local working-directory path and Git records author emails. Those records are retained for preservation. No separate software licence was found in the tracked history. [Contribution and licensing notes](../README.md#contributions-and-licensing)
