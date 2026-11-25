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

The sphere difference also appeared in the direct run. Git history shows that the original value was already present when the sphere experiment was introduced. Its cause remains unresolved. The illustrative ANN example draws an unseeded input, so its scalar output and activation ranges vary. The diagnostic working-directory line also varies by machine.

## Dataset and history review

The CSV was compared with UCI's public spreadsheet. It contains all 1,030 rows in the same apparent order, with simplified headers and rounded numeric values. The age column matched exactly, ingredient values differed by at most 0.05 kg/m³ and target values differed by less than 0.005 MPa. The coursework CSV is the input for reproducing the saved results. [UCI dataset and licence](https://doi.org/10.24432/C5PK67)

A pattern scan of tracked text across the available Git history found no credential candidates after checking matches embedded in PNG data. The saved notebook contains an old local working-directory path and Git records author emails. Those records are retained for preservation. No separate software licence was found in the tracked history. [Contribution and licensing notes](../README.md#contributions-and-licensing)
