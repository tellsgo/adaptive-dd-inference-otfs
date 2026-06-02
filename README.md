# Adaptive DD Inference Draft v0.1

Working title:

> Adaptive-Inference Conditional Diffusion Model for Efficient OTFS Channel Estimation

## Decision

This route is simpler than the MA diffusion detection route.

It is worth keeping as a second insurance paper because it can reuse an existing
OTFS diffusion channel-estimation codebase, dataset, checkpoint, and paper context.
The v0.1 package now contains both a manuscript and a runnable surrogate
experiment. The student should replace the surrogate sparse denoiser with the
real trained checkpoint and then fill the final experiment matrix.

It does not guarantee publication. It improves the probability because the
scope is bounded and the experiment package is easy to complete.

## Files

- `main.tex`: IEEEtran v0.1 paper scaffold.
- `main.pdf`: compiled v0.1 draft with preliminary generated figures.
- `references.bib`: lightweight bibliography.
- `RELATED_WORK_POSITIONING.md`: reviewer-risk and related-work map.
- `scripts/run_preliminary_adaptive_dd.py`: runnable sparse-DD surrogate
  harness.
- `results/`: generated CSVs, LaTeX table, macros, and summary.
- `figures/`: generated threshold sweep, skip-mask heatmap, and ablation plots.

## Current Preliminary Result

The surrogate harness selects `tau=0.24` and gives:

- Mean inactive DD computation: `33.9%`.
- Mean NMSE loss vs. fixed 5-step denoising: `0.28 dB`.
- Mean active DD computation: `66.1%`.

These are not final DeepMIMO/checkpoint results. They prove that the mechanism
can produce a coherent article draft and define the replacement experiment.

## Core Story

The baseline conditional diffusion estimator spends the same compute on every
delay-Doppler grid position. DD-domain channels are sparse, so many tiles are
easy or near zero. The method freezes easy tiles during later denoising steps
and keeps difficult or unstable tiles active.

The contribution is not a new diffusion model.

The contribution is:

> DD-domain sparsity-aware adaptive inference for efficient conditional
> diffusion OTFS channel estimation.

## Student Experiment Gate

Minimum required experiments:

- E0: reproduce the trained baseline NMSE vs. SNR.
- E1: threshold sweep, tau vs. NMSE vs. skip ratio.
- E2: SNR-adaptive threshold schedule.
- E3: DD-domain skip-mask heatmaps.
- E4: ablation without forced refresh, without SNR adaptation, and without residual-change score.
- E5: latency, estimated FLOPs, memory.
- E6: optional speed sweep, e.g. 60/90/120 km/h.
- E7: optional coverage or scene robustness.

## Critical Honesty Rule

If the code runs the full UNet and masks only the output, it does not reduce
real latency. In that case the paper may report skip ratio and estimated active
FLOPs only.

Measured latency reduction can be claimed only after a tile-wise, sparse, or
blocked implementation actually avoids full-grid UNet computation.

## Suggested Four-Week Plan

Week 1:

Reproduce the 5-step DDIM baseline and collect NMSE, latency, and memory.

Week 2:

Implement the simplest full-grid masked update. Sweep thresholds and prove that
freezing easy DD tiles does not destroy NMSE.

Week 3:

Add residual-change score, SNR/step threshold schedule, and forced refresh.
Generate ablations and heatmaps.

Week 4:

If time allows, implement active-tile inference for real latency. Otherwise,
write the paper around estimated active computation and be explicit about the
limitation.

## Build

Requires Python with `numpy` and `pillow` available.

```bash
python3 scripts/run_preliminary_adaptive_dd.py
```

Compile the manuscript with either Tectonic:

```bash
tectonic main.tex
```

or a traditional TeX Live/MacTeX toolchain:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
