# Related Work Positioning

Working title:

> Adaptive-Inference Conditional Diffusion Model for Efficient OTFS Channel Estimation

## Safe Gap

The paper should not claim a new diffusion model.

The safe claim is narrower:

> Existing conditional diffusion OTFS channel estimation accelerates inference mainly by reducing global sampling steps. This work adds DD-domain sparsity-aware adaptive inference inside each step, allocating computation to difficult delay-Doppler tiles.

## Nearest Neighbors

- OTFS modulation and DD-domain receiver modeling:
  Hadani et al., WCNC 2017.

- Embedded pilot OTFS channel estimation:
  Raviteja et al., IEEE TVT 2019. This is important because it makes DD sparsity a classical communication prior, not an invented AI story.

- Diffusion or score-based wireless channel estimation:
  Arvinte and Tamir, IEEE TWC 2023.

- DDIM and fast diffusion sampling:
  Song et al., ICLR 2021. This is the global-step acceleration baseline.

- Dynamic diffusion computation:
  DyDiT, ICLR 2025. This is an inspiration for timestep/spatial dynamic computation, but the paper must not sound like it merely ports DyDiT.

## Reviewer Attack Map

Attack 1:

> This is just generic token pruning.

Defense:

The adaptation signal is DD-domain channel sparsity and residual stability. The metrics are NMSE, BER if available, and receiver latency/FLOPs, not image FID or visual quality.

Attack 2:

> Masking after a full UNet forward does not reduce latency.

Defense:

Do not overclaim. Use the full-grid masked version only to validate NMSE safety. For latency claims, implement tile-wise active inference or report the result as estimated active computation.

Attack 3:

> DDIM already accelerates diffusion.

Defense:

DDIM reduces the number of global denoising steps. Adaptive DD inference reduces active DD regions inside each step. They are complementary and should be compared together.

Attack 4:

> The method damages weak channel paths by skipping low-amplitude cells.

Defense:

Use residual-change score, forced refresh, and SNR-adaptive thresholds. Include heatmaps and per-sample NMSE-loss scatter plots to show that weak but unstable regions are retained.

## Submission Gate

Do not submit unless all are true:

- Baseline 5-step DDIM is reproduced within 0.5 dB NMSE of the existing trained baseline result.
- There is at least one Pareto point with 30--40% inactive DD tiles and less than 0.3 dB NMSE loss.
- E0--E5 in the draft are complete.
- The text distinguishes measured latency from estimated FLOPs.
- All tables are generated from the same test split and hardware.
