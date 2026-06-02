# Preliminary Adaptive DD Inference Summary

This is a surrogate sparse-DD experiment, not the final trained checkpoint result.

- Selected tau: `0.24`
- Mean inactive DD computation: `33.9%`
- Mean NMSE loss vs fixed 5-step: `0.28 dB`
- Target gate: `0.30 dB`

## Main Results

| Method | Steps | NMSE@0 | NMSE@5 | NMSE@10 | Active comp. |
|---|---:|---:|---:|---:|---:|
| LS | -- | 0.01 | -5.00 | -10.00 | 100.0% |
| Fixed sparse denoiser | 5 | -7.35 | -12.19 | -16.99 | 100.0% |
| Fixed sparse denoiser | 3 | -6.58 | -11.47 | -16.32 | 100.0% |
| Adaptive DD inference | 5 | -7.18 | -11.85 | -16.67 | 66.1% |
