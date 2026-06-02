#!/usr/bin/env python3
"""Preliminary adaptive-DD inference harness.

This script is intentionally a small surrogate experiment, not a replacement
for an existing trained OTFS diffusion checkpoint. It tests the paper mechanism:
after early denoising, stable low-energy delay-Doppler tiles can be frozen with
limited NMSE loss. The generated tables and figures are suitable for a v0.1
draft and must be replaced by the real DeepMIMO/checkpoint results before
submission.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SimConfig:
    batch_size: int = 96
    n_doppler: int = 32
    n_delay: int = 64
    n_rx: int = 2
    n_tx: int = 2
    n_paths_min: int = 5
    n_paths_max: int = 10
    tile_doppler: int = 4
    tile_delay: int = 4
    seed: int = 20260531
    snr_db_values: tuple[int, ...] = (0, 5, 10)
    tau_values: tuple[float, ...] = (0.16, 0.20, 0.22, 0.24, 0.26, 0.28, 0.32)
    target_loss_db: float = 0.30
    adaptive_warmup_steps: int = 2


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    real = rng.normal(0.0, scale / math.sqrt(2.0), size=shape)
    imag = rng.normal(0.0, scale / math.sqrt(2.0), size=shape)
    return real + 1j * imag


def generate_sparse_dd_channels(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate sparse complex DD-domain channels with fractional path spread."""
    doppler_grid = np.arange(cfg.n_doppler, dtype=np.float64)[:, None]
    delay_grid = np.arange(cfg.n_delay, dtype=np.float64)[None, :]
    channels = np.zeros(
        (cfg.batch_size, cfg.n_rx, cfg.n_tx, cfg.n_doppler, cfg.n_delay),
        dtype=np.complex128,
    )

    for b_idx in range(cfg.batch_size):
        n_paths = int(rng.integers(cfg.n_paths_min, cfg.n_paths_max + 1))
        base = np.zeros((cfg.n_doppler, cfg.n_delay), dtype=np.complex128)
        for _ in range(n_paths):
            doppler_center = rng.uniform(0, cfg.n_doppler - 1)
            delay_center = rng.uniform(0, cfg.n_delay - 1)
            doppler_spread = rng.uniform(0.35, 0.95)
            delay_spread = rng.uniform(0.35, 1.10)
            kernel = np.exp(
                -0.5 * ((doppler_grid - doppler_center) / doppler_spread) ** 2
                -0.5 * ((delay_grid - delay_center) / delay_spread) ** 2
            )
            kernel /= np.sqrt(np.sum(np.abs(kernel) ** 2) + 1e-12)
            gain = complex_normal(rng, (), scale=1.0 / math.sqrt(n_paths))
            base += gain * kernel

        for rx_idx in range(cfg.n_rx):
            for tx_idx in range(cfg.n_tx):
                antenna_gain = complex_normal(rng, (), scale=1.0)
                phase_slope_d = rng.uniform(-0.05, 0.05)
                phase_slope_l = rng.uniform(-0.03, 0.03)
                phase = np.exp(
                    1j
                    * (
                        phase_slope_d * np.arange(cfg.n_doppler)[:, None]
                        + phase_slope_l * np.arange(cfg.n_delay)[None, :]
                    )
                )
                channels[b_idx, rx_idx, tx_idx] = antenna_gain * base * phase

        power = np.mean(np.abs(channels[b_idx]) ** 2)
        assert power > 0.0
        channels[b_idx] /= math.sqrt(power)

    return channels


def add_ls_noise(
    channel: np.ndarray, snr_db: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    signal_power = float(np.mean(np.abs(channel) ** 2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_sigma = math.sqrt(noise_power)
    noisy = channel + complex_normal(rng, channel.shape, scale=noise_sigma)
    return noisy, noise_sigma


def nmse_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    numerator = np.sum(np.abs(estimate - truth) ** 2, axis=(1, 2, 3, 4))
    denominator = np.sum(np.abs(truth) ** 2, axis=(1, 2, 3, 4)) + 1e-12
    return float(10.0 * np.log10(np.mean(numerator / denominator) + 1e-12))


def complex_soft_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = np.abs(z)
    scale = np.maximum(0.0, 1.0 - threshold / (magnitude + 1e-12))
    return z * scale


def denoise_proposal(current: np.ndarray, h_ls: np.ndarray, noise_sigma: float, step: int, steps: int) -> np.ndarray:
    progress = step / max(steps - 1, 1)
    # Use the LS estimate as a data-consistent anchor. Repeatedly applying
    # shrinkage to the previous estimate can over-suppress weak paths, so the
    # fixed-anchor target makes the 5-step baseline a genuine refinement rather
    # than just a harsher threshold.
    target_threshold = noise_sigma * (0.78 - 0.18 * progress)
    target = complex_soft_threshold(h_ls, target_threshold)
    local_threshold = noise_sigma * (1.05 - 0.35 * progress)
    local = complex_soft_threshold(current, local_threshold)
    anchored = 0.72 * target + 0.28 * local
    relaxation = 0.48 + 0.18 * progress
    return (1.0 - relaxation) * current + relaxation * anchored


def tile_mean(grid: np.ndarray, tile_d: int, tile_l: int) -> np.ndarray:
    """Average a [B, D, L] grid over DD tiles."""
    batch, n_doppler, n_delay = grid.shape
    assert n_doppler % tile_d == 0
    assert n_delay % tile_l == 0
    reshaped = grid.reshape(batch, n_doppler // tile_d, tile_d, n_delay // tile_l, tile_l)
    return reshaped.mean(axis=(2, 4))


def expand_tile_mask(mask: np.ndarray, tile_d: int, tile_l: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, tile_d, axis=1), tile_l, axis=2)


def run_denoiser(
    h_ls: np.ndarray,
    truth: np.ndarray,
    noise_sigma: float,
    *,
    steps: int,
    adaptive: bool,
    tau_scale: float = 0.75,
    use_change_score: bool = True,
    force_refresh_every: int = 0,
    cfg: SimConfig,
) -> dict[str, object]:
    start_time = time.perf_counter()
    current = h_ls.copy()
    previous = current.copy()
    active_history: list[float] = []
    last_active_tile_mask: np.ndarray | None = None

    for step in range(steps):
        proposal = denoise_proposal(current, h_ls, noise_sigma, step, steps)
        if not adaptive or step < cfg.adaptive_warmup_steps:
            current_next = proposal
            active_grid = np.ones(current.shape[0:1] + current.shape[3:5], dtype=bool)
        else:
            amplitude_grid = np.mean(np.abs(current), axis=(1, 2))
            change_grid = np.mean(np.abs(current - previous), axis=(1, 2))
            amplitude_tile = tile_mean(amplitude_grid, cfg.tile_doppler, cfg.tile_delay)
            change_tile = tile_mean(change_grid, cfg.tile_doppler, cfg.tile_delay)

            # Low SNR should be conservative; high SNR can freeze more tiles.
            snr_aggression = np.clip(1.0 / max(noise_sigma, 1e-6), 0.75, 1.35)
            step_aggression = 0.70 + 0.18 * step
            amp_threshold = tau_scale * noise_sigma * snr_aggression * step_aggression
            change_threshold = 0.10 * noise_sigma * (1.0 + 0.20 * step)

            if use_change_score:
                active_tile = (amplitude_tile >= amp_threshold) | (change_tile >= change_threshold)
            else:
                active_tile = amplitude_tile >= amp_threshold

            if force_refresh_every > 0 and step % force_refresh_every == 0:
                active_tile = np.ones_like(active_tile, dtype=bool)

            active_grid = expand_tile_mask(active_tile, cfg.tile_doppler, cfg.tile_delay)
            active_broadcast = active_grid[:, None, None, :, :]
            current_next = np.where(active_broadcast, proposal, current)
            last_active_tile_mask = active_tile

        previous = current
        current = current_next
        active_history.append(float(active_grid.mean()))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / h_ls.shape[0]
    return {
        "estimate": current,
        "nmse_db": nmse_db(current, truth),
        "active_ratio": float(np.mean(active_history)),
        "per_frame_ms": elapsed_ms,
        "last_active_tile_mask": last_active_tile_mask,
        "active_history": np.asarray(active_history),
    }


def evaluate_all(cfg: SimConfig, out_dir: Path) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    truth = generate_sparse_dd_channels(cfg, rng)

    main_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    masks_for_plot: dict[int, np.ndarray] = {}
    full5_by_snr: dict[int, float] = {}

    for snr_db in cfg.snr_db_values:
        snr_rng = np.random.default_rng(cfg.seed + 1000 + snr_db)
        h_ls, noise_sigma = add_ls_noise(truth, snr_db, snr_rng)
        ls_nmse = nmse_db(h_ls, truth)

        full5 = run_denoiser(h_ls, truth, noise_sigma, steps=5, adaptive=False, cfg=cfg)
        full3 = run_denoiser(h_ls, truth, noise_sigma, steps=3, adaptive=False, cfg=cfg)
        full5_by_snr[snr_db] = float(full5["nmse_db"])

        main_rows.extend(
            [
                {
                    "method": "LS",
                    "steps": 0,
                    "snr_db": snr_db,
                    "nmse_db": ls_nmse,
                    "active_ratio": 1.0,
                    "per_frame_ms": 0.0,
                },
                {
                    "method": "Fixed sparse denoiser",
                    "steps": 5,
                    "snr_db": snr_db,
                    "nmse_db": full5["nmse_db"],
                    "active_ratio": full5["active_ratio"],
                    "per_frame_ms": full5["per_frame_ms"],
                },
                {
                    "method": "Fixed sparse denoiser",
                    "steps": 3,
                    "snr_db": snr_db,
                    "nmse_db": full3["nmse_db"],
                    "active_ratio": full3["active_ratio"],
                    "per_frame_ms": full3["per_frame_ms"],
                },
            ]
        )

        for tau in cfg.tau_values:
            adaptive = run_denoiser(
                h_ls,
                truth,
                noise_sigma,
                steps=5,
                adaptive=True,
                tau_scale=tau,
                use_change_score=True,
                force_refresh_every=0,
                cfg=cfg,
            )
            sweep_rows.append(
                {
                    "tau": tau,
                    "snr_db": snr_db,
                    "nmse_db": adaptive["nmse_db"],
                    "loss_vs_full5_db": float(adaptive["nmse_db"]) - float(full5["nmse_db"]),
                    "active_ratio": adaptive["active_ratio"],
                    "inactive_ratio": 1.0 - float(adaptive["active_ratio"]),
                    "per_frame_ms": adaptive["per_frame_ms"],
                }
            )

    tau_summary = []
    for tau in cfg.tau_values:
        rows = [row for row in sweep_rows if row["tau"] == tau]
        tau_summary.append(
            {
                "tau": tau,
                "mean_loss_db": float(np.mean([row["loss_vs_full5_db"] for row in rows])),
                "mean_active_ratio": float(np.mean([row["active_ratio"] for row in rows])),
                "mean_inactive_ratio": float(np.mean([row["inactive_ratio"] for row in rows])),
            }
        )
    feasible = [row for row in tau_summary if row["mean_loss_db"] <= cfg.target_loss_db]
    if feasible:
        selected_tau = max(feasible, key=lambda row: row["mean_inactive_ratio"])["tau"]
    else:
        selected_tau = min(tau_summary, key=lambda row: row["mean_loss_db"])["tau"]

    for snr_db in cfg.snr_db_values:
        snr_rng = np.random.default_rng(cfg.seed + 1000 + snr_db)
        h_ls, noise_sigma = add_ls_noise(truth, snr_db, snr_rng)
        adaptive = run_denoiser(
            h_ls,
            truth,
            noise_sigma,
            steps=5,
            adaptive=True,
            tau_scale=float(selected_tau),
            use_change_score=True,
            force_refresh_every=0,
            cfg=cfg,
        )
        main_rows.append(
            {
                "method": "Adaptive DD inference",
                "steps": 5,
                "snr_db": snr_db,
                "nmse_db": adaptive["nmse_db"],
                "active_ratio": adaptive["active_ratio"],
                "per_frame_ms": adaptive["per_frame_ms"],
            }
        )
        if adaptive["last_active_tile_mask"] is not None:
            masks_for_plot[snr_db] = np.asarray(adaptive["last_active_tile_mask"], dtype=float).mean(axis=0)

        for name, use_change, refresh in [
            ("Ours", True, 0),
            ("Forced refresh K=3", True, 3),
            ("No residual-change score", False, 0),
            ("Amplitude only + refresh", False, 3),
        ]:
            result = run_denoiser(
                h_ls,
                truth,
                noise_sigma,
                steps=5,
                adaptive=True,
                tau_scale=float(selected_tau),
                use_change_score=use_change,
                force_refresh_every=refresh,
                cfg=cfg,
            )
            ablation_rows.append(
                {
                    "variant": name,
                    "snr_db": snr_db,
                    "nmse_db": result["nmse_db"],
                    "loss_vs_full5_db": float(result["nmse_db"]) - full5_by_snr[snr_db],
                    "active_ratio": result["active_ratio"],
                    "inactive_ratio": 1.0 - float(result["active_ratio"]),
                }
            )

    write_csv(out_dir / "main_results.csv", main_rows)
    write_csv(out_dir / "threshold_sweep.csv", sweep_rows)
    write_csv(out_dir / "tau_summary.csv", tau_summary)
    write_csv(out_dir / "ablation.csv", ablation_rows)

    return {
        "truth": truth,
        "main_rows": main_rows,
        "sweep_rows": sweep_rows,
        "tau_summary": tau_summary,
        "selected_tau": selected_tau,
        "ablation_rows": ablation_rows,
        "masks_for_plot": masks_for_plot,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pivot_main_rows(rows: list[dict[str, object]], snrs: tuple[int, ...]) -> list[dict[str, object]]:
    order = [
        ("LS", 0),
        ("Fixed sparse denoiser", 5),
        ("Fixed sparse denoiser", 3),
        ("Adaptive DD inference", 5),
    ]
    pivoted = []
    for method, steps in order:
        selected = [row for row in rows if row["method"] == method and row["steps"] == steps]
        if not selected:
            continue
        item: dict[str, object] = {"method": method, "steps": steps}
        for snr in snrs:
            match = [row for row in selected if row["snr_db"] == snr]
            item[f"nmse_{snr}"] = float(match[0]["nmse_db"]) if match else float("nan")
        item["active_ratio"] = float(np.mean([row["active_ratio"] for row in selected]))
        item["per_frame_ms"] = float(np.mean([row["per_frame_ms"] for row in selected]))
        pivoted.append(item)
    return pivoted


def write_latex_tables_and_macros(out_dir: Path, results: dict[str, object], cfg: SimConfig) -> None:
    rows = pivot_main_rows(results["main_rows"], cfg.snr_db_values)
    table_path = out_dir / "table_main_results.tex"
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by scripts/run_preliminary_adaptive_dd.py\n")
        handle.write("\\begin{tabular}{lccccc}\n")
        handle.write("\\toprule\n")
        handle.write("Method & Steps & NMSE@0 dB & NMSE@5 dB & NMSE@10 dB & Active comp. \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            steps = "--" if row["steps"] == 0 else str(row["steps"])
            handle.write(
                f"{row['method']} & {steps} & "
                f"{row['nmse_0']:.2f} & {row['nmse_5']:.2f} & {row['nmse_10']:.2f} & "
                f"{100.0 * row['active_ratio']:.1f}\\% \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")

    tau = float(results["selected_tau"])
    adaptive_rows = [
        row for row in rows if row["method"] == "Adaptive DD inference" and row["steps"] == 5
    ]
    fixed_rows = [row for row in rows if row["method"] == "Fixed sparse denoiser" and row["steps"] == 5]
    if adaptive_rows and fixed_rows:
        adaptive = adaptive_rows[0]
        fixed = fixed_rows[0]
        losses = [
            float(adaptive[f"nmse_{snr}"]) - float(fixed[f"nmse_{snr}"]) for snr in cfg.snr_db_values
        ]
        mean_loss = float(np.mean(losses))
        mean_active = float(adaptive["active_ratio"])
        mean_inactive = 1.0 - mean_active
    else:
        mean_loss = float("nan")
        mean_active = float("nan")
        mean_inactive = float("nan")

    with (out_dir / "preliminary_macros.tex").open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated by scripts/run_preliminary_adaptive_dd.py\n")
        handle.write(f"\\newcommand{{\\PrelimTau}}{{{tau:.2f}}}\n")
        handle.write(f"\\newcommand{{\\PrelimMeanLoss}}{{{mean_loss:.2f}}}\n")
        handle.write(f"\\newcommand{{\\PrelimMeanActive}}{{{100.0 * mean_active:.1f}\\%}}\n")
        handle.write(f"\\newcommand{{\\PrelimMeanInactive}}{{{100.0 * mean_inactive:.1f}\\%}}\n")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    x_label: str,
    y_label: str,
    title: str,
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    draw.rectangle((left, top, right, bottom), outline=(30, 30, 30), width=2)
    draw.text(((left + right) // 2 - 140, bottom + 42), x_label, fill=(20, 20, 20), font=font)
    draw.text((left, top - 58), title, fill=(20, 20, 20), font=font)
    draw.text((left - 78, top + 8), y_label, fill=(20, 20, 20), font=small)


def map_point(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, right, bottom = box
    px = left + int((x - x_min) / max(x_max - x_min, 1e-9) * (right - left))
    py = bottom - int((y - y_min) / max(y_max - y_min, 1e-9) * (bottom - top))
    return px, py


def color_ramp(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]
    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:]):
        if x0 <= value <= x1:
            alpha = (value - x0) / (x1 - x0)
            return tuple(int(c0[i] * (1 - alpha) + c1[i] * alpha) for i in range(3))
    return stops[-1][1]


def plot_results(fig_dir: Path, results: dict[str, object], cfg: SimConfig) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = results["sweep_rows"]
    font = load_font(24)
    small = load_font(18)
    tiny = load_font(15)
    palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44)]

    # NMSE loss versus active computation.
    image = Image.new("RGB", (1180, 760), "white")
    draw = ImageDraw.Draw(image)
    box = (120, 90, 1060, 600)
    all_x = [100.0 * row["active_ratio"] for row in sweep_rows]
    all_y = [row["loss_vs_full5_db"] for row in sweep_rows] + [cfg.target_loss_db]
    x_min = max(0.0, min(all_x) - 5.0)
    x_max = min(105.0, max(all_x) + 5.0)
    y_min = min(-0.05, min(all_y) - 0.05)
    y_max = max(0.55, max(all_y) + 0.10)
    draw_axes(
        draw,
        box,
        x_label="Active DD computation ratio (%)",
        y_label="NMSE loss (dB)",
        title="Threshold sweep: NMSE loss vs. active DD computation",
        font=font,
        small=small,
    )
    for frac in np.linspace(0, 1, 6):
        x_val = x_min + frac * (x_max - x_min)
        px, _ = map_point(x_val, y_min, x_min, x_max, y_min, y_max, box)
        draw.line((px, box[1], px, box[3]), fill=(230, 230, 230), width=1)
        draw.text((px - 20, box[3] + 8), f"{x_val:.0f}", fill=(60, 60, 60), font=tiny)
    for frac in np.linspace(0, 1, 6):
        y_val = y_min + frac * (y_max - y_min)
        _, py = map_point(x_min, y_val, x_min, x_max, y_min, y_max, box)
        draw.line((box[0], py, box[2], py), fill=(230, 230, 230), width=1)
        draw.text((box[0] - 72, py - 10), f"{y_val:.2f}", fill=(60, 60, 60), font=tiny)
    gate_y = map_point(x_min, cfg.target_loss_db, x_min, x_max, y_min, y_max, box)[1]
    draw.line((box[0], gate_y, box[2], gate_y), fill=(190, 45, 45), width=3)
    draw.text((box[2] - 150, gate_y - 28), "0.3 dB gate", fill=(190, 45, 45), font=tiny)
    for snr_db in cfg.snr_db_values:
        rows = [row for row in sweep_rows if row["snr_db"] == snr_db]
        color = palette[list(cfg.snr_db_values).index(snr_db) % len(palette)]
        points = [
            map_point(100.0 * row["active_ratio"], row["loss_vs_full5_db"], x_min, x_max, y_min, y_max, box)
            for row in rows
        ]
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for point, row in zip(points, rows):
            px, py = point
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline=(20, 20, 20))
            draw.text((px + 8, py - 16), f"{row['tau']:.2g}", fill=color, font=tiny)
        legend_x = 850
        legend_y = 115 + 32 * list(cfg.snr_db_values).index(snr_db)
        draw.line((legend_x, legend_y + 10, legend_x + 40, legend_y + 10), fill=color, width=4)
        draw.text((legend_x + 52, legend_y), f"{snr_db} dB", fill=(20, 20, 20), font=small)
    image.save(fig_dir / "nmse_vs_active_ratio.png")

    masks = results["masks_for_plot"]
    if masks:
        snrs = list(cfg.snr_db_values)
        cell = 28
        panel_w = (cfg.n_delay // cfg.tile_delay) * cell
        panel_h = (cfg.n_doppler // cfg.tile_doppler) * cell
        image = Image.new("RGB", (80 + len(snrs) * (panel_w + 70), panel_h + 130), "white")
        draw = ImageDraw.Draw(image)
        draw.text((40, 20), "Last-step active DD tile probability", fill=(20, 20, 20), font=font)
        for idx, snr_db in enumerate(snrs):
            mask = masks.get(snr_db)
            if mask is None:
                continue
            x0 = 45 + idx * (panel_w + 70)
            y0 = 75
            draw.text((x0, y0 - 32), f"{snr_db} dB", fill=(20, 20, 20), font=small)
            for d_idx in range(mask.shape[0]):
                for l_idx in range(mask.shape[1]):
                    color = color_ramp(float(mask[d_idx, l_idx]))
                    rect = (
                        x0 + l_idx * cell,
                        y0 + d_idx * cell,
                        x0 + (l_idx + 1) * cell - 1,
                        y0 + (d_idx + 1) * cell - 1,
                    )
                    draw.rectangle(rect, fill=color)
            draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline=(40, 40, 40), width=2)
            draw.text((x0 + panel_w // 2 - 42, y0 + panel_h + 12), "Delay tile", fill=(60, 60, 60), font=tiny)
            draw.text((x0 - 8, y0 + panel_h + 12), "Doppler", fill=(60, 60, 60), font=tiny)
        image.save(fig_dir / "skip_mask_heatmap.png")

    ablation_rows = results["ablation_rows"]
    variants = list(dict.fromkeys(row["variant"] for row in ablation_rows))
    image = Image.new("RGB", (1220, 760), "white")
    draw = ImageDraw.Draw(image)
    box = (120, 80, 1080, 570)
    losses_all = [row["loss_vs_full5_db"] for row in ablation_rows] + [cfg.target_loss_db]
    y_min = min(-0.05, min(losses_all) - 0.05)
    y_max = max(0.55, max(losses_all) + 0.10)
    draw_axes(
        draw,
        box,
        x_label="Ablation variant",
        y_label="NMSE loss (dB)",
        title="Ablation: loss relative to fixed 5-step denoising",
        font=font,
        small=small,
    )
    for frac in np.linspace(0, 1, 6):
        y_val = y_min + frac * (y_max - y_min)
        _, py = map_point(0, y_val, 0, 1, y_min, y_max, box)
        draw.line((box[0], py, box[2], py), fill=(230, 230, 230), width=1)
        draw.text((box[0] - 72, py - 10), f"{y_val:.2f}", fill=(60, 60, 60), font=tiny)
    gate_y = map_point(0, cfg.target_loss_db, 0, 1, y_min, y_max, box)[1]
    draw.line((box[0], gate_y, box[2], gate_y), fill=(190, 45, 45), width=3)
    group_w = (box[2] - box[0]) / len(variants)
    bar_w = group_w / (len(cfg.snr_db_values) + 1)
    zero_y = map_point(0, 0.0, 0, 1, y_min, y_max, box)[1]
    for variant_idx, variant in enumerate(variants):
        group_center = box[0] + group_w * (variant_idx + 0.5)
        draw.text((int(group_center - 90), box[3] + 16), variant, fill=(50, 50, 50), font=tiny)
        for offset_idx, snr_db in enumerate(cfg.snr_db_values):
            rows = [row for row in ablation_rows if row["snr_db"] == snr_db and row["variant"] == variant]
            if not rows:
                continue
            loss = float(rows[0]["loss_vs_full5_db"])
            x0 = int(group_center - 1.5 * bar_w + offset_idx * bar_w)
            x1 = int(x0 + bar_w * 0.82)
            y = map_point(0, loss, 0, 1, y_min, y_max, box)[1]
            color = palette[offset_idx % len(palette)]
            draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill=color, outline=(35, 35, 35))
    for offset_idx, snr_db in enumerate(cfg.snr_db_values):
        color = palette[offset_idx % len(palette)]
        lx = 850
        ly = 112 + 32 * offset_idx
        draw.rectangle((lx, ly, lx + 34, ly + 18), fill=color)
        draw.text((lx + 48, ly - 4), f"{snr_db} dB", fill=(20, 20, 20), font=small)
    image.save(fig_dir / "ablation_loss.png")


def write_summary(out_dir: Path, results: dict[str, object], cfg: SimConfig) -> None:
    summary_path = out_dir / "summary.md"
    tau = float(results["selected_tau"])
    tau_summary = [row for row in results["tau_summary"] if row["tau"] == tau][0]
    rows = pivot_main_rows(results["main_rows"], cfg.snr_db_values)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Preliminary Adaptive DD Inference Summary\n\n")
        handle.write("This is a surrogate sparse-DD experiment, not the final trained checkpoint result.\n\n")
        handle.write(f"- Selected tau: `{tau:.2f}`\n")
        handle.write(f"- Mean inactive DD computation: `{100.0 * tau_summary['mean_inactive_ratio']:.1f}%`\n")
        handle.write(f"- Mean NMSE loss vs fixed 5-step: `{tau_summary['mean_loss_db']:.2f} dB`\n")
        handle.write(f"- Target gate: `{cfg.target_loss_db:.2f} dB`\n\n")
        handle.write("## Main Results\n\n")
        handle.write("| Method | Steps | NMSE@0 | NMSE@5 | NMSE@10 | Active comp. |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            steps = "--" if row["steps"] == 0 else str(row["steps"])
            handle.write(
                f"| {row['method']} | {steps} | {row['nmse_0']:.2f} | "
                f"{row['nmse_5']:.2f} | {row['nmse_10']:.2f} | "
                f"{100.0 * row['active_ratio']:.1f}% |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=SimConfig.batch_size)
    parser.add_argument("--n-doppler", type=int, default=SimConfig.n_doppler)
    parser.add_argument("--n-delay", type=int, default=SimConfig.n_delay)
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        batch_size=args.batch_size,
        n_doppler=args.n_doppler,
        n_delay=args.n_delay,
        seed=args.seed,
    )
    assert cfg.n_doppler % cfg.tile_doppler == 0
    assert cfg.n_delay % cfg.tile_delay == 0

    results = evaluate_all(cfg, args.out_dir)
    write_latex_tables_and_macros(args.out_dir, results, cfg)
    plot_results(args.fig_dir, results, cfg)
    write_summary(args.out_dir, results, cfg)

    selected_tau = float(results["selected_tau"])
    selected_summary = [row for row in results["tau_summary"] if row["tau"] == selected_tau][0]
    print("Preliminary adaptive-DD experiment complete")
    print(f"  selected tau: {selected_tau:.2f}")
    print(f"  mean inactive ratio: {100.0 * selected_summary['mean_inactive_ratio']:.1f}%")
    print(f"  mean NMSE loss: {selected_summary['mean_loss_db']:.2f} dB")
    print(f"  results: {args.out_dir}")
    print(f"  figures: {args.fig_dir}")


if __name__ == "__main__":
    main()
