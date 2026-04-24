#!/usr/bin/env python3
"""Effective STRF (eSTRF) reconstruction — *envelope-only* variant.

Why this script exists
----------------------
The original ``lingo_analysis.plot_estrf_reconstruction`` summed raw
``cos()`` carriers when synthesising the 2-D Gabor kernels.  That is
mathematically incorrect for the ``diffAudNeuro`` cortical frontend,
which takes the MAGNITUDE (or power) of each spectro-temporal wavelet
and is therefore *phase-invariant*.  Summing raw phase carriers across
many channels produces meaningless destructive interference — the
"barcode" striping you see in the v2 figure — that obscures the
underlying macro-receptive field the LIME coefficients actually encode.

The scientific fix
------------------
For a phase-invariant feature extractor, the right thing to draw at
each (rate ω, scale Ω) is the *modulation energy envelope*:

    K_{b,k}(t, f) = exp(-t² / 2σ_t²) · exp(-(f - f_b)² / 2σ_f²)

— i.e. just the 2-D Gaussian envelope of the Gabor, dropped at the
band's geometric-mean centre frequency f_b.  We then form the eSTRF as

    eSTRF(t, f) = Σ_{b, k} w_{b, k} · K_{b, k}(t, f)

with the *signed* LIME weight w (excitatory > 0 in red, inhibitory < 0
in blue).  A very light 2-D Gaussian post-smoothing blends the band
seams into a single coherent template.

This faithfully visualises *regions of expected modulation energy*
without faking anything — every weight is the real signed mean LIME
coefficient learnt for that phoneme; we have only swapped the carrier
for its envelope, which is the correct kernel for a magnitude-cortex
representation.

Usage::

    ./generate_estrf_plots.py                      # uses defaults
    ./generate_estrf_plots.py --phones s iy aa t   # custom subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# ────────────────────────────────────────────────────────────────────────
# Project imports — mirror the path-resolution that run.py uses so this
# script is launchable from anywhere on disk.
# ────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_r_candidate = _SCRIPT_DIR.parent / "r_code"
if not _r_candidate.is_dir():
    _r_candidate = _SCRIPT_DIR / "r_code"
if not _r_candidate.is_dir():
    _r_candidate = _SCRIPT_DIR
_R_CODE = _r_candidate.resolve()
sys.path.insert(0, str(_R_CODE))
sys.path.insert(0, str(_SCRIPT_DIR))

from lingo_analysis import (  # noqa: E402
    DEFAULT_BAND_EDGES_HZ,
    FAMILY_COLORS,
    FAMILY_OF,
    LimeResults,
    LINGUISTIC_BAND_NAMES,
    _band_freq_centers_oct,
    aggregate_signed_per_phoneme,
    load_lime_results,
    set_academic_style,
    _despine,
)


# ────────────────────────────────────────────────────────────────────────
# Project-canonical default paths (match run.py / hero script layout).
# ────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path("/Users/eminent/Projects/Cortical_Front")
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "r_code" / "analysis_outputs2"
_DEFAULT_RESULTS = _DEFAULT_OUTPUT_DIR / "results_raw.npz"


# ────────────────────────────────────────────────────────────────────────
# Envelope-only Gabor kernel and eSTRF reconstruction
# ────────────────────────────────────────────────────────────────────────

def _gabor_envelope_kernel(
    omega: float,
    Omega: float,
    t: np.ndarray,
    f_oct: np.ndarray,
    omega_floor_hz: float = 2.0,
    Omega_floor_cpo: float = 0.5,
    cycles: float = 0.5,
) -> np.ndarray:
    """Phase-invariant 2-D Gabor *envelope* kernel with **adaptive** width.

    The diffAudNeuro cortical wavelets are read out as magnitude / power,
    so the relevant template is the Gaussian envelope alone — without
    the temporal or spectral carrier — centred at (t = 0, f_oct = 0).

    Critical: in a modulation filterbank (and in real cortex) the envelope
    width must be **inversely proportional** to the channel's modulation
    rate ω and spectral scale Ω — the constant-Q / wavelet property.
    Otherwise high-rate channels (e.g. the burst of /t/, ~30 Hz) get
    smeared across hundreds of ms and the eSTRF degenerates into a
    blurry cloud.

    Adaptive widths used here::

        |ω̃| = max(|ω|, omega_floor_hz)        # cap to avoid huge envelopes
        |Ω̃| = max(|Ω|, Omega_floor_cpo)
        σ_t  = cycles / |ω̃|                    # seconds   (½-cycle by default)
        σ_f  = cycles / |Ω̃|                    # octaves

    Parameters
    ----------
    omega           : temporal rate ω in Hz (channel-specific).
    Omega           : spectral scale Ω in cycles/octave (channel-specific).
    t               : (n_t,) seconds, relative to centre.
    f_oct           : (n_f,) octaves, relative to the band centre.
    omega_floor_hz  : floor on |ω| to avoid pathological widths near DC.
    Omega_floor_cpo : floor on |Ω| (cycles per octave).
    cycles          : envelope half-width in carrier cycles. 0.5 ⇒ the
                      envelope contains roughly one full carrier period.

    Returns
    -------
    K : (n_t, n_f) real, non-negative — the adaptive 2-D Gaussian envelope.
    """
    abs_omega = max(abs(float(omega)), omega_floor_hz)
    abs_Omega = max(abs(float(Omega)), Omega_floor_cpo)
    sigma_t_s = cycles / abs_omega
    sigma_f_oct = cycles / abs_Omega
    env_t = np.exp(-(t / sigma_t_s) ** 2 / 2.0)
    env_f = np.exp(-(f_oct / sigma_f_oct) ** 2 / 2.0)
    return np.outer(env_t, env_f)


def reconstruct_estrf_envelope(
    coefs: np.ndarray,
    sr_pairs: np.ndarray,
    band_edges_hz: np.ndarray,
    n_t: int = 256,
    n_f: int = 160,
    t_window_s: float = 0.30,
    f_range_oct: Tuple[float, float] = (-1.0, 6.0),
    omega_floor_hz: float = 2.0,
    Omega_floor_cpo: float = 0.5,
    cycles: float = 0.5,
    smoothing_sigma: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise the modulation-energy eSTRF for one phoneme with
    **channel-adaptive** Gabor envelopes.

    For each (band b, STRF channel k) cell of the band-mode LIME
    coefficient matrix we stamp a 2-D Gaussian envelope at the band's
    geometric-mean frequency.  The envelope widths (σ_t, σ_f) are
    derived from this channel's own rate ω and scale Ω, so high-rate /
    high-scale channels produce narrow, impulsive templates and
    low-rate / low-scale channels produce broad ones — the correct
    constant-Q behaviour.  The stamp is weighted by the *signed* LIME
    coefficient and a light Gaussian post-filter (default σ=0.8 px)
    blends the band-edge seams.

    Returns
    -------
    estrf : (n_t, n_f) signed real-valued template
    t     : (n_t,) seconds
    f_oct : (n_f,) octaves above the bottom of the lowest band
    """
    n_bands, n_strfs = band_edges_hz.shape[0], sr_pairs.shape[0]
    coefs_2d = np.asarray(coefs, dtype=np.float64).reshape(n_bands, n_strfs)
    band_centres = _band_freq_centers_oct(band_edges_hz)
    # Column convention in sr_pairs: [scale Ω, rate ω].
    scales = np.asarray(sr_pairs[:, 0], dtype=np.float64)
    rates = np.asarray(sr_pairs[:, 1], dtype=np.float64)

    t = np.linspace(-t_window_s / 2.0, t_window_s / 2.0, n_t)
    f_oct = np.linspace(f_range_oct[0], f_range_oct[1], n_f)

    estrf = np.zeros((n_t, n_f), dtype=np.float64)
    for b in range(n_bands):
        for k in range(n_strfs):
            w = coefs_2d[b, k]
            if w == 0.0:
                continue
            kernel = _gabor_envelope_kernel(
                omega=rates[k],
                Omega=scales[k],
                t=t,
                f_oct=f_oct - band_centres[b],
                omega_floor_hz=omega_floor_hz,
                Omega_floor_cpo=Omega_floor_cpo,
                cycles=cycles,
            )
            estrf += w * kernel

    if smoothing_sigma and smoothing_sigma > 0:
        estrf = gaussian_filter(estrf, sigma=smoothing_sigma)
    return estrf, t, f_oct


# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────

def plot_estrf_envelope(
    res: LimeResults,
    phones: Sequence[str] = ("s", "aa", "iy", "t", "f", "m"),
    cols: int = 4,
    smoothing_sigma: float = 0.8,
    omega_floor_hz: float = 2.0,
    Omega_floor_cpo: float = 0.5,
    cycles: float = 0.5,
) -> plt.Figure:
    """Figure — envelope-only eSTRF reconstructions.

    The signed mean LIME coefficient per phoneme weights a 2-D Gaussian
    envelope (no phase carrier) at each (band, STRF-channel) location;
    the sum is lightly Gaussian-smoothed.  Excitatory regions appear
    red, inhibitory blue, on a zero-centred RdBu_r palette.
    """
    if not res.is_band_mode:
        raise ValueError(
            "Envelope eSTRF requires band-mode results (n_bands > 1)."
        )

    band_edges = (
        res.band_edges_hz if res.band_edges_hz is not None
        else DEFAULT_BAND_EDGES_HZ[: res.n_bands]
    )

    signed = aggregate_signed_per_phoneme(res)
    available = [p for p in phones if p in signed]
    if not available:
        raise ValueError(
            f"None of {list(phones)} have signed LIME coefficients."
        )

    # First pass: synthesise everything and find a joint colour scale.
    panels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    vmax = 0.0
    for p in available:
        estrf, t, f_oct = reconstruct_estrf_envelope(
            signed[p], res.sr_pairs, band_edges,
            omega_floor_hz=omega_floor_hz,
            Omega_floor_cpo=Omega_floor_cpo,
            cycles=cycles,
            smoothing_sigma=smoothing_sigma,
        )
        panels[p] = (estrf, t, f_oct)
        vmax = max(vmax, float(np.max(np.abs(estrf))))
    vmax = vmax + 1e-12

    # Reference frequency for octave→Hz tick mapping.
    base_hz = (
        max(1.0, float(band_edges[0, 0]))
        if band_edges[0, 0] > 0 else 125.0
    )

    n = len(available)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.6 * cols, 3.6 * rows),
        squeeze=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.22},
    )

    hz_ticks = [125, 250, 500, 1000, 2000, 4000, 8000]

    im = None
    for k, p in enumerate(available):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        estrf, t, f_oct = panels[p]
        im = ax.imshow(
            estrf.T, origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            extent=[t[0] * 1000.0, t[-1] * 1000.0,
                    float(f_oct[0]), float(f_oct[-1])],
        )
        # Time-zero reference.
        ax.axvline(0, color="black", lw=0.7, ls=":", alpha=0.7)

        # Dashed band-edge guides on the spectral axis.
        for (lo, hi) in band_edges:
            for edge in (lo, hi):
                if edge <= 0:
                    continue
                y = float(np.log2(edge / base_hz))
                if f_oct[0] <= y <= f_oct[-1]:
                    ax.axhline(
                        y, color="black", lw=0.4, ls="--", alpha=0.45,
                    )

        # Hz-labelled spectral axis (only on the first column for cleanliness).
        if c == 0:
            visible_hz = [h for h in hz_ticks
                          if f_oct[0] <= np.log2(h / base_hz) <= f_oct[-1]]
            yt_oct = [np.log2(h / base_hz) for h in visible_hz]
            ax.set_yticks(yt_oct)
            ax.set_yticklabels([f"{h}" for h in visible_hz])
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.set_yticks([])

        ax.set_xlabel("Time relative to centre (ms)")
        fam = FAMILY_OF.get(p, None)
        col = FAMILY_COLORS.get(fam, "black")
        ax.set_title(
            f"/{p}/  eSTRF (envelope)",
            color=col, fontsize=12, fontweight="bold", pad=6,
        )
        _despine(ax)

    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    fig.suptitle(
        "Effective STRF — Modulation-Energy Envelope Reconstruction",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.955,
        "Signed mean LIME coefficients projected onto phase-invariant "
        "Gabor envelopes  (red = excitatory, blue = inhibitory)",
        ha="center", va="top", fontsize=10, style="italic", color="#444444",
    )

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(),
        shrink=0.85, pad=0.025, aspect=28,
    )
    cbar.set_label(r"Signed weight  ($w_{b,k}$ · envelope)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)
    return fig


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(_DEFAULT_RESULTS),
                   help="Path to results_raw.npz (band-mode required).")
    p.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR),
                   help="Where to write the rendered PNG.")
    p.add_argument(
        "--phones", nargs="+",
        default=[
            # Same 20-phoneme set as Figure 1, in identical order so the
            # eSTRF panels line up phone-for-phone with the cochleagrams.
            "s", "sh", "f", "z",
            "t", "k", "p", "b",
            "aa", "ae", "ah", "uw",
            "iy", "ih", "eh", "uh",
            "m", "n", "ng", "er",
        ],
        help="TIMIT-39 phones to render (one panel each).",
    )
    p.add_argument("--cols", type=int, default=4,
                   help="Number of panel columns in the grid.")
    p.add_argument("--smoothing_sigma", type=float, default=0.8,
                   help="2-D Gaussian post-smoothing σ (pixels).")
    p.add_argument("--omega_floor_hz", type=float, default=2.0,
                   help="Floor on |ω| (Hz) when computing σ_t = cycles/|ω|.")
    p.add_argument("--Omega_floor_cpo", type=float, default=0.5,
                   help="Floor on |Ω| (cyc/oct) when computing σ_f = cycles/|Ω|.")
    p.add_argument("--cycles", type=float, default=0.5,
                   help="Envelope half-width in carrier cycles "
                        "(0.5 ⇒ ~one full carrier period across ±σ).")
    p.add_argument(
        "--output_name",
        default="fig4_estrf_reconstruction_v3_envelope.png",
        help="Output PNG filename (saved into --output_dir).",
    )
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(
            f"results_raw.npz not found at {results_path}.  Run run.py "
            f"once first to produce it."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / args.output_name

    print(f"Loading {results_path} ...")
    res = load_lime_results(str(results_path))
    print(f"  N={res.importances.shape[0]}  n_bands={res.n_bands}  "
          f"n_strfs={res.n_strfs}  band_mode={res.is_band_mode}")

    print(f"\nReconstructing envelope eSTRF for "
          f"{', '.join('/' + p + '/' for p in args.phones)} ...")
    set_academic_style()
    fig = plot_estrf_envelope(
        res,
        phones=tuple(args.phones),
        cols=args.cols,
        smoothing_sigma=args.smoothing_sigma,
        omega_floor_hz=args.omega_floor_hz,
        Omega_floor_cpo=args.Omega_floor_cpo,
        cycles=args.cycles,
    )
    fig.savefig(png_path)
    plt.close(fig)
    print(f"\nSaved → {png_path}")


if __name__ == "__main__":
    main()
