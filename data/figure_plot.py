#!/usr/bin/env python3
"""
PPG R-matrix + closed-form SaO2/SvO2 from 660/940 nm
No CLI: configure everything in the USER CONFIG block below.
"""

from __future__ import annotations
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt


# ============================== USER CONFIG ==============================

# --- Files & layout ---
CSV_PATH       = "data/breath_hold_20251021_124200.csv"     # <-- set your input CSV file path
TIME_COL_NAME  = "timestamp_iso"        # column in the raw CSV if present (ok to leave missing)

# --- Sampling & windowing ---
FS             = 25.0                   # Hz
WIN_SEC        = 3.0                    # seconds
STEP_SEC       = 1.0                    # seconds
CHANNELS_1BASED = tuple(range(1, 13))   # which locations to include (loc_1..loc_12)

# --- Filtering for AC extraction ---
LOWCUT_HZ      = 0.5
HIGHCUT_HZ     = 3.0

# --- Locations to treat as artery/vein (must match column names in R_MATRIX) ---
LOC_ART        = "loc_9"
LOC_VEIN       = "loc_4"

# --- Known arterial saturation (fraction 0..1), e.g., from finger-clip reference ---
SA_KNOWN       = 0.98

# --- Light smoothing for the per-window pathlength ratio rho (odd integer; 1 disables) ---
SMOOTH_RHO_N   = 5

# --- Plots & outputs ---
MAKE_PLOTS     = True
DEBUG_PLOTS    = True
R_MATRIX_OUT   = "R_MATRIX_R660_940.csv"
OXY_OUT        = "oxygenation_timeseries_closed_form.csv"
R_PLOT_FILE    = "R_timeseries_seconds"
OXY_PLOT_FILE  = "Sa_Sv_timeseries.png"

# --- Extinction coefficients (cm^-1/M, decadic) ---
# Replace with your table if different. Keeping these so the script runs end-to-end.
EPS = {
    660: {"Hb": 3226.56, "HbO2": 319.6},
    850: {"Hb": 691.32,  "HbO2": 1058.0},
    940: {"Hb": 693.44,  "HbO2": 1214.0},
}


# ============================== HELPERS ==============================

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 3):
    nyq = 0.5 * fs
    wn = [lowcut / nyq, highcut / nyq]
    return butter(order, wn, btype="band")

def _robust_half_pp(x: np.ndarray) -> float:
    """Robust half peak-to-peak using 5–95th percentiles."""
    p5, p95 = np.percentile(x, [5, 95])
    return 0.5 * (p95 - p5)

def _bandpass_all(X: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Band-pass filter all columns (time on axis 0)."""
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, X, axis=0)


# ======================== R COMPUTATION (660/940) ========================

def compute_windowed_R(
    chn660: pd.DataFrame,
    chn940: pd.DataFrame,
    fs: float,
    win_sec: float,
    step_sec: float,
    channels_1based: tuple[int, ...],
    lowcut: float,
    highcut: float,
    time_col: pd.Series | None = None,
    include_partial: bool = False,
) -> pd.DataFrame:
    """
    Windowed R = (AC/DC)_660 / (AC/DC)_940.
    Returns tidy DataFrame: [time_center_iso, t_center_s, loc, R_660_940]
    """
    X660 = chn660.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").to_numpy()
    X940 = chn940.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").to_numpy()
    assert X660.shape == X940.shape, "660 and 940 arrays must have the same shape"

    # band-pass once for AC
    X660_bp = _bandpass_all(X660, fs, lowcut, highcut)
    X940_bp = _bandpass_all(X940, fs, lowcut, highcut)

    T, n_ch = X660.shape
    w = int(round(win_sec * fs))
    s = int(round(step_sec * fs))
    if w <= 0 or s <= 0:
        raise ValueError("win_sec and step_sec must be > 0")

    idxs = [c - 1 for c in channels_1based]  # 1-based -> 0-based
    for i in idxs:
        if not (0 <= i < n_ch):
            raise ValueError(f"Channel {i+1} out of range for {n_ch}-channel data")

    starts = range(0, T - (0 if include_partial else w) + 1, s)
    rows = []
    for start in starts:
        end = min(start + w, T)
        if end - start < w and not include_partial:
            break
        cidx = start + (end - start) // 2
        t_center_s = cidx / fs

        # Best-effort ISO timestamp at the center
        if (time_col is not None) and (len(time_col) == T):
            try:
                t_center_iso = pd.to_datetime(time_col.iloc[cidx]).tz_localize(None)
                t_center_iso = pd.to_datetime(t_center_iso).isoformat(timespec="milliseconds")
            except Exception:
                t_center_iso = None
        else:
            t_center_iso = None

        for i in idxs:
            ac660 = _robust_half_pp(X660_bp[start:end, i])
            ac940 = _robust_half_pp(X940_bp[start:end, i])
            dc660 = float(np.mean(X660[start:end, i]))
            dc940 = float(np.mean(X940[start:end, i]))
            if abs(dc660) < 1e-12: dc660 = 1e-12 if dc660 >= 0 else -1e-12
            if abs(dc940) < 1e-12: dc940 = 1e-12 if dc940 >= 0 else -1e-12

            r660 = ac660 / dc660
            r940 = ac940 / dc940
            R = r660 / r940

            rows.append({
                "time_center_iso": t_center_iso,
                "t_center_s": t_center_s,
                "loc": f"loc_{i+1}",
                "R_660_940": R,
            })

    return pd.DataFrame(rows)

def make_R_matrix(winR: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot tidy windowed R into a wide matrix with rows=windows, cols=loc_*.
    Columns in result: time_center_iso, t_center_s, loc_*...
    """
    R_wide = (winR.pivot_table(index="t_center_s", columns="loc", values="R_660_940", aggfunc="mean")
                    .sort_index())
    # Map t_center_s -> time_center_iso (if present)
    tmap = (winR.dropna(subset=["time_center_iso"])
                 .sort_values("t_center_s")
                 .drop_duplicates("t_center_s")[["t_center_s", "time_center_iso"]])
    Rm = (R_wide.reset_index().merge(tmap, on="t_center_s", how="left"))

    # Order columns numerically: loc_1, loc_2, ...
    loc_cols = sorted([c for c in Rm.columns if str(c).startswith("loc_")],
                      key=lambda s: int(re.search(r"(\d+)$", s).group(1)))
    return Rm[["time_center_iso", "t_center_s"] + loc_cols]


# ===================== CLOSED-FORM OXYGENATION (660/940) =====================

def _ab(e):
    """Return a_λ = ε_Hb, b_λ = ε_HbO2 − ε_Hb for a wavelength entry e."""
    a = float(e["Hb"]); b = float(e["HbO2"]) - float(e["Hb"])
    return a, b

def rho_from_S_known(R: float, S_known: float, e_num: dict, e_den: dict) -> float:
    """
    Pathlength ratio ρ from measured R and known Sa:
      R = ρ * (aN + bN S) / (aD + bD S)  =>  ρ = R * (aD + bD S) / (aN + bN S)
    """
    aN, bN = _ab(e_num); aD, bD = _ab(e_den)
    denom = (aN + bN * S_known)
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12
    return R * (aD + bD * S_known) / denom

def S_from_R_and_rho(R: float, rho: float, e_num: dict, e_den: dict) -> float:
    """
    Closed-form saturation:
      S = (ρ*aN − R*aD) / (R*bD − ρ*bN)
    """
    aN, bN = _ab(e_num); aD, bD = _ab(e_den)
    denom = (R * bD - rho * bN)
    if abs(denom) < 1e-12:
        denom = 1e-12 if denom >= 0 else -1e-12
    S = (rho * aN - R * aD) / denom
    return float(np.clip(S, 0.0, 1.0))

def oxygenation_from_R_matrix(
    R_matrix: pd.DataFrame,
    eps: dict,
    loc_art: str,
    loc_vein: str,
    sa_known: float,
    smooth_rho_n: int = 3
) -> pd.DataFrame:
    """
    Per-window Sa_est and Sv_est using closed-form inversion, assuming
    the 660/940 pathlength ratio ρ derived from the artery applies to the vein.
    Returns: [t_center_s, R_art, R_vein, rho_hat, Sa_est, Sv_est]
    """
    e660, e940 = eps[660], eps[940]
    need = {"t_center_s", loc_art, loc_vein}
    if not need.issubset(R_matrix.columns):
        missing = need - set(R_matrix.columns)
        raise KeyError(f"R_matrix missing columns: {missing}")

    out = (R_matrix[["t_center_s", loc_art, loc_vein]]
           .rename(columns={loc_art: "R_art", loc_vein: "R_vein"})
           .copy())

    out["rho_hat"] = out["R_art"].astype(float).apply(
        lambda r: rho_from_S_known(r, sa_known, e_num=e660, e_den=e940) if pd.notna(r) else np.nan
    )
    if smooth_rho_n and smooth_rho_n > 1:
        out["rho_hat"] = out["rho_hat"].rolling(smooth_rho_n, center=True, min_periods=1).median()

    out["Sa_est"] = [
        S_from_R_and_rho(r, rho, e_num=e660, e_den=e940) if np.isfinite(r) and np.isfinite(rho) else np.nan
        for r, rho in zip(out["R_art"].to_numpy(float), out["rho_hat"].to_numpy(float))
    ]
    out["Sv_est"] = [
        S_from_R_and_rho(r, rho, e_num=e660, e_den=e940) if np.isfinite(r) and np.isfinite(rho) else np.nan
        for r, rho in zip(out["R_vein"].to_numpy(float), out["rho_hat"].to_numpy(float))
    ]
    return out


# ============================== PLOTTING ==============================

def plot_R_timeseries_seconds(winR: pd.DataFrame, smooth_n: int = 3, figsize=(7.5, 3.4), outfile_base: str | None = None):
    """R vs time (seconds), split by location."""
    if winR.empty:
        print("Nothing to plot: winR is empty.")
        return
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    mpl.rcParams["pdf.fonttype"] = 42; mpl.rcParams["ps.fonttype"] = 42
    dfp = winR.dropna(subset=["t_center_s"]).sort_values(["loc", "t_center_s"]).copy()
    ycol = "R_660_940"
    if smooth_n and smooth_n > 1:
        dfp["R_smooth"] = (dfp.groupby("loc", group_keys=False)["R_660_940"]
                               .apply(lambda s: s.rolling(smooth_n, center=True, min_periods=1).median()))
        ycol = "R_smooth"
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=dfp, x="t_center_s", y=ycol, hue="loc", linewidth=2.0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(r"$R = (AC/DC)_{660}/(AC/DC)_{940}$")
    ax.grid(True, alpha=0.3); sns.despine(trim=True)
    plt.tight_layout()
    if outfile_base:
        plt.savefig(f"{outfile_base}.png", dpi=300); plt.savefig(f"{outfile_base}.pdf")
    plt.show()

def plot_oxy(oxy_df: pd.DataFrame, sa_known: float, figsize=(7.5, 3.4), savepath: str | None = None):
    """Plot Sa_est and Sv_est vs time, with Sa known as dashed reference."""
    if oxy_df.empty:
        print("Nothing to plot: oxygenation dataframe is empty.")
        return
    plt.figure(figsize=figsize)
    plt.plot(oxy_df["t_center_s"], 100*oxy_df["Sa_est"], label="Sa_est (artery)")
    plt.plot(oxy_df["t_center_s"], 100*oxy_df["Sv_est"], label="Sv_est (vein)")
    plt.axhline(sa_known*100, linestyle="--", linewidth=1.2, label="Sa known")
    plt.xlabel("Time (s)"); plt.ylabel("Saturation (%)"); plt.legend(); plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


# ================================ MAIN ================================

def main():
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # Slice wavelength blocks (time × 12). Adjust if your CSV order differs.
    chn660 = df.iloc[:, :12]
    chn940 = df.iloc[:, 12:24]
    _chn850 = df.iloc[:, 24:36]  # reserved for future use

    if DEBUG_PLOTS:
        b, a = butter_bandpass(LOWCUT_HZ, HIGHCUT_HZ, FS, order=3)
        x = chn940.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").to_numpy()
        x_bp = filtfilt(b, a, x, axis=0)
        plt.figure()
        for i in range(x_bp.shape[1]):
            plt.plot(-x_bp[:, i], label=f"loc_{i+1}")
        plt.legend(); plt.title("940nm band-passed (debug)"); plt.show()

    time_series = df[TIME_COL_NAME] if TIME_COL_NAME in df.columns else None

    # 1) Windowed R (tidy)
    winR = compute_windowed_R(
        chn660=chn660, chn940=chn940,
        fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC,
        channels_1based=CHANNELS_1BASED,
        lowcut=LOWCUT_HZ, highcut=HIGHCUT_HZ,
        time_col=time_series, include_partial=False,
    )
    print("winR head:\n", winR.head())
    winR.to_csv("windowed_R.csv", index=False)

    # 2) R matrix (wide)
    R_MATRIX = make_R_matrix(winR)
    print("R_MATRIX shape:", R_MATRIX.shape)
    print(R_MATRIX.head())
    R_MATRIX.to_csv(R_MATRIX_OUT, index=False)

    # 3) Closed-form Sa/Sv
    oxy_out = oxygenation_from_R_matrix(
        R_matrix=R_MATRIX,
        eps=EPS,
        loc_art=LOC_ART,
        loc_vein=LOC_VEIN,
        sa_known=SA_KNOWN,
        smooth_rho_n=SMOOTH_RHO_N,
    )
    oxy_out.assign(Sa_pct=100*oxy_out["Sa_est"], Sv_pct=100*oxy_out["Sv_est"]).to_csv(
        OXY_OUT, index=False
    )
    print(f"Sa known: {SA_KNOWN:.3f} | Sa_est median: {np.nanmedian(oxy_out['Sa_est']):.3f} | "
          f"Sv_est median: {np.nanmedian(oxy_out['Sv_est']):.3f}")

    # 4) Plots
    if MAKE_PLOTS:
        plot_R_timeseries_seconds(winR, smooth_n=3, outfile_base=R_PLOT_FILE)
        plot_oxy(oxy_out, SA_KNOWN, savepath=OXY_PLOT_FILE)


    # extinction (cm^-1/M) you’re already using
    a660, b660 = EPS[660]["Hb"], EPS[660]["HbO2"] - EPS[660]["Hb"]
    a940, b940 = EPS[940]["Hb"], EPS[940]["HbO2"] - EPS[940]["Hb"]

    def rho_from_R_and_S(R, S):
        denom = (a660 + b660*S); 
        denom = denom if abs(denom) > 1e-12 else (1e-12*np.sign(denom if denom!=0 else 1.0))
        return R * (a940 + b940*S) / denom

    rho_art  = rho_from_R_and_S(R_MATRIX['loc_4'],  SA_KNOWN)
    rho_vein_as_if_Sa = rho_from_R_and_S(R_MATRIX['loc_9'], SA_KNOWN)  # just to compare geometry
    print("median ρ at artery:", np.nanmedian(rho_art))
    print("median ρ vein (assuming Sa):", np.nanmedian(rho_vein_as_if_Sa))



if __name__ == "__main__":
    main()
