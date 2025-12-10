
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch, correlate, sosfilt
from dataclasses import dataclass
import pandas as pd
from matplotlib import pyplot as plt

# --------------------------
# Filtering & utilities
# --------------------------

def bandpass(x, fs, lo=0.5, hi=8.0, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def butter_bandpass_sos(fs, lo=0.5, hi=8.0, order=3):
    return butter(order, [lo/(fs/2), hi/(fs/2)], btype='band', output='sos')

def resample_to_len(x, L):
    xi = np.linspace(0, 1, num=len(x), endpoint=True)
    xo = np.linspace(0, 1, num=L, endpoint=True)
    return np.interp(xo, xi, x)

def normalize_beat(x, robust=True):
    """
    Normalize a beat to [0,1]. Robust percentile scaling by default to avoid spikes.
    Returns only the normalized beat (legacy signature).
    """
    x = np.asarray(x).astype(float)
    if robust:
        lo, hi = np.percentile(x, [5, 95])
    else:
        lo, hi = np.min(x), np.max(x)
    dyn = max(hi - lo, 1e-6)
    y = (x - lo) / dyn
    return np.clip(y, 0, 1)

def normalize_beat_01(x, robust=True):
    """
    Normalize a beat to [0,1] AND return dynamic range as an amplitude proxy.
    """
    x = np.asarray(x).astype(float)
    if robust:
        lo, hi = np.percentile(x, [5, 95])
    else:
        lo, hi = np.min(x), np.max(x)
    dyn = max(hi - lo, 1e-6)
    y = (x - lo) / dyn
    return np.clip(y, 0, 1), float(dyn)

def spectral_snr(x, fs, f_lo=0.6, f_hi=3.0):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 256))
    mask = (f >= f_lo) & (f <= f_hi)
    if not np.any(mask):
        return 0.0
    i_peak = mask.nonzero()[0][np.argmax(Pxx[mask])]
    signal_power = Pxx[i_peak]
    noise_power = np.sum(Pxx[mask]) - signal_power
    noise_power = max(noise_power, 1e-12)
    return 10*np.log10(signal_power / noise_power)

def xcorr_max(x, y):
    x0, y0 = x - np.mean(x), y - np.mean(y)
    denom = (np.linalg.norm(x0) * np.linalg.norm(y0)) + 1e-12
    c = correlate(x0, y0, mode='full') / denom
    lag = np.argmax(c) - (len(y)-1)
    return float(c.max()), int(lag)

def nrmse(x, y):
    return float(np.sqrt(np.mean((x - y)**2)) / (np.std(y) + 1e-6))

def _to_float_array(window):
    if window is Ellipsis or window is None:
        raise ValueError("Got Ellipsis/None for window – pass numeric samples.")
    arr = np.asarray(window, dtype=object)
    if arr.dtype == object and any(x is Ellipsis for x in np.ravel(arr)):
        raise ValueError("Window contains Ellipsis (...) – replace with real samples.")
    try:
        return arr.astype(float, copy=False).ravel()
    except Exception as e:
        raise ValueError(f"Window not numeric (dtype={arr.dtype}). Example element: {arr.flat[0]!r}") from e

# --------------------------
# Beat extraction & templates
# --------------------------

def extract_beats(signal, fs, min_hr=40, max_hr=180, detect="valleys"):
    """
    detect: "peaks" for standard PPG (positive pulses),
            "valleys" for raw intensity (systolic is a minimum).
    Returns: feet_idx, pulse_locs_idx, x_filt
    """
    x = bandpass(np.asarray(signal).astype(float), fs)
    min_dist = int(fs * 60.0 / max_hr)
    max_dist = int(fs * 60.0 / min_hr)

    # Use find_peaks on x for "peaks", or on -x for "valleys"
    x_for_detection = -x if detect == "valleys" else x
    prom = np.std(x_for_detection) * 0.3
    pulse_locs, _ = find_peaks(x_for_detection, distance=min_dist, prominence=prom)

    # "Foot" = local baseline just before the pulse:
    # - if detect="peaks": foot is a local MIN before the peak
    # - if detect="valleys": foot is a local MAX before the valley
    feet = []
    for p in pulse_locs:
        left = max(0, p - max_dist)
        seg = x[left:p] if p > left else np.array([x[p]])
        if len(seg) < 3:
            feet.append(max(p-1, 0))
        else:
            if detect == "valleys":
                feet.append(left + int(np.argmax(seg)))  # local max before valley
            else:
                feet.append(left + int(np.argmin(seg)))  # local min before peak
    return np.asarray(feet), np.asarray(pulse_locs), x

def build_template_beats(signal, fs, L=200, corr_thresh=0.8, max_beats=50, detect="valleys"):
    """
    Builds a positive-up morphology template even if input is raw intensity.
    If detect="valleys", each beat segment is flipped so systolic becomes positive.
    """
    feet, pulses, x = extract_beats(signal, fs, detect=detect)
    beats = []
    for i in range(len(feet)-1):
        seg = x[feet[i]:feet[i+1]]
        if len(seg) < int(0.25*fs):
            continue
        # Flip valleys to peaks so template is always positive-up
        if detect == "valleys":
            seg = -seg
        seg = resample_to_len(normalize_beat(seg), L)
        beats.append(seg)
        if len(beats) >= max_beats:
            break

    if not beats:
        raise ValueError("No usable beats found for template. "
                         "Try detect='peaks' or check filtering/HR bounds.")

    beats = np.vstack(beats)
    proto = np.median(beats, axis=0)

    # Correlation gating → robust median
    keep = [xcorr_max(b, proto)[0] >= corr_thresh for b in beats]
    if np.any(keep):
        template = np.median(beats[np.where(keep)[0]], axis=0)
    else:
        template = proto
    return template

def build_template_windows(signal, fs, win_sec=2.0, L=100, stride_sec=0.5):
    win = int(win_sec*fs)
    stride = int(stride_sec*fs)
    X = []
    for i in range(0, len(signal)-win+1, stride):
        seg = bandpass(signal[i:i+win], fs)
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
        X.append(resample_to_len(seg, L))
    if not X:
        raise ValueError("No windows for building a template.")
    return np.median(np.vstack(X), axis=0)

def score_against_template(seg, template, fs):
    """Window-based scoring (fallback)."""
    seg = resample_to_len(seg, len(template))
    seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)  # z-score
    tmpl = (template - np.mean(template)) / (np.std(template) + 1e-6)

    r, lag = xcorr_max(seg, tmpl)
    err = nrmse(seg, tmpl)
    snr_db = spectral_snr(seg, fs)

    r01 = 0.5*(r + 1.0)
    err01 = 1.0 - min(err/2.0, 1.0)
    snr01 = np.clip(snr_db/20.0, 0.0, 1.0)

    sqi = 100.0 * (0.50*r01 + 0.30*err01 + 0.20*snr01)
    return {
        "corr": float(r),
        "lag_samples": int(lag),
        "nrmse": float(err),
        "snr_db": float(snr_db),
        "sqi": float(sqi)
    }

# --------------------------
# Streaming beat matcher
# --------------------------

class BeatTemplateMatcher:
    """
    Stream-based beat SQI for RAW INTENSITY PPG (systolic is a VALLEY) or peaks.
    Segments beats as valley->valley (or peak->peak), flips sign so pulses are positive-up,
    normalizes to [0,1], resamples to template length, and scores.
    """
    def __init__(self, fs, template, L=None, min_hr=40, max_hr=180, prom_scale=0.3, detect="valleys"):
        self.fs = float(fs)
        self.template = np.asarray(template).astype(float)
        self.L = len(template) if L is None else int(L)
        self.min_dist = int(self.fs * 60.0 / max_hr)
        self.max_dist = int(self.fs * 60.0 / min_hr)
        self.prom_scale = prom_scale
        self.detect = detect
        self.sos = butter_bandpass_sos(self.fs, 0.5, 8.0, order=3)
        # streaming state
        self.zi  = np.zeros((self.sos.shape[0], 2), dtype=float)
        self.buf = np.empty(0, dtype=float)
        self.last_used_pulse = None  # index in the buffer of last pulse (valley/peak) used

        # precompute normalized template
        self.tmpl = (self.template - np.mean(self.template)) / (np.std(self.template) + 1e-6)

    def push(self, samples):
        """
        Feed new raw samples (1-D). Returns a list of per-beat dicts:
        {'start': i0, 'end': i1, 'corr': r, 'nrmse': e, 'sqi': s, 'amp_dyn': dyn, 'lag': lag}
        """
        x = _to_float_array(samples)

        # Filter with stateful SOS filter (streaming)
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)

        # Append to buffer
        self.buf = np.concatenate([self.buf, y])

        # Detect pulses (valleys for raw intensity, peaks otherwise)
        det_sig = -self.buf if self.detect == "valleys" else self.buf
        prom = np.std(det_sig) * self.prom_scale
        pulses, _ = find_peaks(det_sig, distance=self.min_dist, prominence=prom)

        out = []
        if len(pulses) < 2:
            return out

        # Start at first pulse after the last used (keep one back to complete a beat)
        start_k = 0
        if self.last_used_pulse is not None:
            later = np.where(pulses > self.last_used_pulse)[0]
            if later.size == 0:
                return out
            start_k = max(later[0] - 1, 0)

        for k in range(start_k, len(pulses) - 1):
            i0, i1 = int(pulses[k]), int(pulses[k+1])
            if i1 - i0 < int(0.25 * self.fs):
                continue
            seg = self.buf[i0:i1]
            if self.detect == "valleys":
                seg = -seg  # flip so systolic is positive-up

            seg01, dyn = normalize_beat_01(seg, robust=True)
            seg01 = resample_to_len(seg01, self.L)

            seg_z = (seg01 - np.mean(seg01)) / (np.std(seg01) + 1e-6)
            r, lag = xcorr_max(seg_z, self.tmpl)
            err   = nrmse(seg_z, self.tmpl)

            r01 = 0.5 * (r + 1.0)
            e01 = 1.0 - min(err / 2.0, 1.0)
            sqi = 100.0 * (0.7 * r01 + 0.3 * e01)

            out.append({
                "start": i0, "end": i1, "corr": r, "nrmse": err, "sqi": sqi,
                "amp_dyn": dyn, "lag": lag
            })

        self.last_used_pulse = int(pulses[-2])  # penultimate; last may be incomplete
        return out

# --------------------------
# Simple SQI engine (legacy API) – keep for compatibility
# --------------------------

@dataclass
class PPGTemplateSQI:
    fs: float
    mode: str = "beat"          # "beat" or "window"
    L: int = 200                # template length (beat)
    win_sec: float = 2.0        # template window size if mode="window"
    detect: str = "valleys"     # "valleys" for raw intensity, "peaks" for positive-going PPG
    template: np.ndarray = None

    def fit(self, signal):
        x = np.asarray(signal).astype(float)
        if self.mode == "beat":
            self.template = build_template_beats(x, self.fs, L=self.L, detect=self.detect)
        else:
            self.template = build_template_windows(x, self.fs, self.win_sec, L=self.L//2)
        return self

    def score_window(self, window):
        if self.template is None:
            raise RuntimeError("Call fit() first.")
        seg = _to_float_array(window)
        return score_against_template(seg, self.template, self.fs)

    def update_online(self, window, alpha=0.1, accept_threshold=80.0):
        """
        EMA update of the template when quality is high (window-based scoring).
        Prefer BeatTemplateMatcher for beat-synchronous scoring in new code.
        """
        metrics = self.score_window(window)
        if metrics["sqi"] >= accept_threshold:
            seg = resample_to_len((window - np.mean(window)) / (np.std(window)+1e-6),
                                  len(self.template))
            tmpl = (self.template - np.mean(self.template)) / (np.std(self.template)+1e-6)
            _, lag = xcorr_max(seg, tmpl)
            if lag != 0:
                pad = np.zeros_like(seg)
                if lag > 0:
                    pad[lag:] = seg[:-lag]
                else:
                    pad[:lag] = seg[-lag:]
                seg = pad
            self.template = (1 - alpha)*self.template + alpha*seg
        return metrics

# ----------------------------------------------------------------------------------
# Window helpers
# ----------------------------------------------------------------------------------

def chunk_windows(x, size=50):
    """
    Non-overlapping windows of length `size` (default 50 = 2 s @ 25 Hz).
    Drops trailing remainder. Works for 1D (N,) or time-first ND (N, ...).
    Returns array shape: (n_windows, size, *features)
    """
    x = np.asarray(x)
    n = x.shape[0]
    n_win = n // size
    if n_win == 0:
        return np.empty((0, size) + x.shape[1:], dtype=x.dtype)
    return x[:n_win*size].reshape(n_win, size, *x.shape[1:])

def sliding_windows(x, size=50, step=50):
    """
    Sliding windows of length `size` every `step` samples.
    """
    x = np.asarray(x)
    n = x.shape[0]
    if n < size:
        return np.empty((0, size) + x.shape[1:], dtype=x.dtype)
    idxs = range(0, n - size + 1, step)
    return np.stack([x[i:i+size] for i in idxs], axis=0)

# ----------------------------------------------------------------------------------
# --- Demo / Test harness ---
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Files & layout ---
    CSV_PATH       = "data/fingertip_2_20251024_151610.csv"   # <-- set your input CSV file path
    CSV_TEST_PATH  = "data/static_20251020_114822.csv"
    fs = 25.0  # Hz
    DETECT_MODE = "valleys"  # "valleys" for raw intensity; set to "peaks" if already absorption-like

    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # Slice wavelength blocks (time × 12). Adjust if your CSV order differs.
    chn660 = df.iloc[:, :12]
    chn940 = df.iloc[:, 12:24]
    # chn850 = df.iloc[:, 24:36]  # reserved

    # Choose a reference channel for template (here: 940 nm, column 7 as before)
    ref_data = chn940.iloc[:-200, 7].values  # drop the last 200 to avoid end-of-record artifacts
    print("Building beat template from reference data...")
    template = build_template_beats(ref_data, fs, L=200, detect=DETECT_MODE)

    # Optionally visualize the template
    try:
        plt.figure()
        plt.plot(template, lw=2)
        plt.title("Beat Template (positive-up)")
        plt.show()
    except Exception as e:
        print("Plotting skipped:", e)

    # Streaming matcher for beat-synchronous scoring
    matcher = BeatTemplateMatcher(fs=fs, template=template, detect=DETECT_MODE)

    # Test stream
    print(f"Reading {CSV_TEST_PATH} ...")
    df_test = pd.read_csv(CSV_TEST_PATH)
    chn940_test = df_test.iloc[:, 12:24]

    # Use 50-sample windows (2 s @ 25 Hz) for streaming
    windows = chunk_windows(chn940_test.iloc[:, 1].values, size=50)  # test same channel index
    print(f"Streaming {len(windows)} windows of 50 samples...")

    for i, incoming_window in enumerate(windows):
        # Beat-segmentation & scoring
        results = matcher.push(incoming_window)
        if results:
            for r in results:
                print(f"beat[{r['start']}:{r['end']}] SQI={r['sqi']:.1f} corr={r['corr']:.2f} "
                      f"nrmse={r['nrmse']:.2f} amp_dyn={r['amp_dyn']:.3f}")
        else:
            # Fallback: window-based SQI if no complete beat in this chunk
            fallback = score_against_template(incoming_window, template, fs)
            print(f"window[{i}] (fallback) SQI={fallback['sqi']:.1f} corr={fallback['corr']:.2f} "
                  f"nrmse={fallback['nrmse']:.2f} SNR={fallback['snr_db']:.1f} dB")

        # Optional quick plot of the incoming (bandpassed) window
        if i < 1:  # only plot first few
            try:
                bp = bandpass(incoming_window, fs, lo=0.5, hi=3.0, order=3)
                plt.figure()
                plt.plot(bp, lw=1.5)
                plt.title(f"Incoming window {i} (bandpassed)")
                plt.show()
            except Exception as e:
                print("Plotting skipped:", e)
