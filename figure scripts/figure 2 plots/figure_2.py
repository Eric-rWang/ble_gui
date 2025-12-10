import csv
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

import pandas as pd
import matplotlib.pyplot as plt

def read_csv_as_df(filepath, **kwargs):
    df = pd.read_csv(filepath, **kwargs)
    return df

def bandpass_filter(x, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    return y

def detect_ppg_peaks(x, fs, min_hr=40, max_hr=100, prominence=0.2):
    x = np.asarray(x)
    min_distance = int(fs * 60.0 / max_hr)  # in samples
    peaks, _ = find_peaks(x, distance=min_distance, prominence=prominence)
    return peaks

def similarity_color(ref_beat, beat):
    """
    +1 corr -> red, -1 corr -> blue, 0 -> purple.
    """
    ref = np.asarray(ref_beat)
    x   = np.asarray(beat)

    ref_z = (ref - ref.mean()) / (ref.std() + 1e-12)
    x_z   = (x   - x.mean())   / (x.std()   + 1e-12)

    corr = np.corrcoef(ref_z, x_z)[0, 1]
    corr = np.clip(corr, -1.0, 1.0)

    # Map [-1, 1] -> [0, 1]
    t = (corr + 1.0) / 2.0

    # Interpolate between blue (0,0,1) and red (1,0,0)
    color = (t, 0.0, 1.0 - t)
    return color, corr


if __name__ == "__main__":
    fs = 25.0  # Sampling frequency in Hz
    n_points = 200  # Number of points to normalize each beat to
    
    df = read_csv_as_df("./figure scripts/figure 2 plots/wrist_ppg_occ.csv")
    df_data = df.iloc[:, :36]
    
    bp_df = df_data.apply(lambda col: bandpass_filter(col.to_numpy(), fs, 0.5, 3.0))

    pre_occ = bp_df.iloc[:700, :]
    occ = bp_df.iloc[1150:1600, :]
    post_occ = bp_df.iloc[2000:, :]

    curr_data = post_occ

    # Optional quick look
    plt.figure(figsize=(10, 6))
    for ch in range(12):
        plt.plot(-curr_data.iloc[:, ch], label=f"Ch {ch}")
    plt.legend()
    plt.ylabel("PPG Abs")
    plt.title("Pre-occlusion PPG (first 12 channels)")
    plt.tight_layout()
    plt.show()

    # --- Build reference beat from one channel (e.g., channel 1) ---
    # data_ref = curr_data.iloc[:, 1].to_numpy() # Pre Occ
    # data_ref = curr_data.iloc[:, 7].to_numpy()  # Occlusion
    data_ref = curr_data.iloc[:, 1].to_numpy() # Post Occ
    peaks = detect_ppg_peaks(data_ref, fs)

    beats_ref = []
    for k in range(len(peaks) - 1):
        start = peaks[k]
        end = peaks[k + 1]

        segment = -1 * data_ref[start:end]
        if len(segment) < 5:
            continue

        old_idx = np.arange(len(segment))
        new_idx = np.linspace(0, len(segment) - 1, n_points)
        resampled = np.interp(new_idx, old_idx, segment)
        beats_ref.append(resampled)

    if len(beats_ref) == 0:
        raise ValueError("No valid beats found for reference. Check peak detection params.")

    beats_ref = np.vstack(beats_ref)
    ref_beat = beats_ref.mean(axis=0)
    t_norm = np.linspace(0, 1, n_points)

    # --- For each channel: compute mean beat, color by similarity, plot with SD shading ---
    for ch in range(12):  # first 12 channels
        curr_data_ch = curr_data.iloc[:, ch].to_numpy()

        beats_ch = []
        for k in range(len(peaks) - 1):
            start = peaks[k]
            end = peaks[k + 1]

            segment = -1 * curr_data_ch[start:end]
            if len(segment) < 5:
                continue

            old_idx = np.arange(len(segment))
            new_idx = np.linspace(0, len(segment) - 1, n_points)
            resampled = np.interp(new_idx, old_idx, segment)
            beats_ch.append(resampled)

        if len(beats_ch) == 0:
            print(f"No valid beats for channel {ch}, skipping.")
            continue

        beats_ch = np.vstack(beats_ch)

        mean_beat = beats_ch.mean(axis=0)
        std_beat = beats_ch.std(axis=0)

        color, corr = similarity_color(ref_beat, mean_beat)

        # Individual plot for this channel
        plt.figure(figsize=(8, 6))
        # mean trace
        plt.plot(t_norm, mean_beat, color=color, linewidth=6,
                 label=f"Ch {ch} mean (r={corr:.2f})")
        # SD shading with same color, lower alpha
        plt.fill_between(
            t_norm,
            mean_beat - std_beat,
            mean_beat + std_beat,
            color=color,
            alpha=0.3,
            label="±1 SD"
        )
        # Optional: overlay reference beat
        # plt.plot(t_norm, ref_beat, color="k", lw=2, label="ref_beat")

        plt.xlabel("Normalized beat time")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.axis('off')
        plt.show()
