#!/usr/bin/env python3
"""
BLE PPG GUI — Waveforms (12×3), Live R plot (selected channels), Live R heatmap (all 12 locations)
MODIFIED: Added "Enable Real-time Plots" toggle.
- When enabled: behaves like original script.
- When disabled: stops plotting/math to save CPU, but KEEPS recording CSV.
"""

import asyncio
import threading
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, List, Tuple, Union, Dict
from collections import deque
import queue

import numpy as np

# ---- TK / Matplotlib glue ----
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ModuleNotFoundError:
    print(
        "Tkinter isn't available for this Python.\n"
        "Install Python from python.org (includes Tk), or create a Conda env: `conda create -n ble python=3.12 tk`.\n"
    )
    sys.exit(1)

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---- BLE deps ----
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice

# ================== Mapping / decode ==================
MASK_19BIT = 0x7FFFF  # keep lower 19 bits

DATA_COLUMNS = ["660 1", "660 2", "660 3", "660 4", "660 5", "660 6", "660 7", "660 8", "660 9", "660 10", "660 11", "660 12",
                "940 1", "940 2", "940 3", "940 4", "940 5", "940 6", "940 7", "940 8", "940 9", "940 10", "940 11", "940 12",
                "850 1", "850 2", "850 3", "850 4", "850 5", "850 6", "850 7", "850 8", "850 9", "850 10", "850 11", "850 12",
                "pkt_counter"]

RAW_DATA_ORDER = [
    12, 24, 1, 0, 13, 25, 14, 26, 3, 2, 15, 27,
    16, 28, 5, 4, 17, 29, 18, 30, 7, 6, 19, 31,
    20, 32, 9, 8, 21, 33, 22, 34, 11, 10, 23, 35,
    12, 24, 1, 0, 13, 25, 14, 26, 3, 2, 15, 27,
    16, 28, 5, 4, 17, 29, 18, 30, 7, 6, 19, 31,
    20, 32, 9, 8, 21, 33, 22, 34, 11, 10, 23, 35,
    36
]
PKT_COL_INDEX = DATA_COLUMNS.index("pkt_counter")
ORDER_S0 = RAW_DATA_ORDER[:36]
ORDER_S1 = RAW_DATA_ORDER[36:72]

# --- helpers aligned with your CLI connect logic ---
def _has_method(obj, name: str) -> bool:
    return callable(getattr(obj, name, None))

async def get_services(client: BleakClient):
    if _has_method(client, "get_services"):
        return await client.get_services()
    return client.services  # type: ignore[attr-defined]

def select_notify_char(services, preferred_uuid: Optional[str]) -> Optional[str]:
    if preferred_uuid:
        return preferred_uuid.lower()
    for svc in services:
        for ch in svc.characteristics:
            if "notify" in ch.properties:
                return str(ch.uuid).lower()
    return None

# -------------------- utilities --------------------
def now_iso_local_ms() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def filename_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def triplet_to_u24(b0: int, b1: int, b2: int, little_endian: bool = False) -> int:
    if little_endian:
        return (b2 << 16) | (b1 << 8) | b0  # b0 = LSB
    return (b0 << 16) | (b1 << 8) | b2      # b0 = MSB

def split_triplets_with_counter(data: bytes):
    if not data:
        return [], None, b""
    counter = data[-1]
    payload = data[:-1]
    n_full = (len(payload) // 3) * 3
    triplets = [tuple(payload[i:i+3]) for i in range(0, n_full, 3)]
    leftover = payload[n_full:]
    return triplets, counter, leftover

def decode_ppg_triplets(triplets: List[Tuple[int,int,int]], little_endian: bool = False) -> List[int]:
    vals = []
    for a,b,c in triplets:
        u24 = triplet_to_u24(a,b,c, little_endian=little_endian)
        vals.append(u24 & MASK_19BIT)
    return vals

def map_values_to_rows(values72: List[int], pkt_counter: int):
    if len(values72) != 72:
        raise ValueError(f"Expected 72 triplets -> 72 values; got {len(values72)}")
    row0 = [0] * len(DATA_COLUMNS)
    row1 = [0] * len(DATA_COLUMNS)
    for i, v in enumerate(values72[:36]):
        row0[ORDER_S0[i]] = v
    row0[PKT_COL_INDEX] = pkt_counter
    for i, v in enumerate(values72[36:72]):
        row1[ORDER_S1[i]] = v
    row1[PKT_COL_INDEX] = pkt_counter
    return row0, row1

def safe_display_name(d: BLEDevice) -> str:
    if getattr(d, "name", None):
        return d.name
    md = getattr(d, "metadata", {}) or {}
    if isinstance(md, dict):
        ln = md.get("local_name")
        if ln:
            return ln
    det = getattr(d, "details", None)
    if det is not None:
        for key in ("localName", "LocalName", "name"):
            try:
                val = getattr(det, key, None)
            except Exception:
                val = None
            if val:
                return str(val)
        try:
            if isinstance(det, dict):
                for key in ("localName", "LocalName", "name"):
                    val = det.get(key)
                    if val:
                        return str(val)
        except Exception:
            pass
    return ""

# ================== BLE background thread ==================
class BLEWorker(threading.Thread):
    def __init__(self, notify_queue: "queue.Queue", status_cb, little_endian: bool = False):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self.notify_queue = notify_queue
        self.status_cb = status_cb
        self.client: Optional[BleakClient] = None
        self.connected = False
        self.little_endian = little_endian
        self._stop = threading.Event()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    async def _run(self):
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        finally:
            try:
                if self.client and self.client.is_connected:
                    await self.client.disconnect()
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass

    def scan(self, timeout=8.0):
        return asyncio.run_coroutine_threadsafe(self._scan(timeout), self.loop)

    def connect(self, device_or_address: Union[BLEDevice, str], char_uuid: Optional[str]):
        return asyncio.run_coroutine_threadsafe(self._connect(device_or_address, char_uuid), self.loop)

    def disconnect(self):
        return asyncio.run_coroutine_threadsafe(self._disconnect(), self.loop)

    async def _scan(self, timeout: float):
        self.status_cb("Scanning…")
        try:
            devices = await BleakScanner.discover(timeout=timeout)
            self.status_cb(f"Scan complete: {len(devices)} device(s).")
            return devices
        except Exception as e:
            self.status_cb(f"Scan error: {e}")
            return []

    async def _disconnect(self):
        if self.client and self.client.is_connected:
            try:
                await self.client.disconnect()
                self.status_cb("Disconnected.")
            except Exception as e:
                self.status_cb(f"Disconnect error: {e}")
        self.client = None
        self.connected = False

    async def _connect(self, address: str, char_uuid: Optional[str]):
        await self._disconnect()
        self.status_cb(f"Connecting to {address} …")

        self.client = BleakClient(address, disconnected_callback=lambda _: self.status_cb("…Disconnected."))
        try:
            await self.client.connect()
            if not self.client.is_connected:
                raise RuntimeError("Connect failed.")

            self.connected = True
            self.status_cb("Connected. Discovering services …")

            services = await get_services(self.client)
            chosen = select_notify_char(services, char_uuid)
            if not chosen:
                msg = ["No 'notify' characteristic found. Characteristics:"]
                for svc in services:
                    for ch in svc.characteristics:
                        msg.append(f" - {ch.uuid} props={ch.properties}")
                self.status_cb("\n".join(msg))
                raise RuntimeError("Could not find a characteristic with 'notify'.")

            await self.client.start_notify(chosen, self._on_notify)
            self.status_cb(f"Subscribed to {chosen}. Streaming…")
            return True

        except Exception as e:
            self.status_cb(f"Connect error: {e}")
            try:
                if self.client:
                    await self.client.disconnect()
            except Exception:
                pass
            self.client = None
            self.connected = False
            return False

    def _on_notify(self, _: int, data: bytearray):
        try:
            t_iso = now_iso_local_ms()
            triplets, counter, _ = split_triplets_with_counter(bytes(data))
            if len(triplets) != 72:
                return
            values = decode_ppg_triplets(triplets, little_endian=self.little_endian)
            pkt = int(counter) if counter is not None else -1
            row0, row1 = map_values_to_rows(values, pkt)
            self.notify_queue.put({"t_iso": t_iso, "row0": row0, "row1": row1})
        except Exception as e:
            self.status_cb(f"Notify parse error: {e}")

# ================== CSV logger ==================
class CSVLogger:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.data_fp = None
        self.events_fp = None

    def start_data(self):
        if self.data_fp: return
        path = self.outdir / f"{filename_stamp()}.csv"
        self.data_fp = open(path, "w", buffering=65536)
        header = ",".join(DATA_COLUMNS + ["timestamp_iso"])
        self.data_fp.write(header + "\n")

    def stop_data(self):
        if self.data_fp:
            try:
                self.data_fp.flush()
                self.data_fp.close()
            except Exception:
                pass
        self.data_fp = None

    def write_rows(self, row0: List[int], row1: List[int], t_iso: str):
        if not self.data_fp:
            return
        self.data_fp.write(",".join(map(str, row0 + [t_iso])) + "\n")
        self.data_fp.write(",".join(map(str, row1 + [t_iso])) + "\n")

    def log_event(self, label: str):
        if not self.events_fp:
            path = self.outdir / f"{filename_stamp()}_events.csv"
            self.events_fp = open(path, "w", buffering=1)
            self.events_fp.write("timestamp_iso,event\n")
        t_iso = now_iso_local_ms()
        safe = label.replace("\n", " ").replace(",", " ")
        self.events_fp.write(f"{t_iso},{safe}\n")

    def close(self):
        self.stop_data()
        if self.events_fp:
            try: self.events_fp.close()
            except Exception: pass
        self.events_fp = None

# ================== Real-time 12×3 grid (fast) ==================
class FastRTPlotGrid:
    def __init__(self, parent, fs: float = 25.0, window_sec: float = 10.0, autoscale_every: int = 10):
        self.parent = parent
        self.fs = fs
        self.window_samples = max(50, int(round(fs * window_sec)))
        self.autoscale_every = max(1, autoscale_every)
        self._frame = 0

        self.W = self.window_samples
        self.buf = np.full((36, self.W), np.nan, dtype=np.float32)
        self.write_pos = 0
        self.samples_seen = 0
        self.x_full = np.arange(self.W, dtype=np.float32) / float(self.fs)

        self.container = ttk.Frame(parent); self.container.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.container, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.fig = Figure(figsize=(11.5, 8.0), dpi=100)
        self.axes = self.fig.subplots(12, 3, sharex=True)

        col_titles = ["660 nm", "940 nm", "850 nm"]
        for c in range(3):
            self.axes[0][c].set_title(col_titles[c], fontsize=9, pad=2)
        for r in range(12):
            for c in range(3):
                ax = self.axes[r][c]
                ax.tick_params(labelsize=7, pad=1)
                if r < 11: ax.set_xticklabels([])
                if c > 0:  ax.set_yticklabels([])
                ax.set_xlim(0, self.x_full[-1])
        for r in range(12):
            self.axes[r][0].set_ylabel(f"ch {r+1}", fontsize=7, rotation=0, labelpad=16, va="center")
        for c in range(3):
            self.axes[11][c].set_xlabel("Time (s)", fontsize=8)

        self.lines = [None] * 36
        for idx in range(36):
            r = idx % 12; c = idx // 12
            (line,) = self.axes[r][c].plot([], [], lw=0.8, antialiased=False)
            self.lines[idx] = line

        self.fig.tight_layout(pad=0.5)
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self.inner)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def clear(self):
        self.buf[:] = np.nan
        self.write_pos = 0
        self.samples_seen = 0

    def push_rows(self, row0: list, row1: list):
        a0 = np.asarray(row0[:36], dtype=np.float32)
        a1 = np.asarray(row1[:36], dtype=np.float32)
        w0 = self.write_pos % self.W
        w1 = (self.write_pos + 1) % self.W
        self.buf[:, w0] = a0
        self.buf[:, w1] = a1
        self.write_pos = (self.write_pos + 2) % self.W
        self.samples_seen += 2

    def _current_indices(self, n: int):
        n = int(n)
        if n <= 0: return np.array([], dtype=int)
        n = min(n, self.W, self.samples_seen)
        if n == 0: return np.array([], dtype=int)
        start = (self.write_pos - n) % self.W
        if start + n <= self.W:
            return np.arange(start, start + n)
        first = np.arange(start, self.W)
        second = np.arange(0, (start + n) % self.W)
        return np.concatenate([first, second])

    def last_samples(self, ch: int, n: int) -> np.ndarray:
        idx = self._current_indices(n)
        if idx.size == 0:
            return np.array([], dtype=np.float32)
        return self.buf[ch, idx].astype(np.float32, copy=False)

    def redraw(self):
        n = min(self.samples_seen, self.W)
        if n <= 1:
            self.canvas_mpl.draw_idle()
            return
        idx = self._current_indices(n)
        x = self.x_full[-n:]  # tail

        do_scale = (self._frame % self.autoscale_every == 0)
        for ch in range(36):
            y = self.buf[ch, idx]
            self.lines[ch].set_data(x, y)
            if do_scale and np.isfinite(y).any():
                ax = self.axes[ch % 12][ch // 12]
                ymin = float(np.nanmin(y)); ymax = float(np.nanmax(y))
                if ymax == ymin: ymax = ymin + 1.0
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

        for c in range(3):
            self.axes[11][c].set_xlim(self.x_full[-n], self.x_full[-1])

        self.canvas_mpl.draw_idle()
        self._frame += 1

    def resize_window(self, fs: float, window_sec: float):
        self.fs = fs
        W_new = max(50, int(round(fs * window_sec)))
        if W_new == self.W:
            return
        new_buf = np.full((36, W_new), np.nan, dtype=np.float32)
        n = min(self.samples_seen, min(self.W, W_new))
        if n > 0:
            idx = self._current_indices(n)
            new_buf[:, W_new - n:] = self.buf[:, idx]
        self.buf = new_buf
        self.W = W_new
        self.window_samples = W_new
        self.x_full = np.arange(self.W, dtype=np.float32) / float(self.fs)
        for r in range(12):
            for c in range(3):
                self.axes[r][c].set_xlim(0, self.x_full[-1])

# =============== Robust helpers for R ==================
def _robust_mad(y: np.ndarray) -> float:
    if y.size == 0 or not np.isfinite(y).any():
        return 0.0
    med = np.nanmedian(y)
    return 1.4826 * float(np.nanmedian(np.abs(y - med)))

def _robust_ac(y: np.ndarray) -> float:
    if y.size == 0 or not np.isfinite(y).any():
        return 0.0
    p5, p95 = np.nanpercentile(y, [5, 95])
    return 0.5 * float(p95 - p5)

# =============== Live R (selected channels) panel ==================
class RPlotPanel:
    def __init__(self, parent, fs: float = 25.0, r_pts_window: int = 50):
        self.parent = parent
        self.fs = fs
        self.dt_per_R = 2.0 / max(1e-9, fs)
        self.r_pts_window = int(r_pts_window)
        self.selected: List[int] = []
        self.buffers: Dict[int, deque] = {}
        self.lines: Dict[int, any] = {}

        self.fig = Figure(figsize=(11.5, 2.6), dpi=100)
        self.ax = self.fig.subplots(1, 1)
        self.ax.set_title("R (combined 660 & 850 vs 940)", fontsize=10, pad=2)
        self.ax.set_xlabel("Time (s)", fontsize=9)
        self.ax.set_ylabel("R", fontsize=9)
        self.ax.grid(True, alpha=0.25, linewidth=0.8)

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(fill="x", expand=False)

    def set_fs(self, fs: float):
        self.fs = fs
        self.dt_per_R = 2.0 / max(1e-9, fs)

    def set_selection(self, indices_0based: List[int]):
        new_set = set(indices_0based)
        old_set = set(self.selected)
        to_remove = old_set - new_set
        for idx in to_remove:
            line = self.lines.pop(idx, None)
            if line is not None:
                line.remove()
            self.buffers.pop(idx, None)

        for idx in indices_0based:
            if idx not in self.lines:
                (line,) = self.ax.plot([], [], lw=2.0, label=f"loc {idx+1}")
                self.lines[idx] = line
            if idx not in self.buffers:
                self.buffers[idx] = deque(maxlen=self.r_pts_window)

        self.selected = list(indices_0based)
        if self.selected:
            leg = self.ax.legend(title="Locations", frameon=False, ncol=min(3, len(self.selected)))
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontsize(8)
        else:
            leg = self.ax.get_legend()
            if leg: leg.remove()
        self.canvas_mpl.draw_idle()

    def append_values(self, r_values: Dict[int, float]):
        for idx, val in r_values.items():
            if idx not in self.buffers:
                self.buffers[idx] = deque(maxlen=self.r_pts_window)
            self.buffers[idx].append(float(val))

    def redraw(self):
        ymins, ymaxs = [], []
        for idx in self.selected:
            buf = self.buffers.get(idx, None)
            if not buf:
                self.lines[idx].set_data([], [])
                continue
            y = np.asarray(buf, dtype=float)
            x = np.arange(len(y), dtype=float) * self.dt_per_R
            self.lines[idx].set_data(x, y)
            if y.size > 0 and np.isfinite(y).any():
                ymins.append(np.nanmin(y))
                ymaxs.append(np.nanmax(y))

        max_len = 0
        for idx in self.selected:
            b = self.buffers.get(idx, None)
            if b: max_len = max(max_len, len(b))
        xmax = max(1, max_len) * self.dt_per_R
        self.ax.set_xlim(max(0, xmax - self.r_pts_window * self.dt_per_R), xmax)

        if ymins and ymaxs:
            ymin, ymax = float(np.nanmin(ymins)), float(np.nanmax(ymaxs))
            if ymax == ymin:
                ymax = ymin + 1e-3
            pad = 0.05 * (ymax - ymin)
            self.ax.set_ylim(ymin - pad, ymax + pad)

        self.canvas_mpl.draw_idle()

# =============== R heatmap ==================
class RHeatmapPanel:
    def __init__(self, parent, history_rows: int = 60, cmap: str = "viridis"):
        self.parent = parent
        self.H = int(history_rows)
        self.W = 12
        self.cmap = cmap
        self.buf = np.full((self.H, self.W), np.nan, dtype=np.float32)
        self.write_pos = 0
        self.rows_seen = 0
        self._frame = 0

        self.fig = Figure(figsize=(11.5, 2.8), dpi=100)
        self.ax = self.fig.subplots(1, 1)
        self.ax.set_title("R Heatmap (time × location)", fontsize=10, pad=2)
        self.ax.set_xlabel("Location (1 → 12)", fontsize=9)
        self.ax.set_ylabel("Time", fontsize=9)

        self.im = self.ax.imshow(self.buf, aspect="auto", interpolation="nearest",
                                 cmap=self.cmap, vmin=0.0, vmax=1.0)
        self.ax.set_yticks([])
        self.ax.set_xticks(np.arange(0, self.W, 1))
        self.ax.set_xticklabels([str(i+1) for i in range(self.W)], fontsize=8)

        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.cbar.set_label("R", fontsize=9)

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(fill="x", expand=False)

    def clear(self):
        self.buf[:] = np.nan
        self.write_pos = 0
        self.rows_seen = 0

    def append_row(self, r12: List[float]):
        if len(r12) != self.W:
            return
        self.buf[self.write_pos, :] = np.asarray(r12, dtype=np.float32)
        self.write_pos = (self.write_pos + 1) % self.H
        self.rows_seen += 1

    def _current_buffer(self) -> np.ndarray:
        n = min(self.rows_seen, self.H)
        if n == 0:
            return self.buf[:0, :]
        start = (self.write_pos - n) % self.H
        if start + n <= self.H:
            return self.buf[start:start+n, :]
        first = self.buf[start:self.H, :]
        second = self.buf[0:(start+n) % self.H, :]
        return np.vstack([first, second])

    def redraw(self):
        Z = self._current_buffer()
        if Z.size == 0:
            self.canvas_mpl.draw_idle()
            return

        if self._frame % 8 == 0:
            finite = Z[np.isfinite(Z)]
            if finite.size >= 4:
                vmin = float(np.nanpercentile(finite, 10))
                vmax = float(np.nanpercentile(finite, 90))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                self.im.set_clim(vmin=vmin, vmax=vmax)
                self.cbar.update_normal(self.im)

        self.im.set_data(Z)
        self.canvas_mpl.draw_idle()
        self._frame += 1

# ================== Main GUI ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BLE PPG GUI — Waveforms + R + Recorder")

        self._autoset_scaling()
        self.after(0, self._maximize_window)
        self.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.bind("<Escape>", lambda e: self._exit_fullscreen())

        # --- State ---
        self.fs = tk.DoubleVar(value=25.0)
        self.window_sec = tk.DoubleVar(value=10.0)
        self.little_endian = tk.BooleanVar(value=False)
        self.enable_plots = tk.BooleanVar(value=True)  # <--- NEW TOGGLE
        self.outdir = tk.StringVar(value=str(Path("./logs").resolve()))
        self.recording = False

        self.notify_queue: "queue.Queue" = queue.Queue()
        self.ble = BLEWorker(self.notify_queue, self.set_status, little_endian=self.little_endian.get())
        self.ble.start()

        self.logger = CSVLogger(Path(self.outdir.get()))
        self.named_devices: List[BLEDevice] = []

        # --- left controls ---
        left = ttk.Frame(self); left.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Label(left, text="BLE Devices (named)").pack(anchor="w")
        dev_frame = ttk.Frame(left); dev_frame.pack(fill="both", expand=False)
        self.listbox = tk.Listbox(dev_frame, height=10)
        sb = ttk.Scrollbar(dev_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        scan_row = ttk.Frame(left); scan_row.pack(fill="x", pady=4)
        ttk.Button(scan_row, text="Scan 8s", command=self.on_scan).pack(side="left")
        ttk.Button(scan_row, text="Connect", command=self.on_connect).pack(side="left", padx=4)
        self.char_uuid = tk.StringVar(value="")
        char_row = ttk.Frame(left); char_row.pack(fill="x", pady=2)
        ttk.Label(char_row, text="Notify Char (optional):").pack(anchor="w")
        ttk.Entry(char_row, textvariable=self.char_uuid, width=36).pack(fill="x")

        # Settings
        ttk.Label(left, text="Settings").pack(anchor="w", pady=(10,2))
        
        # --- NEW CHECKBOX ---
        ttk.Checkbutton(left, text="Enable Real-time Plots", variable=self.enable_plots).pack(anchor="w", pady=2)
        
        row = ttk.Frame(left); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Sampling rate (Hz):").pack(side="left")
        ttk.Entry(row, textvariable=self.fs, width=6).pack(side="left", padx=4)

        row2 = ttk.Frame(left); row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Plot window (s):").pack(side="left")
        ttk.Entry(row2, textvariable=self.window_sec, width=6).pack(side="left", padx=4)

        ttk.Checkbutton(left, text="Little-endian triplets", variable=self.little_endian, command=self.on_toggle_endian).pack(anchor="w", pady=2)

        # Recording
        rec_row = ttk.Frame(left); rec_row.pack(fill="x", pady=(8,2))
        ttk.Label(rec_row, text="Output folder:").pack(side="left")
        ttk.Entry(left, textvariable=self.outdir, width=30).pack(fill="x")
        self.btn_record = ttk.Button(left, text="● Start Recording", command=self.on_toggle_record, style="Record.TButton")
        self.btn_record.pack(fill="x", pady=6)

        # Events
        ttk.Label(left, text="Event label:").pack(anchor="w", pady=(8,2))
        ev_row = ttk.Frame(left); ev_row.pack(fill="x")
        self.event_var = tk.StringVar()
        ttk.Entry(ev_row, textvariable=self.event_var).pack(side="left", fill="x", expand=True)
        ttk.Button(ev_row, text="Log Event", command=self.on_log_event).pack(side="left", padx=4)

        # R selection controls
        rframe = ttk.LabelFrame(left, text="R (ratio-of-ratios) — window=100 samples")
        rframe.pack(fill="x", pady=(12, 4))
        self.r_vars = []
        self._r_checkbuttons: List[ttk.Checkbutton] = []
        grid = ttk.Frame(rframe); grid.pack(fill="x")
        for i in range(12):
            var = tk.IntVar(value=0)
            self.r_vars.append(var)
            cb = ttk.Checkbutton(grid, text=str(i+1), variable=var, command=self.on_r_selection_changed)
            cb.grid(row=i//6, column=i%6, sticky="w", padx=2, pady=1)
            self._r_checkbuttons.append(cb)

        method_row = ttk.Frame(rframe); method_row.pack(fill="x", pady=(4, 0))
        ttk.Label(method_row, text="Combine:").pack(side="left")
        self.r_method = tk.StringVar(value="weighted")
        ttk.Combobox(method_row, state="readonly", width=12, textvariable=self.r_method,
                     values=("weighted", "median", "best")).pack(side="left", padx=4)

        self._set_r_controls_state(False)

        # Status
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(left, textvariable=self.status_var, foreground="#444").pack(anchor="w", pady=(10,0))

        # --- right plots (stacked) ---
        right = ttk.Frame(self); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self.plot = FastRTPlotGrid(right, fs=self.fs.get(), window_sec=self.window_sec.get(), autoscale_every=10)
        self.rplot = RPlotPanel(right, fs=self.fs.get(), r_pts_window=50)
        self.rheat = RHeatmapPanel(right, history_rows=60, cmap="viridis")

        self.after(50, self.poll_notify_queue)
        self.after(120, self.redraw_plot)

        s = ttk.Style(self); s.configure("Record.TButton", foreground="#b00")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _autoset_scaling(self):
        try:
            ppi = self.winfo_fpixels('1i')
            scaling = float(ppi) / 72.0
            if 0.6 <= scaling <= 3.0:
                self.tk.call('tk', 'scaling', scaling)
        except Exception:
            pass

    def _maximize_window(self):
        try: self.state('zoomed'); return
        except Exception: pass
        try: self.attributes('-zoomed', True); return
        except Exception: pass
        w = self.winfo_screenwidth(); h = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+0+0")

    def _toggle_fullscreen(self):
        fs = bool(self.attributes('-fullscreen'))
        self.attributes('-fullscreen', not fs)

    def _exit_fullscreen(self):
        self.attributes('-fullscreen', False)

    def _set_r_controls_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for cb in getattr(self, "_r_checkbuttons", []):
            cb.configure(state=state)

    def _r_selected_indices(self) -> List[int]:
        return [i for i, var in enumerate(self.r_vars) if var.get() == 1]

    def on_r_selection_changed(self):
        self.rplot.set_selection(self._r_selected_indices())

    def set_status(self, msg: str):
        self.after(0, lambda: self.status_var.set(msg))

    def on_scan(self):
        self.listbox.delete(0, "end")
        self.named_devices.clear()
        fut = self.ble.scan(timeout=8.0)

        def _done(_f):
            devices = _f.result()
            named = []
            for d in devices:
                nm = safe_display_name(d).strip()
                if nm:
                    named.append((nm, d))
            def fill_list():
                if not named:
                    self.set_status("No named devices found.")
                    return
                for nm, dev in sorted(named, key=lambda x: x[0].lower()):
                    self.listbox.insert("end", f"{nm} [{dev.address}]")
                    self.named_devices.append(dev)
                self.set_status(f"Scan done. {len(named)} named device(s). Select and Connect.")
            self.after(0, fill_list)
        fut.add_done_callback(_done)

    def selected_device(self) -> Optional[BLEDevice]:
        sel = self.listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        if 0 <= idx < len(self.named_devices):
            return self.named_devices[idx]
        return None

    def on_connect(self):
        dev = self.selected_device()
        if not dev:
            messagebox.showinfo("Select device", "Please select a device in the list.")
            return
        char = self.char_uuid.get().strip() or None
        self.ble.little_endian = self.little_endian.get()

        fut = self.ble.connect(dev.address, char)

        def _done(_f):
            ok = _f.result()
            def after_connect():
                if ok:
                    self.plot.clear()
                    self.rplot.set_selection([])
                    self.rheat.clear()
                    self._set_r_controls_state(True)
                    self.set_status("Connected & streaming.")
                else:
                    self._set_r_controls_state(False)
                    self.set_status("Connect failed. Check permissions/char UUID and try again.")
            self.after(0, after_connect)
        fut.add_done_callback(_done)

    def on_toggle_endian(self):
        self.ble.little_endian = self.little_endian.get()
        self.set_status(f"Little-endian set to {self.little_endian.get()}")

    def on_toggle_record(self):
        self.recording = not self.recording
        outdir = Path(self.outdir.get()); outdir.mkdir(parents=True, exist_ok=True)
        self.logger.outdir = outdir
        if self.recording:
            self.logger.start_data()
            self.btn_record.configure(text="■ Stop Recording")
            self.set_status("Recording to CSV.")
        else:
            self.logger.stop_data()
            self.btn_record.configure(text="● Start Recording")
            self.set_status("Recording stopped.")

    def on_log_event(self):
        label = (self.event_var.get() or "event").strip()
        self.logger.log_event(label)
        self.event_var.set("")
        self.set_status(f"Logged event: {label}")

    def _compute_R_all12(self) -> List[float]:
        method = (self.r_method.get() or "weighted").lower()
        win_n = 100
        eps = 1e-12
        R = [np.nan] * 12

        for idx in range(12):
            y660 = self.plot.last_samples(idx, win_n)
            y940 = self.plot.last_samples(12 + idx, win_n)
            y850 = self.plot.last_samples(24 + idx, win_n)
            if y660.size < 10 or y940.size < 10 or y850.size < 10:
                continue

            ac660, ac940, ac850 = _robust_ac(y660), _robust_ac(y940), _robust_ac(y850)
            dc660 = float(np.nanmean(y660)) if np.isfinite(y660).any() else 0.0
            dc940 = float(np.nanmean(y940)) if np.isfinite(y940).any() else 0.0
            dc850 = float(np.nanmean(y850)) if np.isfinite(y850).any() else 0.0

            if abs(dc660) < eps: dc660 = eps * np.sign(dc660 or 1.0)
            if abs(dc940) < eps: dc940 = eps * np.sign(dc940 or 1.0)
            if abs(dc850) < eps: dc850 = eps * np.sign(dc850 or 1.0)

            s660 = ac660 / dc660
            s940 = ac940 / dc940
            s850 = ac850 / dc850

            r660 = s660 / s940 if abs(s940) > eps else np.nan
            r850 = s850 / s940 if abs(s940) > eps else np.nan

            if method == "median":
                R[idx] = float(np.nanmedian([r660, r850]))
            elif method == "best":
                snr660 = ac660 / max(eps, _robust_mad(y660))
                snr850 = ac850 / max(eps, _robust_mad(y850))
                R[idx] = float(r660 if snr660 >= snr850 else r850)
            else:  # weighted
                snr660 = ac660 / max(eps, _robust_mad(y660))
                snr850 = ac850 / max(eps, _robust_mad(y850))
                w660 = 0.0 if (not np.isfinite(r660)) else max(0.0, float(snr660))
                w850 = 0.0 if (not np.isfinite(r850)) else max(0.0, float(snr850))
                if (w660 + w850) > 0:
                    R[idx] = float((w660 * r660 + w850 * r850) / (w660 + w850))
                else:
                    R[idx] = float(np.nanmedian([r660, r850]))
        return R

    def _compute_R_for_selected(self) -> Dict[int, float]:
        win_n = 100
        method = (self.r_method.get() or "weighted").lower()
        eps = 1e-12
        r_vals: Dict[int, float] = {}

        for idx in self._r_selected_indices():
            y660 = self.plot.last_samples(idx, win_n)
            y940 = self.plot.last_samples(12 + idx, win_n)
            y850 = self.plot.last_samples(24 + idx, win_n)
            if y660.size < 10 or y940.size < 10 or y850.size < 10:
                continue

            ac660, ac940, ac850 = _robust_ac(y660), _robust_ac(y940), _robust_ac(y850)
            dc660 = float(np.nanmean(y660)) if np.isfinite(y660).any() else 0.0
            dc940 = float(np.nanmean(y940)) if np.isfinite(y940).any() else 0.0
            dc850 = float(np.nanmean(y850)) if np.isfinite(y850).any() else 0.0

            if abs(dc660) < eps: dc660 = eps * np.sign(dc660 or 1.0)
            if abs(dc940) < eps: dc940 = eps * np.sign(dc940 or 1.0)
            if abs(dc850) < eps: dc850 = eps * np.sign(dc850 or 1.0)

            s660 = ac660 / dc660
            s940 = ac940 / dc940
            s850 = ac850 / dc850

            r660 = s660 / s940 if abs(s940) > eps else np.nan
            r850 = s850 / s940 if abs(s940) > eps else np.nan

            if method == "median":
                Rv = np.nanmedian([r660, r850])
            elif method == "best":
                snr660 = ac660 / max(eps, _robust_mad(y660))
                snr850 = ac850 / max(eps, _robust_mad(y850))
                Rv = r660 if snr660 >= snr850 else r850
            else:
                snr660 = ac660 / max(eps, _robust_mad(y660))
                snr850 = ac850 / max(eps, _robust_mad(y850))
                w660 = 0.0 if (not np.isfinite(r660)) else max(0.0, float(snr660))
                w850 = 0.0 if (not np.isfinite(r850)) else max(0.0, float(snr850))
                if (w660 + w850) > 0:
                    Rv = (w660 * r660 + w850 * r850) / (w660 + w850)
                else:
                    Rv = np.nanmedian([r660, r850])

            r_vals[idx] = float(Rv)
        return r_vals

    def poll_notify_queue(self):
        updated = False
        try:
            while True:
                item = self.notify_queue.get_nowait()
                t_iso = item["t_iso"]
                row0 = item["row0"]; row1 = item["row1"]
                
                # --- MODIFIED: ONLY PUSH TO PLOT IF ENABLED ---
                if self.enable_plots.get():
                    self.plot.push_rows(row0, row1)
                    
                # --- ALWAYS LOG IF RECORDING ---
                if self.recording:
                    self.logger.write_rows(row0, row1, t_iso)
                    
                updated = True
        except queue.Empty:
            pass

        # --- MODIFIED: ONLY COMPUTE R/HEATMAP IF PLOTS ENABLED ---
        if updated and self.enable_plots.get():
            sel = self._r_selected_indices()
            if sel:
                r_vals = self._compute_R_for_selected()
                if r_vals:
                    self.rplot.append_values(r_vals)
            r12 = self._compute_R_all12()
            if any(np.isfinite(r12)):
                self.rheat.append_row(r12)

        self.after(20, self.poll_notify_queue)

    def redraw_plot(self):
        # --- MODIFIED: SKIP DRAWING IF DISABLED ---
        if not self.enable_plots.get():
            self.after(200, self.redraw_plot)
            return

        fs = float(self.fs.get())
        win = float(self.window_sec.get())
        self.plot.resize_window(fs, win)
        self.plot.redraw()
        self.rplot.set_fs(fs)
        self.rplot.redraw()
        self.rheat.redraw()

        self.after(120, self.redraw_plot)

    def on_close(self):
        try: self.logger.close()
        except Exception: pass
        try: self.ble.disconnect()
        except Exception: pass
        try: self.ble.stop()
        except Exception: pass
        self.destroy()

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass
    App().mainloop()