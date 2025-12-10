#!/usr/bin/env python3
"""
simple_ble.py — BLE PPG → CSV + optional real-time plotting (12×3 grid).

- Decodes 72 triplets + 1B counter per packet → two 36-ch samples (masked to 19 bits).
- CSV: 2 rows/packet with trailing timestamp_iso (system time at receipt).
- Plot (optional): 12×3 grid where col0=660, col1=940, col2=850; rows=channels 1..12.
"""

import argparse
import asyncio
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, List, Tuple
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice

MASK_19BIT = 0x7FFFF  # keep lower 19 bits

# ---------------------------- COLUMN DEFINITIONS ----------------------------
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
# ---------------------------------------------------------------------------

PKT_COL_INDEX = DATA_COLUMNS.index("pkt_counter")

# Sanity checks (raise early if config is off)
if len(DATA_COLUMNS) != 37:
    raise RuntimeError("DATA_COLUMNS must have 37 entries (36 channels + pkt_counter).")
if len(RAW_DATA_ORDER) != 73:
    raise RuntimeError("RAW_DATA_ORDER must have 73 entries (36 + 36 + pkt_index).")
if RAW_DATA_ORDER[-1] != PKT_COL_INDEX:
    raise RuntimeError("The last element of RAW_DATA_ORDER must equal index of 'pkt_counter'.")

ORDER_S0 = RAW_DATA_ORDER[:36]   # mapping for sample 0 (first 36 triplets)
ORDER_S1 = RAW_DATA_ORDER[36:72] # mapping for sample 1 (next 36 triplets)

# -------------------------- parsing --------------------------

def triplet_to_u24(b0: int, b1: int, b2: int, little_endian: bool = False) -> int:
    """Combine 3 bytes into a 24-bit unsigned int."""
    if little_endian:
        return (b2 << 16) | (b1 << 8) | b0  # b0 = LSB
    return (b0 << 16) | (b1 << 8) | b2      # b0 = MSB

def split_triplets_with_counter(data: bytes) -> Tuple[List[Tuple[int, int, int]], Optional[int], bytes]:
    """Split into 3B groups and trailing 1B counter."""
    if not data:
        return [], None, b""
    counter = data[-1]
    payload = data[:-1]
    n_full = (len(payload) // 3) * 3
    triplets = [tuple(payload[i:i+3]) for i in range(0, n_full, 3)]
    leftover = payload[n_full:]
    return triplets, counter, leftover

def decode_ppg_triplets(triplets: List[Tuple[int, int, int]], little_endian: bool = False) -> List[int]:
    """Return masked 19-bit values from triplets."""
    values = []
    for a, b, c in triplets:
        u24 = triplet_to_u24(a, b, c, little_endian=little_endian)
        values.append(u24 & MASK_19BIT)
    return values

def map_values_to_rows(values72: List[int], pkt_counter: int) -> Tuple[List[int], List[int]]:
    """Map 72 values → two rows matching DATA_COLUMNS; both rows include pkt_counter."""
    if len(values72) != 72:
        raise ValueError(f"Expected 72 values, got {len(values72)}")

    row0 = [0] * len(DATA_COLUMNS)
    row1 = [0] * len(DATA_COLUMNS)

    for i, v in enumerate(values72[:36]):
        row0[ORDER_S0[i]] = v
    row0[PKT_COL_INDEX] = pkt_counter

    for i, v in enumerate(values72[36:72]):
        row1[ORDER_S1[i]] = v
    row1[PKT_COL_INDEX] = pkt_counter

    return row0, row1

# -------------------------- BLE helpers --------------------------

def _on_disconnect(_: BleakClient):
    print("…Disconnected.", flush=True)

def _safe_display_name(d: BLEDevice) -> str:
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

async def find_device_by_name(name: str, timeout: float = 10.0) -> Optional[BLEDevice]:
    target = (name or "").lower().strip()
    print(f"Scanning up to {timeout:.0f}s for “{name}” …")
    devices = await BleakScanner.discover(timeout=timeout)

    for d in devices:
        disp = _safe_display_name(d)
        if disp.lower() == target:
            rssi = getattr(d, "rssi", None)
            rssi_str = f" RSSI={rssi}" if rssi is not None else ""
            print(f"  ✓ Found {disp or '<no name>'} [{d.address}]{rssi_str}")
            return d

    if not devices:
        print("  (No BLE devices found.)")
    else:
        print("  (No exact name match. Nearby devices:)")
        for d in devices[:20]:
            disp = _safe_display_name(d) or "<no name>"
            rssi = getattr(d, "rssi", None)
            rssi_str = f" RSSI={rssi}" if rssi is not None else ""
            print(f"   - {disp} [{d.address}]{rssi_str}")
    return None

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

# -------------------------- CSV helpers --------------------------

def now_filename_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def now_iso_local_ms() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

def make_csv_writer(csv_path: Path, write_header: bool = True):
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(DATA_COLUMNS + ["timestamp_iso"])
        csv_file.flush()

    def write_rows(row0, row1, t_iso: str):
        writer.writerow([*row0, t_iso])
        writer.writerow([*row1, t_iso])
        csv_file.flush()

    def close():
        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass

    return write_rows, close

# -------------------------- Real-time plot manager --------------------------

class PlotManager:
    """
    12×3 grid:
      col0: 660 (indices 0..11)
      col1: 940 (indices 12..23)
      col2: 850 (indices 24..35)
    Maintains a deque per channel (maxlen = window_samples). Two samples per packet are appended.
    """
    def __init__(self, fs: float, window_sec: float, titles_36: List[str],
                 autoscale: bool = True, ylim: Optional[Tuple[float, float]] = None):
        self.fs = fs
        self.window_samples = max(10, int(round(fs * window_sec)))
        self.buffers = [deque(maxlen=self.window_samples) for _ in range(36)]
        self.autoscale = autoscale
        self.user_ylim = ylim
        self._stopped = False

        self.fig, self.axes = plt.subplots(12, 3, figsize=(12, 18), sharex=True)
        self.lines = [None] * 36

        for row in range(12):
            for col in range(3):
                idx = col * 12 + row  # 0..35
                ax = self.axes[row][col]
                (line,) = ax.plot([], [], lw=0.9)
                ax.set_xlim(0, self.window_samples)
                title = titles_36[idx] if idx < len(titles_36) else f"ch{idx}"
                ax.set_title(title, fontsize=9)
                if self.user_ylim:
                    ax.set_ylim(*self.user_ylim)
                self.lines[idx] = line

        self.axes[-1][0].set_xlabel("samples")
        self.axes[-1][1].set_xlabel("samples")
        self.axes[-1][2].set_xlabel("samples")
        plt.tight_layout()
        plt.show(block=False)

    def stop(self):
        self._stopped = True

    def push_sample36(self, sample36: List[int]):
        for i, v in enumerate(sample36[:36]):
            self.buffers[i].append(float(v))

    def push_rows(self, row0: List[int], row1: List[int]):
        # rows are in DATA_COLUMNS order (first 36 entries are channels)
        self.push_sample36(row0)
        self.push_sample36(row1)

    def _update_axes(self):
        for idx in range(36):
            buf = self.buffers[idx]
            y = np.fromiter(buf, dtype=float) if buf else np.array([])
            x = np.arange(len(y))
            self.lines[idx].set_data(x, y)

            ax = self.axes[idx % 12][idx // 12]  # row, col
            if self.user_ylim is None:
                # simple autoscale
                if len(y) >= 2:
                    ymin = float(np.min(y))
                    ymax = float(np.max(y))
                    if ymax == ymin:
                        ymax = ymin + 1.0
                    pad = 0.05 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)
                else:
                    ax.set_ylim(0, 1)

    async def run(self, hz: float = 8.0):
        dt = max(0.02, 1.0 / float(hz))
        try:
            while not self._stopped:
                self._update_axes()
                self.fig.canvas.draw_idle()
                plt.pause(0.001)  # allow GUI event loop to breathe
                await asyncio.sleep(dt)
        except asyncio.CancelledError:
            pass

# -------------------------- streaming --------------------------

def make_packet_handler(write_rows, plotter: Optional[PlotManager], little_endian: bool):
    def handle_packet(_: int, data: bytearray):
        if not data:
            return
        t_iso = now_iso_local_ms()

        triplets, counter, leftover = split_triplets_with_counter(bytes(data))
        if leftover:
            print(f"[warn] leftover {len(leftover)} byte(s) (payload not multiple of 3).", flush=True)
        if len(triplets) != 72:
            print(f"[warn] expected 72 triplets, got {len(triplets)}; skipping.", flush=True)
            return

        values = decode_ppg_triplets(triplets, little_endian=little_endian)
        pkt = int(counter) if counter is not None else -1
        row0, row1 = map_values_to_rows(values, pkt)

        # CSV
        if write_rows:
            write_rows(row0, row1, t_iso)

        # RT plot
        if plotter is not None:
            plotter.push_rows(row0, row1)
    return handle_packet

async def stream(address: str,
                 char_uuid: Optional[str],
                 outdir: Path,
                 little_endian: bool,
                 no_header: bool,
                 plot: bool,
                 fs: float,
                 plot_window_sec: float,
                 plot_hz: float,
                 plot_ylim: Optional[Tuple[float, float]]):
    # Prepare CSV
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{now_filename_stamp()}.csv"
    print(f"Writing CSV to: {csv_path}")
    write_rows, close_csv = make_csv_writer(csv_path, write_header=not no_header)

    # Prepare plotter (optional)
    plotter = None
    plot_task = None
    if plot:
        titles36 = DATA_COLUMNS[:36]
        plotter = PlotManager(fs=fs, window_sec=plot_window_sec, titles_36=titles36,
                              autoscale=(plot_ylim is None), ylim=plot_ylim)
        plot_task = asyncio.create_task(plotter.run(hz=plot_hz))

    print(f"Connecting to {address} …")
    async with BleakClient(address, disconnected_callback=_on_disconnect) as client:
        if not client.is_connected:
            if plotter:
                plotter.stop()
            if plot_task:
                plot_task.cancel()
            close_csv()
            raise RuntimeError("Connect failed.")

        print("Connected. Discovering services …")
        services = await get_services(client)
        chosen = select_notify_char(services, char_uuid)
        if not chosen:
            print("No 'notify' characteristic found. Available characteristics:")
            for svc in services:
                for ch in svc.characteristics:
                    print(f" - {ch.uuid}  props={ch.properties}")
            if plotter:
                plotter.stop()
            if plot_task:
                plot_task.cancel()
            close_csv()
            raise RuntimeError("Could not find a characteristic with 'notify'.")

        print(f"Subscribing to notifications on {chosen} …")
        cb = make_packet_handler(write_rows, plotter, little_endian=little_endian)
        await client.start_notify(chosen, cb)

        print("Streaming. Press Ctrl-C to stop.")
        try:
            while client.is_connected:
                await asyncio.sleep(1.0)
        finally:
            try:
                await client.stop_notify(chosen)
            except Exception:
                pass
            close_csv()
            if plotter:
                plotter.stop()
            if plot_task:
                plot_task.cancel()
            print(f"Closed {csv_path}")

# -------------------------- CLI --------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="BLE PPG → CSV (+ optional real-time plot): 72x(3B)->19-bit; mapped; 2 samples/packet; timestamped.")
    parser.add_argument("--name", required=True, help="Exact advertising/local name (case-insensitive).")
    parser.add_argument("--char", help="Notify characteristic UUID (optional).")
    parser.add_argument("--scan-timeout", type=float, default=10.0, help="Scan timeout (s).")
    parser.add_argument("--little-endian", action="store_true",
                        help="Interpret triplets as little-endian (default big-endian).")
    parser.add_argument("--no-header", action="store_true", help="Do not write CSV header.")
    parser.add_argument("--outdir", default=".", help="Directory to write CSV file (default: current directory).")
    # Real-time plot flags
    parser.add_argument("--plot", action="store_true", help="Enable 12×3 real-time plotting.")
    parser.add_argument("--fs", type=float, default=25.0, help="Sampling rate (Hz) to size the plot window.")
    parser.add_argument("--plot-window-sec", type=float, default=10.0, help="Seconds shown in each subplot window.")
    parser.add_argument("--plot-hz", type=float, default=8.0, help="Redraw rate (Hz).")
    parser.add_argument("--plot-ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
                        help="Fix y-limits for all subplots (e.g., --plot-ylim 0 200000).")
    args = parser.parse_args()

    dev = await find_device_by_name(args.name, timeout=args.scan_timeout)
    if dev is None:
        sys.exit(1)

    try:
        ylim = tuple(args.plot_ylim) if args.plot_ylim is not None else None
        await stream(dev.address,
                     args.char.lower() if args.char else None,
                     Path(args.outdir),
                     args.little_endian,
                     args.no_header,
                     args.plot,
                     args.fs,
                     args.plot_window_sec,
                     args.plot_hz,
                     ylim)
    except KeyboardInterrupt:
        print("\nStopping by user request.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass
    asyncio.run(main())
