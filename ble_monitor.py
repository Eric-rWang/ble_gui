#!/usr/bin/env python3
"""
BLE PPG GUI — Windows Optimized (Popup Fixed)
"""

import asyncio
import threading
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Optional, List, Tuple, Dict
import queue

import numpy as np

# ---- TK / Matplotlib glue ----
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog
except ModuleNotFoundError:
    print("Tkinter isn't available. Install via your package manager.")
    sys.exit(1)

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")

# ---- BLE deps ----
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice

# ================== Constants ==================
MASK_19BIT = 0x7FFFF
DATA_COLUMNS = ["660 1", "660 2", "660 3", "660 4", "660 5", "660 6", "660 7", "660 8", "660 9", "660 10", "660 11", "660 12",
                "940 1", "940 2", "940 3", "940 4", "940 5", "940 6", "940 7", "940 8", "940 9", "940 10", "940 11", "940 12",
                "850 1", "850 2", "850 3", "850 4", "850 5", "850 6", "850 7", "850 8", "850 9", "850 10", "850 11", "850 12",
                "pkt_counter"]

# Pre-compute mapping indices for speed
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

# ================== Logic Helpers ==================
def triplet_to_u24(b0, b1, b2):
    return (b0 << 16) | (b1 << 8) | b2

def process_raw_packet(raw_bytes: bytes) -> Optional[Dict]:
    if not raw_bytes: return None
    counter = raw_bytes[-1]
    payload = raw_bytes[:-1]
    n_payload = len(payload)
    n_triplets = n_payload // 3
    if n_triplets != 72: return None 

    values = []
    for i in range(0, n_payload, 3):
        val = (payload[i] << 16) | (payload[i+1] << 8) | payload[i+2]
        values.append(val & MASK_19BIT)
        
    pkt = int(counter)
    row0 = [0] * len(DATA_COLUMNS)
    row1 = [0] * len(DATA_COLUMNS)
    
    for i in range(36):
        row0[ORDER_S0[i]] = values[i]
    row0[PKT_COL_INDEX] = pkt
    
    for i in range(36):
        row1[ORDER_S1[i]] = values[i+36]
    row1[PKT_COL_INDEX] = pkt
    
    return {"row0": row0, "row1": row1, "pkt": pkt}

def safe_display_name(d: BLEDevice) -> str:
    if getattr(d, "name", None): return d.name
    md = getattr(d, "metadata", {}) or {}
    if isinstance(md, dict) and md.get("local_name"): return md.get("local_name")
    return ""

# ================== BLE Worker ==================
class BLEWorker(threading.Thread):
    def __init__(self, raw_queue: "queue.Queue", status_cb, connected_cb, device_found_cb):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self.raw_queue = raw_queue
        self.status_cb = status_cb
        self.connected_cb = connected_cb
        self.device_found_cb = device_found_cb
        self.client: Optional[BleakClient] = None
        self._stop = threading.Event()
        
        self.target_address = None
        self.target_uuid = None
        self.is_reconnecting = False
        self.manual_disconnect = False 
        self.seen_devices = set()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run())

    async def _run(self):
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.1)
        finally:
            if self.client: await self.disconnect()

    def stop(self):
        self._stop.set()
        try: self.loop.call_soon_threadsafe(self.loop.stop)
        except: pass

    def scan(self, timeout=10.0):
        self.seen_devices.clear()
        return asyncio.run_coroutine_threadsafe(self._scan(timeout), self.loop)

    def connect(self, device_or_address, char_uuid):
        self.manual_disconnect = False
        self.target_address = device_or_address
        self.target_uuid = char_uuid
        return asyncio.run_coroutine_threadsafe(self._connect(device_or_address, char_uuid), self.loop)

    def disconnect(self):
        self.manual_disconnect = True
        self.target_address = None 
        return asyncio.run_coroutine_threadsafe(self._disconnect(), self.loop)

    def _on_scan_result(self, device, advertisement_data):
        if device.address in self.seen_devices:
            return
        nm = safe_display_name(device)
        if nm and "Max" in nm:
            self.seen_devices.add(device.address)
            self.device_found_cb(device)

    async def _scan(self, timeout: float):
        self.status_cb("Scanning...")
        try:
            scanner = BleakScanner(detection_callback=self._on_scan_result)
            await scanner.start()
            await asyncio.sleep(timeout)
            await scanner.stop()
            self.status_cb("Scan complete.")
        except Exception as e:
            self.status_cb(f"Scan error: {e}")

    async def _disconnect(self):
        if self.client:
            try: await self.client.disconnect()
            except: pass
        self.client = None
        self.status_cb("Disconnected (Manual).")
        self.connected_cb(False)

    def _on_ble_disconnect(self, client):
        self.connected_cb(False)
        if not self.manual_disconnect and self.target_address and not self.is_reconnecting:
             asyncio.run_coroutine_threadsafe(self._attempt_reconnect(), self.loop)

    async def _attempt_reconnect(self):
        self.is_reconnecting = True
        self.status_cb("Disconnected. Attempting Reconnect...")
        start_time = time.time()
        max_duration = 60.0 
        
        while (time.time() - start_time) < max_duration:
            if self.manual_disconnect or not self.target_address: break 
            remaining = int(max_duration - (time.time() - start_time))
            self.status_cb(f"Reconnecting... ({remaining}s left)")
            try:
                await self._connect(self.target_address, self.target_uuid)
                if self.client and self.client.is_connected:
                    self.is_reconnecting = False
                    return 
            except Exception:
                pass
            await asyncio.sleep(2.0)
            
        self.is_reconnecting = False
        self.target_address = None
        self.status_cb("Connection Lost. Reconnect failed.")

    async def _connect(self, address: str, char_uuid: Optional[str]):
        if self.client:
            try: await self.client.disconnect()
            except: pass
        
        self.status_cb(f"Connecting to {address} …")
        self.client = BleakClient(address, disconnected_callback=self._on_ble_disconnect)
        
        try:
            await self.client.connect()
            if not self.client.is_connected: raise RuntimeError("Connect failed.")

            if sys.platform == "win32":
                try:
                    await self.client._backend.client.request_mtu_async(517)
                except Exception as e:
                    print(f"MTU Set Failed (Ignorable): {e}")

            self.status_cb("Connected. Discovering…")
            
            def _has_method(obj, name): return callable(getattr(obj, name, None))
            services = await self.client.get_services() if _has_method(self.client, "get_services") else self.client.services

            chosen = None
            if char_uuid:
                chosen = char_uuid.lower()
            else:
                for svc in services:
                    for ch in svc.characteristics:
                        if "notify" in ch.properties:
                            chosen = str(ch.uuid).lower()
                            break
                    if chosen: break
            
            if not chosen: raise RuntimeError("No notify char found.")
            
            await self.client.start_notify(chosen, self._on_notify)
            
            self.status_cb(f"Streaming on {chosen}…")
            self.connected_cb(True)
            return True
        except Exception as e:
            self.status_cb(f"Connect error: {e}")
            try: await self.client.disconnect()
            except: pass
            self.client = None
            self.connected_cb(False)
            if not self.is_reconnecting: raise e
            return False

    def _on_notify(self, _: int, data: bytearray):
        t_now = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.raw_queue.put((t_now, bytes(data)))

# ================== CSV Logger ==================
class CSVLogger:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.data_fp = None
        self.events_fp = None
        self.current_filename = None

    def start_data(self, custom_name: str = ""):
        if self.data_fp: return
        self.outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name.strip():
            safe_name = "".join([c for c in custom_name if c.isalnum() or c in (' ', '-', '_')]).strip()
            fname = f"{safe_name}_{stamp}.csv"
        else:
            fname = f"{stamp}.csv"
        path = self.outdir / fname
        self.current_filename = path.name
        self.data_fp = open(path, "w", buffering=65536)
        header = ",".join(DATA_COLUMNS + ["timestamp_iso"])
        self.data_fp.write(header + "\n")

    def stop_data(self):
        if self.data_fp:
            try: self.data_fp.flush(); self.data_fp.close()
            except: pass
        self.data_fp = None

    def write_rows(self, row0: List[int], row1: List[int], t_iso: str):
        if not self.data_fp: return
        self.data_fp.write(",".join(map(str, row0 + [t_iso])) + "\n")
        self.data_fp.write(",".join(map(str, row1 + [t_iso])) + "\n")

    def log_event(self, label: str):
        if not self.events_fp and self.current_filename:
            base = self.current_filename.replace(".csv", "")
            path = self.outdir / f"{base}_events.csv"
            self.events_fp = open(path, "w", buffering=1)
            self.events_fp.write("timestamp_iso,event\n")
        if self.events_fp:
            t_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
            safe = label.replace("\n", " ").replace(",", " ")
            self.events_fp.write(f"{t_iso},{safe}\n")

    def close(self):
        self.stop_data()
        if self.events_fp:
            try: self.events_fp.close()
            except: pass

# ================== Plots ==================
class FastRTPlotGrid:
    def __init__(self, parent, fs: float = 25.0, window_sec: float = 5.0, autoscale_every: int = 10):
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

        self.fig = Figure(figsize=(13.0, 7.5), dpi=100)
        self.axes = self.fig.subplots(12, 3, sharex=True)
        
        col_titles = ["660 nm", "940 nm", "850 nm"]
        for c in range(3): self.axes[0][c].set_title(col_titles[c], fontsize=9)
        for r in range(12):
            self.axes[r][0].set_ylabel(f"ch {r+1}", fontsize=7, rotation=0, labelpad=16, va="center")
            for c in range(3):
                ax = self.axes[r][c]
                ax.tick_params(labelsize=7, pad=1)
                ax.set_xlim(0, self.x_full[-1])
                if r < 11: ax.set_xticklabels([])
                if c > 0: ax.set_yticklabels([])

        self.lines = [None] * 36
        for idx in range(36):
            r = idx % 12; c = idx // 12
            (line,) = self.axes[r][c].plot([], [], lw=0.8)
            self.lines[idx] = line

        self.fig.tight_layout(pad=0.5)
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self.inner)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def clear(self):
        self.buf[:] = np.nan
        self.write_pos = 0
        self.samples_seen = 0

    def push_batch(self, rows_list: List[List[float]]):
        n_new = len(rows_list)
        if n_new == 0: return
        incoming = np.array(rows_list, dtype=np.float32).T 
        start_idx = self.write_pos
        end_idx = start_idx + n_new
        if end_idx <= self.W:
            self.buf[:, start_idx:end_idx] = incoming
        else:
            first_chunk = self.W - start_idx
            self.buf[:, start_idx:] = incoming[:, :first_chunk]
            second_chunk = n_new - first_chunk
            if second_chunk > self.W:
                 self.buf[:, :] = incoming[:, -self.W:]
                 self.write_pos = 0 
            else:
                 self.buf[:, :second_chunk] = incoming[:, first_chunk:]
        self.write_pos = (self.write_pos + n_new) % self.W
        self.samples_seen += n_new

    def _current_indices(self, n: int):
        n = min(int(n), self.W, self.samples_seen)
        if n <= 0: return np.array([], dtype=int)
        start = (self.write_pos - n) % self.W
        if start + n <= self.W: return np.arange(start, start + n)
        return np.concatenate([np.arange(start, self.W), np.arange(0, (start + n) % self.W)])

    def redraw(self):
        n = min(self.samples_seen, self.W)
        if n <= 1: return
        idx = self._current_indices(n)
        x = self.x_full[-n:]
        
        do_scale = (self._frame % self.autoscale_every == 0)
        for ch in range(36):
            y = self.buf[ch, idx]
            self.lines[ch].set_data(x, y)
            if do_scale and np.isfinite(y).any():
                ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
                if ymax == ymin: ymax = ymin + 1.0
                pad = 0.05 * (ymax - ymin)
                self.axes[ch % 12][ch // 12].set_ylim(ymin - pad, ymax + pad)
        
        self.canvas_mpl.draw_idle()
        self._frame += 1

    def resize_window(self, fs: float, window_sec: float):
        self.fs = fs
        W_new = max(50, int(round(fs * window_sec)))
        if W_new == self.W: return
        new_buf = np.full((36, W_new), np.nan, dtype=np.float32)
        n = min(self.samples_seen, self.W, W_new)
        if n > 0:
            idx = self._current_indices(n)
            new_buf[:, W_new - n:] = self.buf[:, idx]
        self.buf = new_buf
        self.W = W_new
        self.window_samples = W_new
        self.x_full = np.arange(self.W, dtype=np.float32) / float(self.fs)
        for r in range(12):
            for c in range(3): self.axes[r][c].set_xlim(0, self.x_full[-1])

# ================== Main GUI ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BLE Sensor Monitor")
        self._autoset_scaling()
        
        self.geometry("1600x780")
        
        # FIX: Define style here, AFTER self is created
        self.style = ttk.Style(self)
        self.style.configure("Record.TButton", foreground="red")
        
        # State
        self.fs = tk.DoubleVar(value=25.0)
        self.window_sec = tk.DoubleVar(value=5.0)
        self.enable_plots = tk.BooleanVar(value=True)
        self.outdir = tk.StringVar(value=str(Path("./logs").resolve()))
        self.recording = False
        self.record_start_time = None 
        
        self.last_pkt_counter = -1

        self.raw_queue = queue.Queue()  
        
        # FIX: Pass 'self.on_device_found' to constructor (it was missing!)
        self.ble = BLEWorker(
            self.raw_queue, 
            self.set_status, 
            self.on_connected_change, 
            self.on_device_found
        )
        self.ble.start()

        self.logger = CSVLogger(Path(self.outdir.get()))
        self.named_devices = []
        self.connected_device_name = None

        # --- Layout ---
        left = ttk.Frame(self); left.pack(side="left", fill="y", padx=8, pady=8)
        right = ttk.Frame(self); right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        # Device List
        ttk.Label(left, text="BLE Devices", font=("Arial", 9, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(left, height=10)
        self.listbox.pack(fill="x", pady=2)
        
        btn_row = ttk.Frame(left); btn_row.pack(fill="x", pady=4)
        self.btn_scan = ttk.Button(btn_row, text="Scan", command=self.on_scan)
        self.btn_scan.pack(side="left")
        self.btn_connect = ttk.Button(btn_row, text="Connect", command=self.on_connect)
        self.btn_connect.pack(side="left", padx=4)
        self.btn_disconnect = ttk.Button(btn_row, text="Disconnect", command=self.on_disconnect, state="disabled")
        self.btn_disconnect.pack(side="left", padx=4)
        
        self.lbl_connected = ttk.Label(left, text="Connected to: None", foreground="blue", font=("Arial", 9, "bold"))
        self.lbl_connected.pack(anchor="w", pady=(5,10))

        self.char_uuid = tk.StringVar()
        ttk.Label(left, text="Notify UUID (opt):").pack(anchor="w")
        ttk.Entry(left, textvariable=self.char_uuid).pack(fill="x")

        # Controls
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Checkbutton(left, text="Enable Real-time Plots", variable=self.enable_plots).pack(anchor="w")

        row = ttk.Frame(left); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Fs (Hz):").pack(side="left")
        ttk.Entry(row, textvariable=self.fs, width=6).pack(side="left")

        row = ttk.Frame(left); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Win (s):").pack(side="left")
        ttk.Entry(row, textvariable=self.window_sec, width=6).pack(side="left")

        # Recording
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Recording", font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.btn_record = ttk.Button(left, text="● Start Recording", command=self.on_toggle_record)
        self.btn_record.pack(fill="x", pady=5)
        
        self.lbl_timer = ttk.Label(left, text="00:00:00", font=("Consolas", 12))
        self.lbl_timer.pack(anchor="center", pady=2)
        
        ttk.Label(left, text="Folder:").pack(anchor="w")
        ttk.Entry(left, textvariable=self.outdir).pack(fill="x")

        # Events
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        self.event_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.event_var).pack(fill="x")
        ttk.Button(left, text="Log Event", command=self.on_log_event).pack(fill="x", pady=2)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(left, textvariable=self.status_var, wraplength=180).pack(side="bottom", anchor="w")

        # Plots
        self.plot = FastRTPlotGrid(right, fs=25.0, window_sec=5.0)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Start Loops
        self.after(10, self.process_raw_data)
        self.after(60, self.redraw_plot_canvas)

    def _autoset_scaling(self):
        try: self.tk.call('tk', 'scaling', self.winfo_fpixels('1i')/72.0)
        except: pass

    def set_status(self, msg):
        self.after(0, lambda: self.status_var.set(msg))

    def on_connected_change(self, is_connected):
        def _update():
            if is_connected:
                self.lbl_connected.configure(text=f"Connected to: {self.connected_device_name or 'Unknown'}", foreground="green")
                self.btn_disconnect.configure(state="normal")
                self.btn_connect.configure(state="disabled")
                self.last_pkt_counter = -1 # Reset
            else:
                self.lbl_connected.configure(text="Connected to: Disconnected", foreground="red")
                self.btn_disconnect.configure(state="disabled")
                self.btn_connect.configure(state="normal")
        self.after(0, _update)

    def on_device_found(self, device):
        def _add():
            nm = safe_display_name(device)
            if not any(device.address in item for item in self.listbox.get(0, "end")):
                self.named_devices.append(device)
                self.listbox.insert("end", f"{nm} [{device.address}]")
        self.after(0, _add)

    def on_scan(self):
        self.listbox.delete(0, "end")
        self.named_devices.clear()
        self.ble.scan()

    def on_connect(self):
        sel = self.listbox.curselection()
        if not sel: return
        dev = self.named_devices[sel[0]]
        self.connected_device_name = safe_display_name(dev)
        self.ble.connect(dev.address, self.char_uuid.get())
        self.plot.clear()

    def on_disconnect(self):
        self.ble.disconnect()

    def on_toggle_record(self):
        if not self.recording:
            fname = simpledialog.askstring("File Name", "Enter filename:", parent=self)
            if fname is None: return 
            
            self.recording = True
            
            out_path = Path(self.outdir.get())
            try:
                self.logger.outdir = out_path
                self.logger.start_data(custom_name=fname)
            except Exception as e:
                self.recording = False
                messagebox.showerror("File Error", f"Could not create file:\n{e}")
                return

            self.btn_record.configure(text="■ Stop Recording", style="Record.TButton")
            self.set_status(f"Recording: {self.logger.current_filename}")
            
            self.record_start_time = datetime.now()
            self._update_record_timer()
        else:
            self.recording = False
            self.logger.stop_data()
            self.btn_record.configure(text="● Start Recording", style="TButton")
            self.set_status("Recording saved.")
            self.lbl_timer.configure(text="00:00:00")

    def _update_record_timer(self):
        if self.recording:
            delta = datetime.now() - self.record_start_time
            s_delta = str(delta).split('.')[0]
            self.lbl_timer.configure(text=s_delta)
            self.after(1000, self._update_record_timer)

    def on_log_event(self):
        if self.event_var.get():
            self.logger.log_event(self.event_var.get())
            self.event_var.set("")

    def process_raw_data(self):
        temp_batch = []
        try:
            for _ in range(100):
                item = self.raw_queue.get_nowait()
                t_iso = item[0]
                raw_bytes = item[1]
                
                data = process_raw_packet(raw_bytes)
                if not data: continue
                
                pkt = data["pkt"]
                if self.last_pkt_counter != -1:
                    diff = (pkt - self.last_pkt_counter) % 256
                    if diff != 1:
                        print(f"[DROP DETECTED] Prev: {self.last_pkt_counter}, Curr: {pkt}, Lost: {diff-1}")
                self.last_pkt_counter = pkt

                if self.recording:
                    self.logger.write_rows(data["row0"], data["row1"], t_iso)
                
                if self.enable_plots.get():
                    temp_batch.append(data["row0"][:36])
                    temp_batch.append(data["row1"][:36])
        except queue.Empty:
            pass
        
        if temp_batch:
            self.plot.push_batch(temp_batch)
        
        self.after(10, self.process_raw_data)

    def redraw_plot_canvas(self):
        if self.enable_plots.get():
            self.plot.resize_window(self.fs.get(), self.window_sec.get())
            self.plot.redraw()
        self.after(60, self.redraw_plot_canvas)

    def on_close(self):
        try: self.logger.close()
        except: pass
        try: self.ble.stop()
        except: pass
        self.destroy()

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    App().mainloop()