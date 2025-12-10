#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLE Desktop App (Mac + Windows)
--------------------------------
- Scans for BLE devices and lists them (click to connect)
- Subscribes to a notify characteristic
- Decodes packets using tattoo_init.txt via packet_init.py/packet_decoder.py
- Realtime plotting via pyqtgraph with a bounded ring buffer to avoid UI lag
- NEW: "Decoded (latest)" table that shows parsed values (timestamp + selected channels)

Install deps (recommend a fresh venv):
  python -m pip install --upgrade pip
  python -m pip install bleak PySide6 pyqtgraph qasync numpy

Run:
  python ble_desktop_app.py
"""
from __future__ import annotations

import asyncio
import sys
import os
from dataclasses import dataclass
from typing import Optional, List
from collections import deque
from datetime import datetime

import numpy as np
import csv

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

import pyqtgraph as pg
from qasync import QEventLoop, asyncSlot

from bleak import BleakScanner, BleakClient

# Import your packet init/decoder (must be in same folder or on PYTHONPATH)
try:
    from packet_init import parse_init_file
    from packet_decoder import PacketDecoder
except Exception:
    parse_init_file = None
    PacketDecoder = None


DEFAULT_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"  # Nordic UART Service (example)
DEFAULT_CHAR_UUID    = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # NUS RX notify characteristic


class RingBuffer:
    """Simple numpy ring buffer: shape = (C, T)."""
    def __init__(self, num_channels: int, window: int):
        self.C = num_channels
        self.T = window
        self.buf = np.full((self.C, self.T), np.nan, dtype=np.float32)
        self.t = 0
        self.filled = 0

    def append(self, sample: np.ndarray):
        self.buf[:, self.t] = sample
        self.t = (self.t + 1) % self.T
        self.filled = min(self.filled + 1, self.T)

    def get(self) -> np.ndarray:
        N = self.filled
        if N < self.T:
            return self.buf[:, :N]
        start = self.t
        if start == 0:
            return self.buf
        return np.concatenate([self.buf[:, start:], self.buf[:, :start]], axis=1)


@dataclass
class DecoderState:
    cfg_path: Optional[str] = None
    decoder: Optional[PacketDecoder] = None
    order: Optional[List[int]] = None      # channel order derived from cfg.signal_order (first 36 unique)
    plot_indices: Optional[List[int]] = None  # which indices (from order) to plot


class BLEController(QtCore.QObject):
    """Handles scanning, connecting and notifications via bleak (async)."""
    device_found = QtCore.Signal(list)  # list of (name, address)
    status = QtCore.Signal(str)
    connected = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client: Optional[BleakClient] = None
        self._notify_char_uuid: Optional[str] = None

        # Fast path callback (UI-safe work happens on a timer)
        self.on_packet_bytes = None  # callable(bytes)

    async def scan(self, timeout: float = 5.0):
        self.status.emit("Scanning for devices...")
        try:
            devices = await BleakScanner.discover(timeout=timeout)
        except Exception as e:
            self.status.emit(f"Scan failed: {e}")
            self.device_found.emit([])
            return
        rows = []
        for d in devices:
            name = (d.name or "").strip() or "(unknown)"
            rows.append([name, d.address])
        self.device_found.emit(rows)
        self.status.emit(f"Found {len(rows)} device(s).")

    async def connect(self, address: str, service_uuid: str, char_uuid: str):
        await self.disconnect()  # in case
        self.status.emit(f"Connecting to {address} ...")
        self.client = BleakClient(address, disconnected_callback=self._on_disconnect)
        try:
            ok = await self.client.connect(timeout=10.0)
            if not ok:
                raise RuntimeError("BleakClient.connect() returned False")
        except Exception as e:
            self.status.emit(f"Connect failed: {type(e).__name__}: {e}")
            self.client = None
            self.connected.emit(False)
            return
        self.status.emit(f"Connected: {self.client.address}")
        self.connected.emit(True)
        self._notify_char_uuid = char_uuid
        # Verify service/char
        await self._ensure_notify(service_uuid, char_uuid)

    async def _ensure_notify(self, service_uuid: str, char_uuid: str):
        if self.client is None:
            return
        # get services (across Bleak versions)
        services = None
        get_services = getattr(self.client, "get_services", None)
        services = await get_services() if callable(get_services) else getattr(self.client, "services", None)

        svc = services.get_service(service_uuid) if services else None
        if svc is None:
            s_list = "\n".join([f"  {s.uuid} ({s.description})" for s in services]) if services else "  (none)"
            self.status.emit(f"Service {service_uuid} not found. Available:\n{s_list}")
            raise RuntimeError("Service not found")

        char = svc.get_characteristic(char_uuid)
        if char is None or "notify" not in char.properties:
            c_list = "\n".join([f"  {c.uuid} props={c.properties}" for c in svc.characteristics])
            self.status.emit(f"Characteristic {char_uuid} not found / notifiable in {service_uuid}.\n{c_list}")
            raise RuntimeError("Characteristic not found / notifiable")

        # start notifications
        await self.client.start_notify(char_uuid, self._handle_notification)
        self.status.emit(f"Subscribed to notifications on {char_uuid}")

    def _on_disconnect(self, _client: BleakClient):
        self.status.emit("Disconnected")
        self.connected.emit(False)

    async def disconnect(self):
        if self.client:
            try:
                if self._notify_char_uuid:
                    try:
                        await self.client.stop_notify(self._notify_char_uuid)
                    except Exception:
                        pass
                await self.client.disconnect()
            except Exception:
                pass
            finally:
                self.client = None
                self._notify_char_uuid = None
                self.connected.emit(False)
                self.status.emit("Disconnected")

    # Runs on the asyncio event loop thread — keep it very light
    def _handle_notification(self, _char, data: bytes):
        if self.on_packet_bytes:
            try:
                self.on_packet_bytes(data)
            except Exception:
                pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("BLE Desktop App (plots + decoded table)")
        self.resize(1300, 820)

        # BLE controller
        self.ble = BLEController()
        self.ble.device_found.connect(self._update_device_list)
        self.ble.status.connect(self._set_status)
        self.ble.connected.connect(self._on_connected_changed)

        # Decoder state
        self.dec = DecoderState()

        # Plot / data buffers
        self.num_plot_channels = 6
        self.window_len = 1000  # samples
        self.ring = RingBuffer(self.num_plot_channels, self.window_len)

        # Decoded rows buffer for table (timestamp + selected channels)
        self.decoded_rows = deque(maxlen=500)  # recent rows for UI
        self._rows_rendered = 0                # how many rows already in the table

        # Build UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left controls
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 0)

        hb = QtWidgets.QHBoxLayout()
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.setEnabled(False)
        hb.addWidget(self.btn_scan)
        hb.addWidget(self.btn_connect)
        hb.addWidget(self.btn_disconnect)
        # Recording controls
        self.btn_start_record = QtWidgets.QPushButton("Start Recording")
        self.btn_stop_record = QtWidgets.QPushButton("Stop Recording")
        self.btn_stop_record.setEnabled(False)
        hb.addWidget(self.btn_start_record)
        hb.addWidget(self.btn_stop_record)
        left.addLayout(hb)

        self.tbl_devices = QtWidgets.QTableWidget(0, 2)
        self.tbl_devices.setHorizontalHeaderLabels(["Name", "Address"])
        self.tbl_devices.horizontalHeader().setStretchLastSection(True)
        self.tbl_devices.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_devices.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left.addWidget(self.tbl_devices, 1)

        form = QtWidgets.QFormLayout()
        self.le_service = QtWidgets.QLineEdit(DEFAULT_SERVICE_UUID)
        self.le_char = QtWidgets.QLineEdit(DEFAULT_CHAR_UUID)
        form.addRow("Service UUID:", self.le_service)
        form.addRow("Notify Char UUID:", self.le_char)

        self.le_init = QtWidgets.QLineEdit("tattoo_init.txt")
        self.btn_browse = QtWidgets.QPushButton("Browse…")
        hb2 = QtWidgets.QHBoxLayout()
        hb2.addWidget(self.le_init, 1)
        hb2.addWidget(self.btn_browse)
        form.addRow("Init file:", hb2)

        self.le_channels = QtWidgets.QLineEdit("0,1,2,3,4,5")
        form.addRow("Channels to plot/table:", self.le_channels)

        # Recording file inputs
        self.le_decoded_out = QtWidgets.QLineEdit("decoded.csv")
        self.le_raw_out = QtWidgets.QLineEdit("raw_packets.bin")
        form.addRow("Decoded CSV out:", self.le_decoded_out)
        form.addRow("Raw packet dump:", self.le_raw_out)

        self.spn_window = QtWidgets.QSpinBox()
        self.spn_window.setRange(100, 20000)
        self.spn_window.setSingleStep(100)
        self.spn_window.setValue(self.window_len)
        form.addRow("Plot window (samples):", self.spn_window)

        self.spn_interval = QtWidgets.QSpinBox()
        self.spn_interval.setRange(10, 1000)
        self.spn_interval.setValue(50)
        form.addRow("Update interval (ms):", self.spn_interval)

        left.addLayout(form)

        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        left.addWidget(self.lbl_status)

        left.addStretch(1)

        # Right: split vertically — plots on top, decoded table below
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)

        self.graph = pg.GraphicsLayoutWidget()
        right.addWidget(self.graph, 3)

        self.plots = []
        self.curves = []
        self._build_plots()

        # Decoded table
        group = QtWidgets.QGroupBox("Decoded (latest)")
        vbox = QtWidgets.QVBoxLayout(group)
        self.tbl_decoded = QtWidgets.QTableWidget(0, 1)  # will set columns after init load
        self.tbl_decoded.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_decoded.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_decoded.setAlternatingRowColors(True)
        self.tbl_decoded.verticalHeader().setVisible(False)
        self.tbl_decoded.horizontalHeader().setStretchLastSection(True)
        vbox.addWidget(self.tbl_decoded)
        right.addWidget(group, 2)

        # Timed GUI update
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(self.spn_interval.value())

        # Wire UI actions
        self.btn_scan.clicked.connect(self._scan_clicked)
        self.btn_connect.clicked.connect(self._connect_clicked)
        self.btn_disconnect.clicked.connect(self._disconnect_clicked)
        self.btn_start_record.clicked.connect(self._start_record_clicked)
        self.btn_stop_record.clicked.connect(self._stop_record_clicked)
        self.btn_browse.clicked.connect(self._browse_init)
        self.spn_window.valueChanged.connect(self._update_window_len)
        self.spn_interval.valueChanged.connect(self._update_timer_interval)

        # Assign the notification -> decode -> buffers pipeline
        self.ble.on_packet_bytes = self._on_raw_bytes_from_ble

        # Mutable per-session decode context
        self._pkt_decoder = None
        self._order36 = None  # first unique 36 indices for vector assembly
        self._plot_indices = [0,1,2,3,4,5]  # indices within order36 to plot/table

        # Recording state
        self._recording = False
        self._decoded_f = None
        self._raw_f = None

        # Parse channel indices
        self._parse_channel_text()
        self._refresh_decoded_table_headers()  # set initial headers (no init yet)

    # ---------- UI plumbing ----------
    def _build_plots(self):
        self.graph.clear()
        self.plots.clear()
        self.curves.clear()
        N = self.num_plot_channels
        for i in range(N):
            p = self.graph.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setClipToView(True)
            p.setLabel('left', f'ch{i}')
            if i < N - 1:
                p.setXLink(self.plots[0] if self.plots else None)
                p.setMenuEnabled(False)
            c = p.plot([], [])
            self.plots.append(p)
            self.curves.append(c)
        self.plots[-1].setLabel('bottom', 'sample')

    def _update_window_len(self, v: int):
        self.window_len = int(v)
        self.ring = RingBuffer(self.num_plot_channels, self.window_len)

    def _update_timer_interval(self, v: int):
        self.timer.setInterval(int(v))

    def _set_status(self, txt: str):
        self.lbl_status.setText(txt)

    def _update_device_list(self, rows: List[List[str]]):
        self.tbl_devices.setRowCount(len(rows))
        for r, (name, addr) in enumerate(rows):
            self.tbl_devices.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            self.tbl_devices.setItem(r, 1, QtWidgets.QTableWidgetItem(addr))

    def _selected_address(self) -> Optional[str]:
        sel = self.tbl_devices.selectedItems()
        if not sel:
            return None
        row = sel[0].row()
        addr = self.tbl_devices.item(row, 1).text()
        return addr

    def _browse_init(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select init file", "", "Text files (*.txt);;All files (*)")
        if fn:
            self.le_init.setText(fn)

    def _parse_channel_text(self):
        txt = self.le_channels.text().strip()
        idx = []
        if txt:
            for part in txt.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    i = int(part)
                    if 0 <= i < 36:
                        idx.append(i)
                except Exception:
                    pass
        if not idx:
            idx = list(range(6))
        self._plot_indices = idx[:self.num_plot_channels]
        self._refresh_decoded_table_headers()

    # ---------- Recording control ----------
    def _make_timestamped(self, base: str) -> str:
        base = base.strip() or base
        name, ext = os.path.splitext(base)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{ts}{ext}"

    def _start_record_clicked(self):
        if self._recording:
            return
        decoded_base = self.le_decoded_out.text().strip() or "decoded.csv"
        raw_base = self.le_raw_out.text().strip() or "raw_packets.bin"
        decoded_fn = self._make_timestamped(decoded_base)
        raw_fn = self._make_timestamped(raw_base)
        try:
            # open CSV and write header (timestamp + full order36)
            self._decoded_f = open(decoded_fn, "w", newline="")
            self._decoded_writer = csv.writer(self._decoded_f)
            # build header: time + mapped labels for the first unique 36 channels (or raw numeric labels if unknown)
            hdr = ["timestamp_iso"]
            if self._order36:
                for pos, sig_idx in enumerate(self._order36):
                    if pos < 12:
                        group = "660"; which = pos + 1
                    elif pos < 24:
                        group = "940"; which = pos - 12 + 1
                    else:
                        group = "850"; which = pos - 24 + 1
                    hdr.append(f"{group}_{which}")
            else:
                # fallback: numeric headers 0..35
                hdr.extend([str(i) for i in range(36)])
            # Packet counter (if device had one) is not included in order36 mapping; keep as needed
            self._decoded_writer.writerow(hdr)

            # open raw dump
            self._raw_f = open(raw_fn, "wb")

            self._recording = True
            self._set_status(f"Recording -> {decoded_fn} , {raw_fn}")
            self.btn_start_record.setEnabled(False)
            self.btn_stop_record.setEnabled(True)
        except Exception as e:
            self._set_status(f"Failed to start recording: {e}")
            try:
                if self._decoded_f:
                    self._decoded_f.close()
            except Exception:
                pass

    def _stop_record_clicked(self):
        if not self._recording:
            return
        try:
            if self._decoded_f:
                try:
                    self._decoded_f.flush()
                    self._decoded_f.close()
                except Exception:
                    pass
                self._decoded_f = None
            if self._raw_f:
                try:
                    self._raw_f.flush()
                    self._raw_f.close()
                except Exception:
                    pass
                self._raw_f = None
        finally:
            self._recording = False
            self.btn_start_record.setEnabled(True)
            self.btn_stop_record.setEnabled(False)
            self._set_status("Recording stopped")

    def _refresh_decoded_table_headers(self):
        # columns: timestamp + selected channel labels
        headers = ["time"]
        for ch_idx in self._plot_indices:
            headers.append(self._label_for(ch_idx))
        self.tbl_decoded.setColumnCount(len(headers))
        self.tbl_decoded.setHorizontalHeaderLabels(headers)

    # ---------- Async actions ----------
    @asyncSlot()
    async def _scan_clicked(self):
        await self.ble.scan(timeout=5.0)

    @asyncSlot()
    async def _connect_clicked(self):
        addr = self._selected_address()
        if not addr:
            self._set_status("Select a device from the list first.")
            return

        # Prepare decoder if init file exists
        cfg_path = self.le_init.text().strip()
        self._pkt_decoder = None
        self._order36 = None
        if cfg_path and os.path.isfile(cfg_path) and parse_init_file and PacketDecoder:
            try:
                cfg = parse_init_file(cfg_path)
                self._pkt_decoder = PacketDecoder.from_init(cfg)
                # Build unique order36 (first time each channel appears)
                seen = set()
                order = []
                for s in cfg.signal_order:
                    if s not in seen and s != 36:
                        seen.add(s)
                        order.append(s)
                        if len(order) == 36:
                            break
                self._order36 = order
                # Update plot and table labels to human-readable if possible
                for p, ch_idx in zip(self.plots, self._plot_indices):
                    p.setLabel('left', self._label_for(ch_idx))
                self._refresh_decoded_table_headers()
                self._set_status(f"Loaded init ({len(cfg.signals)} signals).")
            except Exception as e:
                self._set_status(f"Init parse failed: {e}")
                self._pkt_decoder = None
                self._order36 = None

        # Parse channels again in case the user changed them
        self._parse_channel_text()

        # Reset buffers
        self.ring = RingBuffer(self.num_plot_channels, self.window_len)
        self.decoded_rows.clear()
        self._rows_rendered = 0

        # Connect
        try:
            await self.ble.connect(addr, self.le_service.text().strip(), self.le_char.text().strip())
            self.btn_disconnect.setEnabled(True)
            self.btn_connect.setEnabled(False)
        except Exception as e:
            self._set_status(str(e))

    @asyncSlot()
    async def _disconnect_clicked(self):
        await self.ble.disconnect()
        self.btn_disconnect.setEnabled(False)
        self.btn_connect.setEnabled(True)

    def _on_connected_changed(self, ok: bool):
        if not ok:
            self.btn_disconnect.setEnabled(False)
            self.btn_connect.setEnabled(True)

    # ---------- Notify pipeline ----------
    def _label_for(self, ch_idx_from_order36: int) -> str:
        """Return a human-ish label for plot/table given index in order36 (0..35)."""
        if self._order36 is None or ch_idx_from_order36 >= len(self._order36):
            return f"ch{ch_idx_from_order36}"
        sig_idx = self._order36[ch_idx_from_order36]
        pos = ch_idx_from_order36
        if pos < 12:
            group = "660"; which = pos + 1
        elif pos < 24:
            group = "940"; which = pos - 12 + 1
        else:
            group = "850"; which = pos - 24 + 1
        return f"{group}_{which} (sig {sig_idx})"

    def _on_raw_bytes_from_ble(self, data: bytes):
        """Called on each BLE notification (asyncio thread). Keep work minimal."""
        if self._pkt_decoder is None or self._order36 is None:
            return
        try:
            packets = self._pkt_decoder.feed(bytes(data))
        except Exception:
            return
        if not packets:
            return

        # For each decoded packet:
        now_iso = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        for pkt in packets:
            # Build sample vector for plots (selected channels)
            sample = np.zeros((self.num_plot_channels,), dtype=np.float32)
            row = [now_iso]
            for i, ord_pos in enumerate(self._plot_indices[:self.num_plot_channels]):
                if ord_pos >= len(self._order36):
                    val = np.nan
                else:
                    sig_idx = self._order36[ord_pos]
                    v = pkt.get(sig_idx, np.nan)
                    try:
                        val = float(v)
                    except Exception:
                        val = np.nan
                sample[i] = val
                row.append("" if np.isnan(val) else f"{val:.3f}")

            # push to ring buffer (for plots)
            self.ring.append(sample)
            # push to decoded table buffer (for UI thread)
            self.decoded_rows.append(row)

    # ---------- GUI update ----------
    def _on_timer(self):
        # Plot refresh
        data = self.ring.get()  # (C, N)
        if data.size != 0:
            N = data.shape[1]
            max_points = 5000
            if N > max_points:
                step = int(np.ceil(N / max_points))
                xs = np.arange(0, N, step, dtype=np.int32)
                for i, curve in enumerate(self.curves[:data.shape[0]]):
                    ys = data[i, ::step]
                    curve.setData(xs, ys)
            else:
                xs = np.arange(N, dtype=np.int32)
                for i, curve in enumerate(self.curves[:data.shape[0]]):
                    curve.setData(xs, data[i])
            for p in self.plots[:data.shape[0]]:
                p.setXRange(max(0, N - self.window_len), N, padding=0.01)

        # Decoded table refresh (append only the new rows)
        new_rows = len(self.decoded_rows)
        if new_rows > self._rows_rendered:
            self.tbl_decoded.setUpdatesEnabled(False)
            for row_idx in range(self._rows_rendered, new_rows):
                r = self.decoded_rows[row_idx]
                dest = self.tbl_decoded.rowCount()
                self.tbl_decoded.insertRow(dest)
                for c, val in enumerate(r):
                    self.tbl_decoded.setItem(dest, c, QtWidgets.QTableWidgetItem(val))
                # Auto-scroll to bottom as rows come in
                if row_idx == new_rows - 1:
                    self.tbl_decoded.scrollToBottom()
            self.tbl_decoded.setUpdatesEnabled(True)
            self._rows_rendered = new_rows


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False, useOpenGL=False)

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    w = MainWindow()
    w.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
