from __future__ import annotations

import functools
import json
import math
import socket
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy missing
    np = None

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtGui import QIcon
except Exception:  # pragma: no cover - PyQt5 missing
    QtCore = QtGui = QtWidgets = None
    QIcon = None

try:  # OpenCV might be missing in some environments
    import cv2
except Exception:  # pragma: no cover - graceful fallback
    cv2 = None

if QtCore is None or np is None or cv2 is None:
    print("Required dependencies are missing: PyQt5, NumPy or OpenCV.")
    sys.exit(1)
class HCRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        buf = b""
        depth = 0
        start = 0
        self.request.settimeout(0.1)
        while True:
            try:
                data = self.request.recv(4096)
                if not data:
                    break
                buf += data
                i = 0
                while i < len(buf):
                    bch = buf[i:i+1]
                    if bch == b"{":
                        if depth == 0:
                            start = i
                        depth += 1
                    elif bch == b"}" and depth:
                        depth -= 1
                        if depth == 0:
                            seg = buf[start:i + 1]
                            text = seg.decode("utf-8", errors="ignore")
                            self.server.on_message(text)
                            buf = buf[i + 1:]
                            i = -1
                    i += 1
            except socket.timeout:
                continue

class HCVisionServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    def __init__(self, host, port, on_message):
        self.on_message = on_message
        super().__init__((host, port), HCRequestHandler)

def start_server(host="0.0.0.0", port=9760, on_message=lambda x: None):
    svr = HCVisionServer(host, port, on_message)
    threading.Thread(target=svr.serve_forever, daemon=True).start()
    return svr

class CamScanner(QtCore.QThread):
    resultReady = QtCore.pyqtSignal(list)

    def run(self):
        cams = list_camera_indices()
        self.resultReady.emit(cams)


# 辅助函数: 资源路径 (兼容 PyInstaller)
def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base, rel))

def load_colors(path: str) -> List[ColorCfg]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        colors = []
        for item in data:
            colors.append(
                ColorCfg(
                    item["name"],
                    tuple(item["bgr"]),
                    np.array(item["lower"], dtype=np.uint8),
                    np.array(item["upper"], dtype=np.uint8),
                )
            )
        print(f"[颜色] 已加载 {len(colors)} 种颜色配置")
        return colors
    except Exception as e:
        print("[颜色] 加载失败:", e)
        return []

# 数据类: 颜色配置 (HSV 范围 + UI 辅助)
@dataclass
class ColorCfg:
    name: str
    bgr: tuple  # 绘制颜色的 BGR 值
    lower: np.ndarray
    upper: np.ndarray
    # UI 滑块引用将在运行时填充
    sliders: Dict[str, "HSVSlider"] = field(default_factory=dict)

    @property
    def group_title(self) -> str:
        return f"HSV {self.name.title()}"

    @property
    def mask_button_title(self) -> str:
        return f"{self.name.title()} 掩膜"

# GUI 组件
class HSVSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, text: str, mn: int, mx: int, val: int, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(text)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(mn, mx)
        self.slider.setValue(val)
        self.val_lbl = QtWidgets.QLabel(str(val))
        self.val_lbl.setFixedWidth(32)
        lay.addWidget(self.label)
        lay.addWidget(self.slider)
        lay.addWidget(self.val_lbl)
        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, v):
        self.val_lbl.setText(str(v))
        self.valueChanged.emit(v)

    def value(self):
        return self.slider.value()


class MaskWindow(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent,QtCore.Qt.Window)
        self.setWindowTitle(title)
        self.resize(400, 300)
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        QtWidgets.QVBoxLayout(self).addWidget(self.label)

    def update_mask(self, mask_np: np.ndarray):
        h, w = mask_np.shape
        qimg = QtGui.QImage(mask_np.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.label.setPixmap(pix)

# 列出当前系统可用的摄像头索引
def list_camera_indices(max_index: int = 4,backend=cv2.CAP_MSMF) -> List[int]:
    valid = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            valid.append(idx)
            cap.release()
    return valid

class TcpSender:
    """Simple TCP client with background receive loop."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6000,
                 on_recv=None):
        self.host = host
        self.port = port
        self.on_recv = on_recv
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
            print(f"[TCP] 已连接 {self.host}:{self.port}")
        except Exception as e:
            self.sock.close()
            print("[TCP] 连接失败:", e)
            raise
        else:
            threading.Thread(target=self._recv_loop, daemon=True).start()

    def _recv_loop(self):
        buf = b""
        depth = 0
        start = 0
        while True:
            try:
                data = self.sock.recv(4096)
                if not data:
                    print("[TCP] 连接已关闭")
                    break
                buf += data
                i = 0
                while i < len(buf):
                    bch = buf[i:i+1]
                    if bch == b"{":
                        if depth == 0:
                            start = i
                        depth += 1
                    elif bch == b"}" and depth:
                        depth -= 1
                        if depth == 0:
                            seg = buf[start:i + 1]
                            text = seg.decode("utf-8", errors="ignore")
                            if self.on_recv:
                                self.on_recv(text)
                            else:
                                print("[TCP] 收到:", text)
                            buf = buf[i + 1:]
                            i = -1
                    i += 1
            except Exception as e:
                print("[TCP] 接收失败:", e)
                break

    def send_data(self, msg: str):
        try:
            self.sock.sendall(msg.encode("utf-8") + b"\n")
            print("[TCP] 已发送:", msg)
        except Exception as e:
            print("[TCP] 发送失败:", e)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        finally:
            self.sock.close()

# 全局常量
CHS = {"circle": "圆形","triangle": "三角形", "rect": "正方形"}
MIN_AREA, MAX_AREA = 500, 300_000
MIN_CIRCULARITY = 0.65
ROI_RATIO = 0.5

def _side_lengths(pts: np.ndarray) -> list[float]:
    pts = pts.reshape(-1, 2)
    return [math.hypot(*(pts[(i+1) % len(pts)] - pts[i])) for i in range(len(pts))]

def classify_contour(cnt, circularity):
    peri  = cv2.arcLength(cnt, True)
    poly  = cv2.approxPolyDP(cnt, 0.04 * peri, True)   # ≈2% 精度
    verts = len(poly)
    if verts == 3:
        sides = _side_lengths(poly)
        ratio = max(sides) / min(sides)                 # ≈1 ⇒ 等边
        if ratio <= 1.20:                               # 允许 ≤20% 误差
            return "triangle"
    elif verts == 4:
        x, y, w, h = cv2.boundingRect(poly)
        ar = w / float(h)
        if 0.85 <= ar <= 1.15:                          # 长宽接近
            # 再用对角线法判断正交性（可选）
            pts = poly.reshape(-1, 2)
            v1  = pts[1] - pts[0]
            v2  = pts[2] - pts[1]
            angle = abs(np.dot(v1, v2) /
                        (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5))
            if angle <= 0.15:                           # 夹角≈90°，阈值自己调
                return "rect"
    elif circularity > 0.75:                            # 保守一点
        return "circle"
    return None

def detect_shapes(frame_bgr: np.ndarray,
                  color_cfgs: List[ColorCfg],
                  shapes_enabled: set) -> List[tuple]:
    labels = []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    for cfg in color_cfgs:
        if not cfg.sliders:
            continue
        mask = cv2.inRange(hsv, cfg.lower, cfg.upper)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if not (MIN_AREA < area < MAX_AREA):
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            shape = classify_contour(c, circ)
            if not (shape and shape in shapes_enabled):
                continue
            label_txt = f"{cfg.name}-{CHS[shape]}"
            qcolor    = QtGui.QColor(*reversed(cfg.bgr))
            # ====== 画几何形状 ======
            if shape == "circle":
                (x, y), r = cv2.minEnclosingCircle(c)
                cv2.circle(frame_bgr, (int(x), int(y)), int(r), cfg.bgr, 2)
                text_pos = (int(x - r), int(y - r - 6))
            else:
                poly = cv2.approxPolyDP(c, 0.04*peri, True)
                cv2.polylines(frame_bgr, [poly], True, cfg.bgr, 2)
                bx, by, _, _ = cv2.boundingRect(poly)
                text_pos = (bx, by - 6)
            # OpenCV 里先不写文字 → 交给 Qt
            labels.append((label_txt, text_pos, qcolor))
    return labels

class QTextEditLogger(QtCore.QObject):
    append_text = QtCore.pyqtSignal(str)

    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.append_text.connect(self.widget.append)

    def write(self, msg):
        if msg.strip():   # 避免空行
            self.append_text.emit(msg.strip())

    def flush(self):
        pass

# 主窗口
class MainWindow(QtWidgets.QWidget):
    FPS_CALC_INTERVAL = 30
    tcp_msg_sig = QtCore.pyqtSignal(str)

    def __init__(self, colors: List[ColorCfg]):
        super().__init__(None, QtCore.Qt.Window)
        self.colors = {c.name: c for c in colors}
        self.mask_windows: Dict[str, MaskWindow] = {}

        self.setWindowTitle("Camera")
        self.resize(1200, 720)

        # 布局分割: 控制面板 | 视频显示
        hbox = QtWidgets.QHBoxLayout(self)
        self.ctrl_panel = QtWidgets.QFrame(); self.ctrl_panel.setFixedWidth(310)
        hbox.addWidget(self.ctrl_panel)
        video_panel = QtWidgets.QVBoxLayout()

        # 摄像头画面
        self.video_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_lbl.setMinimumSize(640, 480)
        video_panel.addWidget(self.video_lbl, 1)

        # CMD输出框
        self.cmd_output = QtWidgets.QTextEdit()
        self.cmd_output.setReadOnly(True)
        self.cmd_output.setFixedHeight(150)
        video_panel.addWidget(self.cmd_output)
        # 重定向 print 输出到日志框
        logger = QTextEditLogger(self.cmd_output)
        sys.stdout = logger
        sys.stderr = logger

        # 把右侧整体加入主布局
        video_container = QtWidgets.QWidget()
        video_container.setLayout(video_panel)
        hbox.addWidget(video_container, 1)

        self._init_controls()
        self.load_cmd_map()

        # 摄像头初始化
        self.capture = None
        self.cam_combo.currentIndexChanged.connect(self.open_camera)
        self.open_camera()

        # 定时器
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(30)
        self.frame_cnt = 0
        self.tcp_msg_sig.connect(self._process_tcp_msg)
        self.svr = start_server(on_message=self.handle_tcp_msg)
        self._running = True

    # 异步摄像头扫描
    def start_scanning(self):
        if getattr(self, 'scanner', None) and self.scanner.isRunning():
            return
        self.scanner = CamScanner(self)
        self.scanner.resultReady.connect(self._populate_cameras_async)
        self.scanner.start()

    def _populate_cameras_async(self, cams: List[int]):
        self.cam_combo.clear()
        for idx in cams:
            self.cam_combo.addItem(f"Camera {idx}", idx)
        if cams:
            self.cam_combo.setCurrentIndex(0)
            self.open_camera()
    def _save_server_config(self, path="tcp.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        if "cmd_map" not in data:
            data["cmd_map"] = getattr(self, "cmd_map", {})

        data["server"] = {
            "host": self.ip_input.text().strip(),
            "port": int(self.port_input.text())
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("[配置] 已保存服务器地址到 tcp.json")

    def start_heartbeat(self, interval: int = 10):
        if getattr(self, "_hb_thread", None):
            return

        def loop():
            while self._running:
                if self.tcp_sender:
                    hb = {"dsID": "www.hc-system.com.cam", "reqType": "heartbeat"}
                    self._send_json(hb)
                time.sleep(interval)

        self._hb_thread = threading.Thread(target=loop, daemon=True)
        self._hb_thread.start()

    def _send_json(self, obj):
        if not self.tcp_sender:
            print("[TCP] 未连接")
            return
        text = json.dumps(obj, ensure_ascii=False)
        self.tcp_sender.send_data(text)
        self.append_log(f"[发送] {text}")

    def send_position_data(self, cam_id: int, detections: list):
        frame = {
            "dsID": "www.hc-system.com.cam",
            "dsData": [{"camID": str(cam_id), "data": detections}],
        }
        self._send_json(frame)

    def build_model_list_reply(self):
        return {"dsID": "www.hc-system.com.cam", "models": []}

    def do_capture_and_send(self, cam_id: int):
        if not self.capture or not self.capture.isOpened():
            print("[摄像头] 未就绪")
            return
        if cam_id != self.cam_combo.currentData():
            print(f"[警告] 请求的相机 {cam_id} 与当前选择的不一致")
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            print("[摄像头] 读取失败")
            return

        shapes_enabled = {
            s for s, chk in [
                ("circle", self.chk_circle),
                ("triangle", self.chk_tri),
                ("rect", self.chk_rect),
            ]
            if chk.isChecked()
        }
        labels = detect_shapes(frame, list(self.colors.values()), shapes_enabled)

        if not labels:
            print("[识别] 未检测到目标")
        for text, _pos, _col in labels:
            if text in self.cmd_map:
                msg = self.cmd_map[text]
                if self.tcp_sender:
                    self.tcp_sender.send_data(msg)
                    self.append_log(f"[发送] {text} -> {msg}")
                else:
                    print("[TCP] 未连接")
            else:
                print(f"[未配置] {text}")

    def handle_hc_cmd(self, text: str):
        self.append_log(f"[指令] {text}")
        try:
            cmd = json.loads(text)
        except Exception as e:
            print("[协议] 非法 JSON:", e)
            return

        tp = cmd.get("reqType")
        cam = int(cmd.get("camID", 0))

        if tp == "photo":
            self.do_capture_and_send(cam)
            ack = {
                "dsID": "www.hc-system.com.cam",
                "reqType": "photo",
                "camID": cam,
                "ret": 1,
            }
            self._send_json(ack)
        elif tp == "listModel":
            self._send_json(self.build_model_list_reply())
        elif tp == "changeModel":
            self.current_model = (cmd["name"], cmd["model"])
        # 根据协议可继续扩展其它分支

    # ------------------- UI 构建 -------------------
    
    def _init_controls(self):
        vbox = QtWidgets.QVBoxLayout(self.ctrl_panel)
        vbox.setAlignment(QtCore.Qt.AlignTop)

        # 摄像头组
        cam_group = QtWidgets.QGroupBox("摄像头")
        cam_layout = QtWidgets.QVBoxLayout(cam_group)

        self.cam_combo = QtWidgets.QComboBox()
        cam_layout.addWidget(self.cam_combo)

        refresh_btn = QtWidgets.QPushButton("刷新摄像头")
        refresh_btn.clicked.connect(self.start_scanning)
        cam_layout.addWidget(refresh_btn)

        vbox.addWidget(cam_group)

        self.cam_combo.addItem("扫描摄像头中...", -1)
        self.start_scanning()
        self.cam_combo.currentIndexChanged.connect(self.open_camera)

        # 颜色组
        for cfg in self.colors.values():
            self._add_color_group(vbox, cfg)

        shape_box = QtWidgets.QGroupBox("形状")
        shape_layout = QtWidgets.QVBoxLayout(shape_box)

        self.chk_circle = QtWidgets.QCheckBox("圆形");   self.chk_circle.setChecked(True)
        self.chk_tri    = QtWidgets.QCheckBox("等边三角形"); self.chk_tri.setChecked(False)
        self.chk_rect   = QtWidgets.QCheckBox("正方形"); self.chk_rect.setChecked(False)

        shape_layout.addWidget(self.chk_circle)
        shape_layout.addWidget(self.chk_tri)
        shape_layout.addWidget(self.chk_rect)
        shape_layout.addStretch(1)

        vbox.addWidget(shape_box)
                # ---- TCP 通讯设置 ----
        # ---- TCP 组（放在最下面）----
        tcp_group = QtWidgets.QGroupBox("通讯设置")
        tcp_layout = QtWidgets.QVBoxLayout(tcp_group)

        # IP & 端口输入（底部）
        tcp_layout.addWidget(QtWidgets.QLabel("服务器 IP:"))
        self.ip_input = QtWidgets.QLineEdit("127.0.0.1")
        tcp_layout.addWidget(self.ip_input)

        tcp_layout.addWidget(QtWidgets.QLabel("端口:"))
        self.port_input = QtWidgets.QLineEdit("6000")
        tcp_layout.addWidget(self.port_input)

        # 数据模板输入
        self.cmd_input = QtWidgets.QLineEdit('')
        tcp_layout.addWidget(QtWidgets.QLabel("数据发送"))
        tcp_layout.addWidget(self.cmd_input)

        # 连接和测试按钮
        btn_layout = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("连接TCP")
        self.send_btn = QtWidgets.QPushButton("测试发送")
        self.recognize_btn = QtWidgets.QPushButton("识别")
        # 自动发送勾选框
        self.auto_send_chk = QtWidgets.QCheckBox("识别成功后自动发送")
        tcp_layout.addWidget(self.auto_send_chk)

        btn_layout.addWidget(self.connect_btn)
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.recognize_btn)
        tcp_layout.addLayout(btn_layout)

        vbox.addWidget(tcp_group)

        # 初始化为空
        self.tcp_sender = None

        # 按钮事件
        self.connect_btn.clicked.connect(self.connect_tcp)
        self.send_btn.clicked.connect(self.test_send)
        self.recognize_btn.clicked.connect(
            lambda: self.do_capture_and_send(self.cam_combo.currentData() or 0)
        )
        vbox.addStretch(1)
    def append_log(self, msg: str):
        if msg.startswith("[发送]") or msg.startswith("[TCP 收到]") or msg.startswith("[测试发送]"):
            msg = f"<b>{msg}</b>"
        self.cmd_output.append(msg)

    def _add_color_group(self, parent_layout, cfg: ColorCfg):
        # ---------- 1. 可折叠 GroupBox ----------
        gbox = QtWidgets.QGroupBox(cfg.group_title)
        gbox.setCheckable(True)
        gbox.setChecked(False)          # 默认折叠
        gbox.setFlat(True)              # 视觉去边框（可选）
        parent_layout.addWidget(gbox)
        # ---------- 2. 内层容器（真正放控件） ----------
        container = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        gbox.setLayout(QtWidgets.QVBoxLayout())
        gbox.layout().addWidget(container)
        # 勾选展开 / 折叠
        gbox.toggled.connect(container.setVisible)
        container.setVisible(False)
        # ---------- 3. 6 个 HSV 滑块 ----------
        labels = ["H", "S", "V"]
        ranges = [(0, 179), (0, 255), (0, 255)]
        for idx, ch in enumerate(labels):
            mn, mx = ranges[idx]
            key_min = f"{cfg.name}_min{ch}"; key_max = f"{cfg.name}_max{ch}"
            init_min = int(cfg.lower[idx]); init_max = int(cfg.upper[idx])
            for key, init in [(key_min, init_min), (key_max, init_max)]:
                s = HSVSlider(key, mn, mx, init)
                s.valueChanged.connect(functools.partial(self._sync_cfg_from_sliders, cfg))
                vlay.addWidget(s)
                cfg.sliders[key] = s
        # ---------- 4. 掩膜按钮也放进去 ----------
        btn = QtWidgets.QPushButton(cfg.mask_button_title)
        btn.clicked.connect(lambda _=0, n=cfg.name: self.toggle_mask(n))
        vlay.addWidget(btn)
    # ------------------- 滑块同步 -------------------
    def _sync_cfg_from_sliders(self, cfg: ColorCfg):
        lh = cfg.sliders[f"{cfg.name}_minH"].value(); uh = cfg.sliders[f"{cfg.name}_maxH"].value()
        ls = cfg.sliders[f"{cfg.name}_minS"].value(); us = cfg.sliders[f"{cfg.name}_maxS"].value()
        lv = cfg.sliders[f"{cfg.name}_minV"].value(); uv = cfg.sliders[f"{cfg.name}_maxV"].value()
        cfg.lower[:] = [lh, ls, lv]
        cfg.upper[:] = [uh, us, uv]
    def connect_tcp(self):
        ip = self.ip_input.text().strip()
        port = int(self.port_input.text())
        try:
            self.tcp_sender = TcpSender(ip, port, on_recv=self.handle_tcp_msg)
            QtWidgets.QMessageBox.information(self, "成功", f"已连接 {ip}:{port}")
            self._save_server_config()
            self.start_heartbeat()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"连接失败: {e}")

    def load_cmd_map(self, path="tcp.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.cmd_map = data.get("cmd_map", {})

            svr  = data.get("server", {})
            host = svr.get("host", "127.0.0.1")
            port = svr.get("port", 6000)

            self.ip_input.setText(str(host))
            self.port_input.setText(str(port))
            print(f"[配置] 默认服务器: {host}:{port}")

        except Exception as e:
            print("[配置] 加载失败:", e)
            self.cmd_map = {}
            # 填回硬编码默认
            self.ip_input.setText("127.0.0.1")
            self.port_input.setText("6000")


    def test_send(self):
        if not self.tcp_sender:
            QtWidgets.QMessageBox.warning(self, "警告", "请先连接TCP服务器")
            return
        try:
            msg = self.cmd_input.text().format(color="测试色", shape="测试形", area=1234)
            self.tcp_sender.send_data(msg)
            self.append_log(f"[测试发送] {msg}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"发送失败: {e}")

    def handle_tcp_msg(self, text: str):
        self.tcp_msg_sig.emit(text)

    @QtCore.pyqtSlot(str)
    def _process_tcp_msg(self, text: str):
        """Callback for data received from the remote TCP server."""
        self.append_log(f"[TCP] 收到: {text}")
        try:
            cmd = json.loads(text)
        except Exception as e:
            print(f"[协议] 非法 JSON: {e}")
            return

        if cmd.get("reqType") == "photo":
            self.handle_hc_cmd(text)
    # ------------------- 摄像头 -------------------
    def open_camera(self):
        idx = self.cam_combo.currentData()
        if idx is None or idx == -1:         # 还在扫描或无设备
            return
        if self.capture:
            self.capture.release(); self.capture = None
        self.capture = cv2.VideoCapture(idx,cv2.CAP_MSMF)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_cnt = 0
        self.last_time = time.time()
        self.fps = 0.0
    # ------------------- 掩膜窗口 -------------------
    def toggle_mask(self, name: str):
        if name in self.mask_windows and self.mask_windows[name].isVisible():
            self.mask_windows[name].close(); return
        if name not in self.mask_windows:
            self.mask_windows[name] = MaskWindow(self.colors[name].mask_button_title, self)
            self.mask_windows[name].destroyed.connect(lambda _, n=name: self.mask_windows.pop(n, None))
        self.mask_windows[name].show(); self.mask_windows[name].raise_(); self.mask_windows[name].activateWindow()
    # ------------------- 主循环 -------------------
    def on_timer(self):
        if not self.capture or not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            return
        self.frame_cnt += 1
        if self.frame_cnt % self.FPS_CALC_INTERVAL == 0:
            now = time.time()
            self.fps = self.FPS_CALC_INTERVAL / (now - self.last_time)
            self.last_time = now
        # --------  形状检测 --------
        shapes_enabled = { s for s, chk in [
            ("circle",   self.chk_circle),
            ("triangle", self.chk_tri),
            ("rect",     self.chk_rect)
        ] if chk.isChecked() }
        labels = detect_shapes(frame, list(self.colors.values()), shapes_enabled)
        # --- 掩膜窗口更新 ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for cfg in self.colors.values():
            if cfg.sliders:
                mask = cv2.inRange(hsv, cfg.lower, cfg.upper)
                if cfg.name in self.mask_windows and self.mask_windows[cfg.name].isVisible():
                    try:
                        self.mask_windows[cfg.name].update_mask(mask)
                    except Exception as e:
                        print(f"[掩膜更新失败] {cfg.name}: {e}")

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, channels * w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qimg)
        # --------  用 QPainter 叠中文 --------
        painter = QtGui.QPainter(pix)
        painter.setFont(QtGui.QFont("微软雅黑", 16, QtGui.QFont.Bold))
        for text, (tx, ty), qcol in labels:
            painter.setPen(qcol)
            painter.drawText(tx, ty, text)
                        # === 根据配置和勾选框决定是否发送 ===
            key = text   # text 形如 "红色-圆形"
            if self.auto_send_chk.isChecked() and self.tcp_sender:
                if key in self.cmd_map:
                    msg = self.cmd_map[key]
                    self.tcp_sender.send_data(msg)
                    self.append_log(f"[发送] {key} -> {msg}")
                else:
                    print(f"[未配置] {key}")
        painter.end()
        self.video_lbl.setPixmap(pix.scaled(self.video_lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def closeEvent(self, e):
        if self.capture:
            self.capture.release(); self.capture = None
        if self.svr:
            try:
                self.svr.shutdown()
            except Exception:
                pass
        if self.tcp_sender:
            self.tcp_sender.close()
            self.tcp_sender = None
        self._running = False
        if getattr(self, "_hb_thread", None):
            self._hb_thread.join(timeout=0)
            self._hb_thread = None
        super().closeEvent(e)

def main():
    if cv2 is None or np is None or QtCore is None:
        print("Required dependencies are missing: OpenCV, NumPy or PyQt5.")
        return
    colors = load_colors("colors.json")
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("Camera.ico")))
    win = MainWindow(colors)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()