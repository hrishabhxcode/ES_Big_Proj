import os
import sys
import time
import joblib
import serial
import threading
import warnings
import winsound  # Native Windows OS Audio Pipeline
import numpy as np
import pandas as pd
import pyqtgraph as pg
from scipy.io import loadmat
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QTextEdit, QPushButton, QComboBox)
from PyQt6.QtCore import QTimer, Qt

warnings.filterwarnings("ignore")

# ============================================================
# ADVANCED AI BATTERY HEALTH ANALYTICS SYSTEM
# EMBEDDED SYSTEM PROJECT — DEVELOPMENT BY HRISHABH
# SERIAL VERSION FOR PROTEUS + COMPIM + ATMEGA328P
# ============================================================

# ===================== NASA .mat DATASET LOADER =====================
def load_nasa_mat_dataset(mat_path):
    mat = loadmat(mat_path)
    battery_key = [k for k in mat.keys() if k.startswith('B')][0]
    cycles = mat[battery_key][0, 0]['cycle'][0]
    records = []
    
    for cycle in cycles:
        if 'type' in cycle.dtype.fields and cycle['type'][0] == 'discharge':
            data = cycle['data'][0, 0]
            voltage = data[0].mean() if len(data[0]) else 0
            current = data[1].mean() if len(data[1]) else 0
            temperature = data[2].mean() if len(data[2]) else 0
            
            resistance = 0.05  
            power = voltage * current
            efficiency = voltage / 4.2 if voltage else 0
            cycles_num = int(cycle['index'][0, 0][0, 0]) if 'index' in cycle.dtype.fields else 0
            
            # Print explicit extraction data steps to terminal during execution setup
            print(f"   -> Processing Discharge Cycle Index: {cycles_num} | Voltage: {voltage:.3f}V | Temp: {temperature:.1f}°C")
            
            capacity = cycle['capacity'][0, 0][0, 0] if 'capacity' in cycle.dtype.fields else 2000 - cycles_num * 1.1
            soc = max(0, 100 - cycles_num * 0.05)
            
            records.append({
                'voltage': voltage, 'current': current, 'temperature': temperature,
                'resistance': resistance, 'power': power, 'efficiency': efficiency,
                'capacity': capacity, 'soc': soc, 'cycles': cycles_num
            })
            
    df = pd.DataFrame(records)
    df['soh'] = 100 - (df['cycles'] * 0.07) - (0.00004 * df['cycles']**2)
    df['soh'] = np.clip(df['soh'], 50, 100)
    df['rul'] = np.maximum(0, ((df['soh'] - 80) / 0.07))
    df['health_index'] = df['soh'] * 0.6 + df['soc'] * 0.4
    return df

# ===================== DATA VALIDATION/CONVERSION =====================
def validate_and_convert_live_data(raw_data):
    expected = {
        "voltage": float, "current": float, "temperature": float, "resistance": float,
        "power": float, "efficiency": float, "capacity": float, "soc": float, "cycles": int
    }
    converted = {}
    for key, typ in expected.items():
        val = raw_data.get(key, 0)
        try:
            if key == "resistance" and val > 1:
                val = val / 1000.0
            if key == "efficiency" and val > 1.5:
                val = val / 100.0
            converted[key] = typ(val)
        except Exception:
            converted[key] = 0 if typ is not str else ""
    return converted

# ============================================================
# NASA .mat DATASET INITIALIZATION
# ============================================================
NASA_FOLDER = "5"  
if not os.path.exists(NASA_FOLDER):
    os.makedirs(NASA_FOLDER, exist_ok=True)

mat_files = [f for f in os.listdir(NASA_FOLDER) if f.endswith('.mat')]
all_dfs = []

print(f"\n[*] STARTING MODEL SETUP FROM DATASET FOLDER: '{NASA_FOLDER}'")
for fname in mat_files:
    fpath = os.path.join(NASA_FOLDER, fname)
    print(f"[*] Parsing target repository matrix file: {fpath}")
    try:
        df_part = load_nasa_mat_dataset(fpath)
        df_part['source_file'] = fname
        all_dfs.append(df_part)
    except Exception as e:
        print(f"[!] Target skip alert for {fname}: {e}")
        pass

if all_dfs:
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"[✓] SUCCESS: Loaded {len(df)} total operational cycle profiles.\n")
else:
    print("[!] No NASA dataset files found in folder 5. Synthesizing offline fallback generation data matrix...")
    dummy_records = []
    for c in range(1, 200):
        v = 3.7 + np.random.normal(0, 0.1)
        i = 1.5 + np.random.normal(0, 0.05)
        t = 24.0 + c * 0.05
        soh = max(50.0, 100.0 - (c * 0.07) - (0.00004 * c**2))
        soc = max(0.0, 100.0 - c * 0.05)
        dummy_records.append({
            'voltage': v, 'current': i, 'temperature': t, 'resistance': 0.05,
            'power': v * i, 'efficiency': v / 4.2, 'capacity': 2000 - c * 1.1,
            'soc': soc, 'cycles': c, 'soh': soh, 'rul': max(0.0, (soh - 80) / 0.07),
            'health_index': soh * 0.6 + soc * 0.4
        })
    df = pd.DataFrame(dummy_records)

FEATURES = ["voltage", "current", "temperature", "resistance", "power", "efficiency", "capacity", "soc", "cycles"]
X = df[FEATURES]
y_soh = df["soh"]
y_rul = df["rul"]

# ============================================================
# TRAIN OR LOAD MODELS
# ============================================================
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

soh_model_paths = {name: os.path.join(MODELS_DIR, f"soh_{name.replace(' ', '_').lower()}.joblib") for name in [
    "Random Forest", "Gradient Boosting", "Decision Tree", "Linear Regression", "KNN", "SVR"]}
rul_model_path = os.path.join(MODELS_DIR, "rul_random_forest.joblib")

def get_models():
    models_soh = {
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    }
    rf_rul = RandomForestRegressor(n_estimators=100)
    return models_soh, rf_rul

def save_models(models_soh, rf_rul):
    for name, model in models_soh.items():
        joblib.dump(model, soh_model_paths[name])
    joblib.dump(rf_rul, rul_model_path)

def load_models():
    models_soh, rf_rul = get_models()
    all_exist = all(os.path.exists(p) for p in soh_model_paths.values()) and os.path.exists(rul_model_path)
    if all_exist:
        for name in models_soh:
            models_soh[name] = joblib.load(soh_model_paths[name])
        rf_rul = joblib.load(rul_model_path)
        return models_soh, rf_rul, True
    return models_soh, rf_rul, False

models_soh, rf_rul, loaded = load_models()
if not loaded:
    print("[*] INITIALIZING SCIKIT-LEARN MACHINE LEARNING TRAINING REGIMEN...")
    X_train, X_test, y_soh_train, y_soh_test = train_test_split(X, y_soh, test_size=0.2, random_state=42)
    X_rul_train, X_rul_test, y_rul_train, y_rul_test = train_test_split(X, y_rul, test_size=0.2, random_state=42)
    for name, model in models_soh.items():
        print(f" -> Fitting algorithmic hyperparameters for: [{name}]")
        model.fit(X_train, y_soh_train)
    print(" -> Finalizing multi-tree configuration for RUL Random Forest regression layer...")
    rf_rul.fit(X_rul_train, y_rul_train)
    save_models(models_soh, rf_rul)
    print("[✓] TRAINING PHASE COMPLETE: Serialized binaries written to storage workspace directory.\n")
else:
    print("[✓] DISK VERIFICATION: Core regression components loaded from serialized memory buffers.\n")

# ============================================================
# LIVE GLOBAL STORAGE
# ============================================================
live_data = {
    "voltage": 0, "current": 0, "temperature": 0, "resistance": 0,
    "power": 0, "efficiency": 0, "capacity": 0, "soc": 0, "cycles": 0,
    "soh": 100, "rul": 0, "health_index": 0, "status": "WAITING", "predictions": {},
    "diff_text": "No streaming hardware telemetry detected yet."
}

history_time = []
history_soh = []
history_cycles = []
history_temp = []
history_rul = []
history_volt = []

thermal_stress_accumulator = 0.0
last_time_stamp = None
last_audio_status = "WAITING" 

is_muted = False

# ============================================================
# AUDIO ANNOUNCEMENT & CONTINUOUS ALARM LOOP MODULE
# ============================================================
def play_health_alert_sound(status_string):
    global is_muted
    if is_muted:
        return
    try:
        if status_string == "GOOD":
            winsound.Beep(880, 150)
            winsound.Beep(1200, 250)
        elif status_string == "WARNING":
            winsound.Beep(600, 300)
            time.sleep(0.05)
            winsound.Beep(600, 300)
    except Exception as e:
        print(f"[!] Audio engine pipeline error: {e}")

def continuous_critical_alarm_server():
    global is_muted
    while True:
        try:
            if live_data["status"] == "REPLACE BATTERY" and not is_muted:
                winsound.Beep(350, 500)
                time.sleep(0.5)  
            else:
                time.sleep(1.0)  
        except Exception:
            time.sleep(1.0)

threading.Thread(target=continuous_critical_alarm_server, daemon=True).start()

# ============================================================
# SERIAL COMMUNICATION PROCESSOR
# ============================================================
def serial_server():
    global thermal_stress_accumulator, last_time_stamp, last_audio_status
    print("[*] Accessing Windows Hardware Communications Layer: Querying COM32...")
    try:
        ser = serial.Serial(port='COM32', baudrate=9600, timeout=1)
        time.sleep(2)
        print("[✓] SUCCESS: COM32 is connected and streaming data.")
    except Exception as e:
        print(f"[!] ERROR: Failed to establish channel link over COM32 ({e}). Launching fallback demo tracker mode...")
        return

    while True:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                # Direct streaming output logic onto terminal window console
                print(f"[STREAMING TELEMETRY] -> Raw Line: {line}")
                
                parts = {}
                for item in line.split(","):
                    if ":" in item:
                        try:
                            k, v = item.split(":")
                            parts[k.strip()] = v.strip()
                        except:
                            pass
                            
                voltage = float(parts.get("V", 0))
                current = float(parts.get("I", 0))
                temperature = float(parts.get("T", 0))
                resistance = float(parts.get("R", 0))
                cycles = float(parts.get("C", 0))
                
                power = voltage * current
                efficiency = voltage / 4.2 if voltage else 0
                capacity = 2000 - cycles * 1.1
                soc = max(0, 100 - cycles * 0.05)
                previous_soh = 100
                soh = (100 - (cycles * 0.07) - ((temperature-25) **1.3 )*0.12)
                calculated_soh = (100-(cycles * 0.07) - ((temperature-25) **1.3 )*0.12)
                SOH = min(previous_soh , calculated_soh)
                soh = np.clip(soh, 50, 100)
                temperature_factor = max(0.5, 1-((temperature - 25) * 0.015))
                rul = int(((soh-50)*5)*temperature_factor)
                rul = max(0, rul)
                health_index = soh * 0.6 + soc * 0.4
                
                current_time = time.time()
                if last_time_stamp is not None:
                    dt = current_time - last_time_stamp
                    if temperature > 25.0:
                        thermal_stress_accumulator += (temperature - 25.0) * dt
                last_time_stamp = current_time

                current_status = "GOOD" if soh >= 85 else ("WARNING" if soh >= 80 else "REPLACE BATTERY")

                if current_status != last_audio_status:
                    last_audio_status = current_status
                    if current_status != "REPLACE BATTERY":
                        threading.Thread(target=play_health_alert_sound, args=(current_status,), daemon=True).start()

                live_features = {
                    "voltage": voltage, "current": current, "temperature": temperature,
                    "resistance": resistance, "power": power, "efficiency": efficiency,
                    "capacity": capacity, "soc": soc, "cycles": cycles, "soh": soh,
                    "rul": rul, "health_index": health_index
                }

                live_features_valid = validate_and_convert_live_data(live_features)
                X_live = pd.DataFrame([live_features_valid])

                preds = {}
                for name, model in models_soh.items():
                    preds[name] = float(model.predict(X_live)[0])

                live_data.update({
                    "voltage": round(live_features['voltage'], 3),
                    "current": round(live_features['current'], 3),
                    "temperature": round(live_features['temperature'], 2),
                    "resistance": round(live_features['resistance'] * 1000, 1), 
                    "power": round(live_features['power'], 2),
                    "efficiency": round(live_features['efficiency'] * 100, 1),
                    "capacity": round(live_features['capacity'], 1),
                    "soc": round(live_features['soc'], 1),
                    "cycles": int(live_features['cycles']),
                    "soh": round(live_features['soh'], 1),
                    "rul": int(live_features['rul']),
                    "health_index": round(live_features['health_index'], 1),
                    "status": current_status,
                    "predictions": preds
                })

                try:
                    idx = (df['cycles'] - live_data['cycles']).abs().idxmin()
                    nasa_row = df.iloc[idx]
                    live_data["diff_text"] = (
                        f"Voltage Delta: {live_data['voltage'] - nasa_row['voltage']:+.3f} V\n"
                        f"Current Delta: {live_data['current'] - nasa_row['current']:+.3f} A\n"
                        f"Temperature Delta: {live_data['temperature'] - nasa_row['temperature']:+.2f} °C\n\n"
                        f"(Calculated against closest match cycle in NASA repository)"
                    )
                except Exception as e:
                    live_data["diff_text"] = f"[!] Reference compilation variance fault: {e}"

        except Exception:
            time.sleep(1)

# ============================================================
# STANDALONE DATA DIFFERENCE WINDOW
# ============================================================
class DataDifferenceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NASA Dataset Variance Workspace")
        self.setGeometry(200, 200, 600, 350)
        self.setStyleSheet("background-color: #151515; color: #FFF;")
        
        layout = QVBoxLayout(self)
        
        lbl_info = QLabel("<b>Telemetry Variance Analysis</b>")
        lbl_info.setStyleSheet("font-size: 18px; color: #FFD700;")
        layout.addWidget(lbl_info)
        
        self.txt_display = QTextEdit()
        self.txt_display.setReadOnly(True)
        self.txt_display.setStyleSheet("background-color: #222; color: #FFD700; font-family: Courier New; font-size: 16px; border-radius: 6px; padding: 10px;")
        layout.addWidget(self.txt_display)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_variance)
        self.timer.start(1000)
        
    def refresh_variance(self):
        self.txt_display.setText(live_data["diff_text"])

# ============================================================
# EXTENDED DIAGNOSTICS SUITE (ADVANCED MATH AND 2X2 QUAD GRAPHS)
# ============================================================
class DiagnosticsSuiteWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Analytical Engine & Predictive Degradation Framework")
        self.setGeometry(50, 80, 1680, 880)
        self.setStyleSheet("background-color: #111; color: white;")
        
        root_layout = QVBoxLayout(self)
        
        ctrl_bar = QHBoxLayout()
        lbl_ctrl = QLabel("<b>DSP Filter Mode:</b>")
        lbl_ctrl.setStyleSheet("font-size: 14px; color: #00FFAA;")
        
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["Raw Signal Tracking", "3-Point Moving Average", "5-Point Moving Average", "Exponential Smoothing (EMA)"])
        self.combo_filter.setStyleSheet("background-color: #222; color: white; padding: 5px; font-size:14px; border-radius:4px;")
        
        self.lbl_metrics_summary = QLabel("Health Velocity: 0.00%/s | Peak Temp: --.-°C")
        self.lbl_metrics_summary.setStyleSheet("font-size: 14px; color: #FFF; font-weight: bold; margin-left: 20px;")
        
        ctrl_bar.addWidget(lbl_ctrl)
        ctrl_bar.addWidget(self.combo_filter)
        ctrl_bar.addWidget(self.lbl_metrics_summary)
        ctrl_bar.addStretch()
        root_layout.addLayout(ctrl_bar)
        
        grid_graphs = QGridLayout()
        
        self.plot_time = pg.PlotWidget()
        self.plot_time.setBackground("#111")
        self.plot_time.showGrid(x=True, y=True)
        self.plot_time.setTitle("SOH vs Time (Samples)", color="w", size="11pt")
        self.plot_time.setLabel("left", "SOH %")
        self.plot_time.setLabel("bottom", "Runtime Index")
        
        self.plot_cycle = pg.PlotWidget()
        self.plot_cycle.setBackground("#111")
        self.plot_cycle.showGrid(x=True, y=True)
        self.plot_cycle.setTitle("SOH vs Cycle Progression", color="w", size="11pt")
        self.plot_cycle.setLabel("left", "SOH %")
        self.plot_cycle.setLabel("bottom", "Cycles Count")

        self.plot_rul_temp = pg.PlotWidget()
        self.plot_rul_temp.setBackground("#111")
        self.plot_rul_temp.showGrid(x=True, y=True)
        self.plot_rul_temp.setTitle("RUL vs Temperature Profile", color="w", size="11pt")
        self.plot_rul_temp.setLabel("left", "RUL (Cycles)")
        self.plot_rul_temp.setLabel("bottom", "Temperature (°C)")

        self.plot_volt_temp = pg.PlotWidget()
        self.plot_volt_temp.setBackground("#111")
        self.plot_volt_temp.showGrid(x=True, y=True)
        self.plot_volt_temp.setTitle("Thermal Safety Mapping (Voltage vs Temp)", color="w", size="11pt")
        self.plot_volt_temp.setLabel("left", "Voltage (V)")
        self.plot_volt_temp.setLabel("bottom", "Temperature (°C)")
        
        grid_graphs.addWidget(self.plot_time, 0, 0)
        grid_graphs.addWidget(self.plot_cycle, 0, 1)
        grid_graphs.addWidget(self.plot_rul_temp, 1, 0)
        grid_graphs.addWidget(self.plot_volt_temp, 1, 1)
        root_layout.addLayout(grid_graphs)
        
        self.lbl_stats_title = QLabel("<b>Statistical Descriptive Summary & Advanced Battery Health Matrix</b>")
        self.lbl_stats_title.setStyleSheet("color: #00FFAA; font-size:14px; margin-top:5px;")
        root_layout.addWidget(self.lbl_stats_title)
        
        self.txt_stats_matrix = QTextEdit()
        self.txt_stats_matrix.setReadOnly(True)
        self.txt_stats_matrix.setMaximumHeight(110)
        self.txt_stats_matrix.setStyleSheet("background-color: #1a1a1a; color: #00FFAA; font-family: Consolas; font-size:14px; border-radius:5px;")
        root_layout.addWidget(self.txt_stats_matrix)
        
        self.curve_time = self.plot_time.plot(pen=pg.mkPen(color='#00FFCC', width=2), symbol='o', symbolSize=5, symbolBrush=('#00FFFF'))
        self.curve_cycle = self.plot_cycle.plot(pen=pg.mkPen(color='#FF5555', width=2), symbol='o', symbolSize=5, symbolBrush=('#FF8888'))
        self.curve_rul_temp = self.plot_rul_temp.plot(pen=pg.mkPen(color='#FFAA00', width=2), symbol='o', symbolSize=5, symbolBrush=('#FFCC44'))
        self.curve_volt_temp = self.plot_volt_temp.plot(pen=None, symbol='o', symbolSize=6, symbolBrush=('#FF00FF')) 

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.process_advanced_analytics)
        self.update_timer.start(1000)

    def process_advanced_analytics(self):
        if not history_soh:
            return
            
        soh_data = np.array(history_soh)
        filter_mode = self.combo_filter.currentText()
        
        if "3-Point" in filter_mode and len(soh_data) >= 3:
            soh_data = np.convolve(soh_data, np.ones(3)/3, mode='valid')
            display_time = history_time[len(history_time)-len(soh_data):]
        elif "5-Point" in filter_mode and len(soh_data) >= 5:
            soh_data = np.convolve(soh_data, np.ones(5)/5, mode='valid')
            display_time = history_time[len(history_time)-len(soh_data):]
        elif "Exponential" in filter_mode and len(soh_data) > 1:
            df_ema = pd.Series(soh_data).ewm(alpha=0.3, adjust=False).mean()
            soh_data = df_ema.to_numpy()
            display_time = history_time
        else:
            display_time = history_time

        self.curve_time.setData(display_time, list(soh_data))
        self.curve_cycle.setData(history_cycles, history_soh)
        self.curve_rul_temp.setData(history_temp, history_rul)
        self.curve_volt_temp.setData(history_temp, history_volt)
        
        degradation_rate = 0.0
        if len(history_soh) > 1:
            degradation_rate = (history_soh[0] - history_soh[-1]) / len(history_soh)
            
        peak_temp = max(history_temp) if history_temp else 0.0
        self.lbl_metrics_summary.setText(
            f"Health Loss Velocity Rate: {degradation_rate:.4f}%/sec  |  "
            f"Peak Operating Temp: {peak_temp:.1f} °C"
        )
        
        soh_mean = np.mean(history_soh)
        soh_std = np.std(history_soh)
        soh_variance = np.var(history_soh)
        
        delta_resistance = live_data["resistance"] - (history_temp[0] if history_temp else 0.0)
        
        stats_string = (
            f" [SOH MEAN]: {soh_mean:.2f}%  |  [SOH VARIANCE]: {soh_variance:.4f}  |  [SOH STD DEV]: {soh_std:.4f}\n"
            f" [INTERNAL RESISTANCE DELTA (ΔR)]: {delta_resistance:+.1f} mΩ (Total growth since initialization)\n"
            f" [CUMULATIVE THERMAL STRESS INDEX]: {thermal_stress_accumulator:.2f} °C·sec (Integrated stress >25°C boundary)\n"
            f" [HEALTH DATA SAMPLES POOL COUNT]: {len(history_soh)} Telemetry frames logged"
        )
        self.txt_stats_matrix.setText(stats_string)

# ============================================================
# MAIN APPLICATION MANAGEMENT FRAMEWORK
# ============================================================
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Battery Health Dashboard Suite")
        self.setGeometry(50, 50, 1700, 950)
        self.setStyleSheet("background:#111; color:white; font-size:16px;")

        self.window_diagnostics = None
        self.window_variance = None

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        lbl_credits = QLabel("Made by Hrishabh (Embedded System Project)")
        lbl_credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_credits.setStyleSheet("font-size: 16px; color: #00FFAA; font-weight: bold; font-family: Consolas; letter-spacing: 1px; padding: 2px; background: #1a1a1a; border-radius:4px;")
        layout.addWidget(lbl_credits)

        title = QLabel("AI BATTERY HEALTH ANALYTICS SYSTEM")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:28px; font-weight:bold; margin-top:5px; margin-bottom:10px;")
        layout.addWidget(title)

        button_layout = QHBoxLayout()
        
        self.btn_graph_output = QPushButton("Open Analytical Diagnostics Suite")
        self.btn_graph_output.setStyleSheet("""
            QPushButton { background-color: #00AAFF; color: white; font-weight: bold; font-size: 16px; padding: 10px 22px; border-radius: 6px; }
            QPushButton:hover { background-color: #0088CC; }
        """)
        self.btn_graph_output.clicked.connect(self.open_diagnostics_suite)
        button_layout.addWidget(self.btn_graph_output)
        
        self.btn_variance_output = QPushButton("Compare NASA Dataset Variance")
        self.btn_variance_output.setStyleSheet("""
            QPushButton { background-color: #FF9900; color: white; font-weight: bold; font-size: 16px; padding: 10px 22px; border-radius: 6px; margin-left: 12px; }
            QPushButton:hover { background-color: #CC7700; }
        """)
        self.btn_variance_output.clicked.connect(self.open_variance_window)
        button_layout.addWidget(self.btn_variance_output)

        self.btn_mute_audio = QPushButton("Mute Alarm")
        self.btn_mute_audio.setStyleSheet("""
            QPushButton { background-color: #E63946; color: white; font-weight: bold; font-size: 16px; padding: 10px 22px; border-radius: 6px; margin-left: 12px; }
            QPushButton:hover { background-color: #C1121F; }
        """)
        self.btn_mute_audio.clicked.connect(self.toggle_mute_state)
        button_layout.addWidget(self.btn_mute_audio)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)

        grid = QGridLayout()
        self.labels = {}
        params = [
            "Voltage", "Current", "Temperature", "Resistance", "Power",
            "Efficiency", "Capacity", "SOC", "Cycles", "SOH", "RUL",
            "Health Index", "Status"
        ]

        row = 0
        col = 0
        for p in params:
            lbl = QLabel(f"{p}: --")
            lbl.setStyleSheet("background:#222; padding:15px; border-radius:10px; font-size:18px;")
            self.labels[p] = lbl
            grid.addWidget(lbl, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        layout.addLayout(grid)

        self.graph = pg.PlotWidget()
        self.graph.setBackground("#111")
        self.graph.showGrid(x=True, y=True)
        self.graph.setTitle("SOH Degradation Live Curve")
        self.graph.setLabel("left", "SOH %")
        self.graph.setLabel("bottom", "Runtime Frame Index")

        self.curve = self.graph.plot(pen=pg.mkPen(color='#00FF00', width=2))
        layout.addWidget(self.graph)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui_core)
        self.timer.start(1000)

    def toggle_mute_state(self):
        global is_muted
        is_muted = not is_muted
        if is_muted:
            self.btn_mute_audio.setText("Unmute Alarm")
            self.btn_mute_audio.setStyleSheet("""
                QPushButton { background-color: #457B9D; color: white; font-weight: bold; font-size: 16px; padding: 10px 22px; border-radius: 6px; margin-left: 12px; }
                QPushButton:hover { background-color: #1D3557; }
            """)
        else:
            self.btn_mute_audio.setText("Mute Alarm")
            self.btn_mute_audio.setStyleSheet("""
                QPushButton { background-color: #E63946; color: white; font-weight: bold; font-size: 16px; padding: 10px 22px; border-radius: 6px; margin-left: 12px; }
                QPushButton:hover { background-color: #C1121F; }
            """)

    def open_diagnostics_suite(self):
        if self.window_diagnostics is None:
            self.window_diagnostics = DiagnosticsSuiteWindow()
        self.window_diagnostics.show()
        self.window_diagnostics.activateWindow()

    def open_variance_window(self):
        if self.window_variance is None:
            self.window_variance = DataDifferenceWindow()
        self.window_variance.show()
        self.window_variance.activateWindow()

    def update_ui_core(self):
        self.labels["Voltage"].setText(f"Voltage: {live_data['voltage']} V")
        self.labels["Current"].setText(f"Current: {live_data['current']} A")
        self.labels["Temperature"].setText(f"Temperature: {live_data['temperature']} °C")
        self.labels["Resistance"].setText(f"Resistance: {live_data['resistance']} mΩ")
        self.labels["Power"].setText(f"Power: {live_data['power']} W")
        self.labels["Efficiency"].setText(f"Efficiency: {live_data['efficiency']} %")
        self.labels["Capacity"].setText(f"Capacity: {live_data['capacity']} mAh")
        self.labels["SOC"].setText(f"SOC: {live_data['soc']} %")
        self.labels["Cycles"].setText(f"Cycles: {live_data['cycles']}")
        self.labels["SOH"].setText(f"SOH: {live_data['soh']} %")
        self.labels["RUL"].setText(f"RUL: {live_data['rul']} Cycles")
        self.labels["Health Index"].setText(f"Health Index: {live_data['health_index']}")
        self.labels["Status"].setText(f"Status: {live_data['status']}")

        global is_muted
        if live_data['status'] != "REPLACE BATTERY" and is_muted:
            self.toggle_mute_state()

        history_time.append(len(history_time))
        history_soh.append(live_data["soh"])
        history_cycles.append(live_data["cycles"])
        history_temp.append(live_data["temperature"])
        history_rul.append(live_data["rul"])
        history_volt.append(live_data["voltage"])

        self.curve.setData(history_time, history_soh)

# ============================================================
# MAIN APPLICATION LAUNCH EXECUTION
# ============================================================
if __name__ == '__main__':
    threading.Thread(target=serial_server, daemon=True).start()

    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())