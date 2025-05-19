import cv2
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import scipy.signal as signal
import concurrent.futures
import tqdm
import pickle
import argparse
import json
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton,
                            QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os




def plot_rec(x1,x2, y1,y2):
    plt.plot([y1,y2,y2,y1,y1],[x1,x1,x2,x2,x1], 'r')
    
def get_frame_default(BINING = 2):

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}
    WIN_SIZE = 50 // BINING
    
    rr = [[{} for j in range(10)] for i in range(10)]
    ll = [[{} for j in range(10)] for i in range(10)]

    for y_offset in range(10):
        for x_offset in range(10):
            ya = FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE
            yb = FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE
            xa = FRAME_BIAS['xa']//BINING + x_offset*WIN_SIZE
            xb = FRAME_BIAS['xb']//BINING + x_offset*WIN_SIZE

            rr[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}

            ya = FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE
            yb = FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE
            xa = FRAME_BIAS['xa2']//BINING - x_offset*WIN_SIZE
            xb = FRAME_BIAS['xb2']//BINING - x_offset*WIN_SIZE

            ll[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}
    return rr, ll


def get_frame_2a(BINING = 2):

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}
    WIN_SIZE = 50 // BINING
    
    rr = [[{} for j in range(10)] for i in range(10)]
    ll = [[{} for j in range(10)] for i in range(10)]

    for y_offset in range(10):
        for x_offset in range(10):
            xa = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            ya = np.round((FRAME_BIAS['xa']//BINING + x_offset*WIN_SIZE)/1.3).astype(int) + 120
            yb = np.round((FRAME_BIAS['xb']//BINING + x_offset*WIN_SIZE)/1.3).astype(int) + 120

            rr[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}

            xa = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            ya = np.round((FRAME_BIAS['xa2']//BINING - x_offset*WIN_SIZE)/1.3).astype(int) + 120
            yb = np.round((FRAME_BIAS['xb2']//BINING - x_offset*WIN_SIZE)/1.3).astype(int) + 120

            ll[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}
    return rr, ll

def get_frame_4b(BINING = 2):

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}
    WIN_SIZE = 50 // BINING
    
    rr = [[{} for j in range(10)] for i in range(10)]
    ll = [[{} for j in range(10)] for i in range(10)]

    for y_offset in range(10):
        for x_offset in range(10):
            xa = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            ya = np.round((FRAME_BIAS['xa']//BINING + x_offset*WIN_SIZE)/1.3).astype(int) + 120
            yb = np.round((FRAME_BIAS['xb']//BINING + x_offset*WIN_SIZE)/1.3).astype(int) + 120

            rr[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}

            xa = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            ya = np.round((FRAME_BIAS['xa2']//BINING - x_offset*WIN_SIZE)/1.3).astype(int) + 120
            yb = np.round((FRAME_BIAS['xb2']//BINING - x_offset*WIN_SIZE)/1.3).astype(int) + 120

            ll[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}
    return rr, ll

def get_frame_2b(BINING = 2):

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}
    WIN_SIZE = 50 // BINING
    
    rr = [[{} for j in range(10)] for i in range(10)]
    ll = [[{} for j in range(10)] for i in range(10)]

    for y_offset in range(10):
        for x_offset in range(10):
            xa = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            ya = np.round((FRAME_BIAS['xa']//BINING + x_offset*WIN_SIZE)/1.3).astype(int) + 120
            yb = np.round((FRAME_BIAS['xb']//BINING + x_offset*WIN_SIZE)/1.3).astype(int) + 120

            rr[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}

            xa = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            ya = np.round((FRAME_BIAS['xa2']//BINING - x_offset*WIN_SIZE)/1.3).astype(int) + 120
            yb = np.round((FRAME_BIAS['xb2']//BINING - x_offset*WIN_SIZE)/1.3).astype(int) + 120

            ll[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}
    return rr, ll


def get_frame_3b(BINING = 2):

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}
    WIN_SIZE = 50 // BINING
    
    rr = [[{} for j in range(10)] for i in range(10)]
    ll = [[{} for j in range(10)] for i in range(10)]

    for y_offset in range(10):
        for x_offset in range(10):
            ya = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            yb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xa = np.round((FRAME_BIAS['xa']//BINING + x_offset*WIN_SIZE)/0.9).astype(int) - 200
            xb = np.round((FRAME_BIAS['xb']//BINING + x_offset*WIN_SIZE)/0.9).astype(int) - 200

            rr[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}

            ya = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            yb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xa = np.round((FRAME_BIAS['xa2']//BINING - x_offset*WIN_SIZE)/0.9).astype(int) + 90
            xb = np.round((FRAME_BIAS['xb2']//BINING - x_offset*WIN_SIZE)/0.9).astype(int) + 90

            ll[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}
    return rr, ll

def get_frame_3a(BINING = 2):

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}
    WIN_SIZE = 50 // BINING
    
    rr = [[{} for j in range(10)] for i in range(10)]
    ll = [[{} for j in range(10)] for i in range(10)]

    for y_offset in range(10):
        for x_offset in range(10):
            ya = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            yb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xa = np.round((FRAME_BIAS['xa']//BINING + x_offset*WIN_SIZE)/0.9).astype(int) - 200
            xb = np.round((FRAME_BIAS['xb']//BINING + x_offset*WIN_SIZE)/0.9).astype(int) - 200

            rr[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}

            ya = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            yb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xa = np.round((FRAME_BIAS['xa2']//BINING - x_offset*WIN_SIZE)/0.9).astype(int) + 90
            xb = np.round((FRAME_BIAS['xb2']//BINING - x_offset*WIN_SIZE)/0.9).astype(int) + 90

            ll[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}
    return rr, ll

def get_frame_4a(BINING = 2):

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}
    WIN_SIZE = 50 // BINING
    
    rr = [[{} for j in range(10)] for i in range(10)]
    ll = [[{} for j in range(10)] for i in range(10)]

    for y_offset in range(10):
        for x_offset in range(10):
            ya = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            yb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xa = np.round((FRAME_BIAS['xa']//BINING + x_offset*WIN_SIZE)/0.9).astype(int) - 200
            xb = np.round((FRAME_BIAS['xb']//BINING + x_offset*WIN_SIZE)/0.9).astype(int) - 200

            rr[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}

            ya = (FRAME_BIAS['ya']//BINING + y_offset*WIN_SIZE)*3 - 1300
            yb = (FRAME_BIAS['yb']//BINING + y_offset*WIN_SIZE)*3 - 1300
            xa = np.round((FRAME_BIAS['xa2']//BINING - x_offset*WIN_SIZE)/0.9).astype(int) + 90
            xb = np.round((FRAME_BIAS['xb2']//BINING - x_offset*WIN_SIZE)/0.9).astype(int) + 90

            ll[x_offset][y_offset] = {'ya':ya, 'yb':yb, 'xa':xa, 'xb':xb}
    return rr, ll

def get_frame(proto = '1', BINING = 2):
    if (proto == '2a'):
        rr, ll = get_frame_2a(BINING);
    elif (proto == '2b'):
        rr, ll = get_frame_2b(BINING);
    elif (proto == '4b'):
        rr, ll = get_frame_4b(BINING);
    elif (proto == '4a'):
        rr, ll = get_frame_4b(BINING);
    elif (proto == '3b'):
        rr, ll = get_frame_3b(BINING);
    elif (proto == '3a'):
        rr, ll = get_frame_3a(BINING);
    else:
        rr,ll = get_frame_default(BINING);

    return rr,ll

def plot_frame(input_video_path  = '../Data52.avi', rotation = 1, shift = (-5, 100,0), BINING = 2, start_frame = 2, proto = 0):
    

    cap = cv2.VideoCapture(input_video_path)
    for i in tqdm.tqdm(range(start_frame)):
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

    WIN_SIZE = 50 // BINING

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}

    frame1 = ndimage.rotate(frame, rotation)
    frame1 = ndimage.shift(frame1, shift)
    plt.figure()
    plt.imshow(frame1)


    plt.plot([0,2216//BINING], [1200//BINING,1200//BINING], 'b--')
    plt.plot([1200//BINING,1200//BINING],[0,2216//BINING],  'b--')

    rr,ll = get_frame(proto)    

    for y_offset in range(10):
        for x_offset in range(10):

            ya = rr[x_offset][y_offset]['ya']
            yb = rr[x_offset][y_offset]['yb']
            xa = rr[x_offset][y_offset]['xa']
            xb = rr[x_offset][y_offset]['xb']
            
            plot_rec(ya,
                     yb,
                     xa,
                     xb)

            ya = ll[x_offset][y_offset]['ya']
            yb = ll[x_offset][y_offset]['yb']
            xa = ll[x_offset][y_offset]['xa']
            xb = ll[x_offset][y_offset]['xb']


            plot_rec(ya,
                     yb,
                     xa,
                     xb)



def cal_frame(input_video_path  = '../Data52.avi', rotation = 1, shift = (-5, 100,0), BINING = 4, proto = '0'):

    X_SAMPLE_RANGE = 10
    Y_SAMPLE_RANGE = 10
    SAMPLE_RANGE = X_SAMPLE_RANGE * Y_SAMPLE_RANGE
    SAMPLE_DIM = 25
    NUM_OF_FRAMES = 1000

    WIN_SIZE = 50 // BINING

    cap = cv2.VideoCapture(input_video_path)
    NUM_OF_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    FRAME_BIAS = {'ya':2*500, 'yb':2*525,'xa':2*725, 'xb':2*750, 'xa2':2*435, 'xb2':2*460}

    time_series_r = np.zeros((SAMPLE_RANGE, NUM_OF_FRAMES))
    time_series_l = np.zeros((SAMPLE_RANGE, NUM_OF_FRAMES))

    rr,ll = get_frame(proto)

    for index in tqdm.tqdm(range(NUM_OF_FRAMES)):
        ret, frame = cap.read()
        frame1 = ndimage.rotate(frame, rotation)
        frame1 = ndimage.shift(frame1, shift)


        for y_offset in range(Y_SAMPLE_RANGE):
            for x_offset in range(X_SAMPLE_RANGE):
                ya = rr[x_offset][y_offset]['ya']
                yb = rr[x_offset][y_offset]['yb']
                xa = rr[x_offset][y_offset]['xa']
                xb = rr[x_offset][y_offset]['xb']
                time_series_r[y_offset*X_SAMPLE_RANGE + x_offset ,index] = np.mean(frame1[ya:yb,xa:xb,:])


                ya = ll[x_offset][y_offset]['ya']
                yb = ll[x_offset][y_offset]['yb']
                xa = ll[x_offset][y_offset]['xa']
                xb = ll[x_offset][y_offset]['xb']
                time_series_l[y_offset*X_SAMPLE_RANGE + x_offset ,index] = np.mean(frame1[ya:yb,xa:xb,:])

    return time_series_l, time_series_r

def post_process(l_in):

    sampleRate = 25
    lpOrder = 8
    lpPassBand = .5

    lpFilterType='butter'
    lpPassBandRipple=1
    lpStopBandAttenuation=60.0

    hpPassBand=np.float32(0.25)
    hpOrder=2

    bl, al = signal.iirfilter(lpOrder,
                              lpPassBand / (sampleRate / 2),
                              btype='low',
                              ftype=lpFilterType,
                              rp=lpPassBandRipple,
                              rs=lpStopBandAttenuation)

    bh, ah = signal.iirfilter(hpOrder,
                              hpPassBand / (sampleRate / 2),
                              btype='high',
                              ftype=lpFilterType,
                              rp=lpPassBandRipple,
                              rs=lpStopBandAttenuation)
    
    
    plt.figure(figsize=(20,4))
    for j_index in range(10):
        for index in range(1):
            data = l_in[index + (10 * j_index)]

            data = signal.lfilter(bl, al, data)
            data = signal.lfilter(bh, ah, data)
            plt.plot(data[200:], label = f'{j_index}')
            plt.legend()
            
            
            

class GridAlignmentGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grid Alignment Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize video capture as None
        self.cap = None
        self.frame = None
        self.current_file = None

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Create controls panel
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_panel.setFixedWidth(300)
        layout.addWidget(controls_panel)

        # Add Load File button
        self.load_button = QPushButton("Load Video File")
        self.load_button.clicked.connect(self.load_file)
        controls_layout.addWidget(self.load_button)

        # Add current file label
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        controls_layout.addWidget(self.file_label)

        # Pattern selection dropdown
        pattern_label = QLabel("Pattern:")
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(['1', '2a', '2b', '3a', '3b', '4a', '4b'])
        self.pattern_combo.currentTextChanged.connect(self.update_plot)
        controls_layout.addWidget(pattern_label)
        controls_layout.addWidget(self.pattern_combo)

        # Rotation slider
        rotation_label = QLabel("Rotation:")
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(rotation_label)
        controls_layout.addWidget(self.rotation_slider)

        # X-shift slider
        x_shift_label = QLabel("X Shift:")
        self.x_shift_slider = QSlider(Qt.Orientation.Horizontal)
        self.x_shift_slider.setRange(-500, 500)
        self.x_shift_slider.setValue(0)
        self.x_shift_slider.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(x_shift_label)
        controls_layout.addWidget(self.x_shift_slider)

        # Y-shift slider
        y_shift_label = QLabel("Y Shift:")
        self.y_shift_slider = QSlider(Qt.Orientation.Horizontal)
        self.y_shift_slider.setRange(-500, 500)
        self.y_shift_slider.setValue(0)
        self.y_shift_slider.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(y_shift_label)
        controls_layout.addWidget(self.y_shift_slider)

        # Value labels
        self.rotation_value = QLabel("Rotation: 0°")
        self.x_shift_value = QLabel("X Shift: 0")
        self.y_shift_value = QLabel("Y Shift: 0")
        controls_layout.addWidget(self.rotation_value)
        controls_layout.addWidget(self.x_shift_value)
        controls_layout.addWidget(self.y_shift_value)

        # Save Settings button
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        self.save_button.setEnabled(False)  # Disable until file is loaded
        controls_layout.addWidget(self.save_button)

        # Add matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Store the current axes
        self.current_axes = None
        
        self.BINING = 2

    def load_file(self):
        # Open file dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.avi *.mp4 *.mov);;All Files (*.*)"
        )

        if file_path:
            # Release previous capture if it exists
            if self.cap is not None:
                self.cap.release()

            # Try to open the new file
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(
                    self,
                    "Error",
                    "Could not open video file. Please check if the file is valid."
                )
                self.cap = None
                return

            # Read first frame
            ret, self.frame = self.cap.read()
            if not ret:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Could not read frame from video file."
                )
                self.cap.release()
                self.cap = None
                return

            # Update UI
            self.current_file = file_path
            self.file_label.setText(f"Current file: {os.path.basename(file_path)}")
            self.save_button.setEnabled(True)

            # Try to load existing settings
            self.load_settings()

            # Update plot
            self.update_plot()

    def load_settings(self):
        """Try to load existing settings for the current file"""
        if self.current_file:
            json_path = f"{self.current_file}.json"
            if os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, 'r') as f:
                        settings = json.load(f)
                        
                    # Update UI with loaded settings
                    self.rotation_slider.setValue(settings.get('r', 0))
                    self.x_shift_slider.setValue(settings.get('x', 0))
                    self.y_shift_slider.setValue(settings.get('y', 0))
                    
                    pattern = settings.get('p', '1')
                    index = self.pattern_combo.findText(pattern)
                    if index >= 0:
                        self.pattern_combo.setCurrentIndex(index)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Could not load existing settings: {str(e)}"
                    )

    def save_settings(self):
        """Save current settings to a JSON file"""
        if self.current_file:
            import json
            settings = {
                "r": self.rotation_slider.value(),
                "x": self.x_shift_slider.value(),
                "y": self.y_shift_slider.value(),
                "p": self.pattern_combo.currentText(),
                "s": 2  # Default start frame
            }
            
            try:
                with open(f"{self.current_file}.json", "w") as f:
                    json.dump(settings, f, indent=4)
                QMessageBox.information(
                    self,
                    "Success",
                    "Settings saved successfully!"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Could not save settings: {str(e)}"
                )

    def plot_rec(self, x1, x2, y1, y2):
        """Plot a rectangle on the current axes"""
        if self.current_axes is not None:
            self.current_axes.plot([y1,y2,y2,y1,y1], [x1,x1,x2,x2,x1], 'r')

    def update_plot(self):
        if self.frame is None:
            return

        # Update value labels
        rotation = self.rotation_slider.value()
        x_shift = self.x_shift_slider.value()
        y_shift = self.y_shift_slider.value()
        
        self.rotation_value.setText(f"Rotation: {rotation}°")
        self.x_shift_value.setText(f"X Shift: {x_shift}")
        self.y_shift_value.setText(f"Y Shift: {y_shift}")

        # Clear the figure
        self.figure.clear()
        self.current_axes = self.figure.add_subplot(111)

        # Apply transformations
        frame1 = ndimage.rotate(self.frame, rotation)
        frame1 = ndimage.shift(frame1, (x_shift, y_shift, 0))

        # Display the image
        self.current_axes.imshow(frame1)

        # Draw reference lines
        self.current_axes.plot([0, 2216//self.BINING], [1200//self.BINING, 1200//self.BINING], 'b--')
        self.current_axes.plot([1200//self.BINING, 1200//self.BINING], [0, 2216//self.BINING], 'b--')

        # Get and plot grid pattern
        proto = self.pattern_combo.currentText()
        rr, ll = get_frame(proto, self.BINING)

        # Plot grid rectangles
        for y_offset in range(10):
            for x_offset in range(10):
                # Right side
                ya = rr[x_offset][y_offset]['ya']
                yb = rr[x_offset][y_offset]['yb']
                xa = rr[x_offset][y_offset]['xa']
                xb = rr[x_offset][y_offset]['xb']
                self.plot_rec(ya, yb, xa, xb)

                # Left side
                ya = ll[x_offset][y_offset]['ya']
                yb = ll[x_offset][y_offset]['yb']
                xa = ll[x_offset][y_offset]['xa']
                xb = ll[x_offset][y_offset]['xb']
                self.plot_rec(ya, yb, xa, xb)

        # Update the canvas
        self.canvas.draw()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = GridAlignmentGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

