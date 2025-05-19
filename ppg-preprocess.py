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
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




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
            
            
            
class PPGPreprocessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG Video Preprocessor")
        
        # Variables
        self.filename = tk.StringVar()
        self.rotation = tk.StringVar(value="0")
        self.x_shift = tk.StringVar(value="0")
        self.y_shift = tk.StringVar(value="0")
        self.start_frame = tk.StringVar(value="2")
        self.proto = tk.StringVar(value='1')
        
        # Add trace to variables for real-time updates
        self.rotation.trace_add("write", self.on_value_change)
        self.x_shift.trace_add("write", self.on_value_change)
        self.y_shift.trace_add("write", self.on_value_change)
        self.start_frame.trace_add("write", self.on_value_change)
        self.proto.trace_add("write", self.on_value_change)
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        ttk.Label(main_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.filename, width=50, state='readonly').grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Alignment Parameters", padding="5")
        params_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Create sliders and entries for parameters
        # Rotation
        ttk.Label(params_frame, text="Rotation:").grid(row=0, column=0, sticky=tk.W)
        rotation_frame = ttk.Frame(params_frame)
        rotation_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Scale(rotation_frame, from_=-180, to=180, orient=tk.HORIZONTAL, 
                  variable=self.rotation, command=self.on_slider_change).grid(row=0, column=0, padx=5)
        ttk.Entry(rotation_frame, textvariable=self.rotation, width=5).grid(row=0, column=1, padx=5)
        
        # X Shift
        ttk.Label(params_frame, text="X Shift:").grid(row=1, column=0, sticky=tk.W)
        x_frame = ttk.Frame(params_frame)
        x_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Scale(x_frame, from_=-500, to=500, orient=tk.HORIZONTAL, 
                 variable=self.x_shift, command=self.on_slider_change).grid(row=0, column=0, padx=5)
        ttk.Entry(x_frame, textvariable=self.x_shift, width=5).grid(row=0, column=1, padx=5)
        
        # Y Shift
        ttk.Label(params_frame, text="Y Shift:").grid(row=2, column=0, sticky=tk.W)
        y_frame = ttk.Frame(params_frame)
        y_frame.grid(row=2, column=1, sticky=(tk.W, tk.E))
        ttk.Scale(y_frame, from_=-500, to=500, orient=tk.HORIZONTAL, 
                 variable=self.y_shift, command=self.on_slider_change).grid(row=0, column=0, padx=5)
        ttk.Entry(y_frame, textvariable=self.y_shift, width=5).grid(row=0, column=1, padx=5)
        
        # Frame Selection
        ttk.Label(params_frame, text="Frame:").grid(row=3, column=0, sticky=tk.W)
        frame_frame = ttk.Frame(params_frame)
        frame_frame.grid(row=3, column=1, sticky=(tk.W, tk.E))
        self.frame_scale = ttk.Scale(frame_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                 variable=self.start_frame, command=self.on_slider_change)
        self.frame_scale.grid(row=0, column=0, padx=5)
        ttk.Entry(frame_frame, textvariable=self.start_frame, width=5).grid(row=0, column=1, padx=5)
        
        # Proto Selection
        ttk.Label(params_frame, text="Proto:").grid(row=4, column=0, sticky=tk.W)
        proto_combo = ttk.Combobox(params_frame, textvariable=self.proto, 
                                 values=['1', '2a', '2b', '3a', '3b', '4a', '4b'],
                                 state='readonly', width=5)
        proto_combo.grid(row=4, column=1, padx=5, sticky=tk.W)
        proto_combo.bind('<<ComboboxSelected>>', self.on_value_change)
        
        # Process Button
        ttk.Button(main_frame, text="Process Video", command=self.process_video).grid(row=2, column=0, columnspan=3, pady=10)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Select a video file to begin")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=4, column=0, columnspan=3)
        
    def get_int_value(self, var):
        try:
            return int(float(var.get()))
        except ValueError:
            return 0
            
    def on_slider_change(self, *args):
        self.update_preview()
        
    def on_value_change(self, *args):
        self.root.after(100, self.update_preview)  # Delay update to prevent too frequent refreshes
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if filename:
            try:
                cap = cv2.VideoCapture(filename)
                if not cap.isOpened():
                    self.status_var.set("Error: Could not open video file")
                    return
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                self.filename.set(filename)
                self.frame_scale.configure(to=frame_count)
                self.status_var.set(f"File loaded. Total frames: {frame_count}")
                self.update_preview()
            except Exception as e:
                self.status_var.set(f"Error reading video: {str(e)}")
            
    def update_preview(self):
        if not self.filename.get():
            return
            
        try:
            # Clear the previous plot
            self.ax.clear()
            
            # Get the frame and apply transformations
            cap = cv2.VideoCapture(self.filename.get())
            if not cap.isOpened():
                self.status_var.set("Error: Could not open video file")
                return
                
            # Set frame position
            frame_pos = self.get_int_value(self.start_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos - 1)  # -1 because frame numbers start at 0
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                self.status_var.set(f"Error: Could not read frame {frame_pos}")
                return
                
            # Release video capture
            cap.release()
            
            # Apply rotation and shift
            frame = ndimage.rotate(frame, self.get_int_value(self.rotation))
            frame = ndimage.shift(frame, (self.get_int_value(self.x_shift), 
                                        self.get_int_value(self.y_shift), 0))
            
            # Display frame
            self.ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Draw grid overlay
            self.ax.plot([0,2216//2], [1200//2,1200//2], 'b--')
            self.ax.plot([1200//2,1200//2], [0,2216//2], 'b--')
            
            # Get grid points based on proto
            rr, ll = get_frame(self.proto.get())
            
            # Draw rectangles
            for y_offset in range(10):
                for x_offset in range(10):
                    # Right side
                    ya = rr[x_offset][y_offset]['ya']
                    yb = rr[x_offset][y_offset]['yb']
                    xa = rr[x_offset][y_offset]['xa']
                    xb = rr[x_offset][y_offset]['xb']
                    self.ax.plot([xa,xb,xb,xa,xa], [ya,ya,yb,yb,ya], 'r')
                    
                    # Left side
                    ya = ll[x_offset][y_offset]['ya']
                    yb = ll[x_offset][y_offset]['yb']
                    xa = ll[x_offset][y_offset]['xa']
                    xb = ll[x_offset][y_offset]['xb']
                    self.ax.plot([xa,xb,xb,xa,xa], [ya,ya,yb,yb,ya], 'r')
            
            # Update the canvas
            self.canvas.draw()
            self.status_var.set(f"Previewing frame {frame_pos}")
            
        except Exception as e:
            self.status_var.set(f"Error updating preview: {str(e)}")
        
    def process_video(self):
        if not self.filename.get():
            self.status_var.set("Please select a video file first")
            return
            
        try:
            self.status_var.set("Processing video... This may take a while.")
            self.root.update()
            
            # Save parameters to JSON
            params = {
                "r": self.get_int_value(self.rotation),
                "x": self.get_int_value(self.x_shift),
                "y": self.get_int_value(self.y_shift),
                "p": self.proto.get(),
                "s": self.get_int_value(self.start_frame)
            }
            
            with open(f'{self.filename.get()}.json', "w") as outfile:
                json.dump(params, outfile, indent=4)
                
            # Process video and save PKL
            l, r = cal_frame(
                input_video_path=self.filename.get(),
                rotation=self.get_int_value(self.rotation),
                shift=(self.get_int_value(self.x_shift), 
                      self.get_int_value(self.y_shift), 0),
                proto=self.proto.get()
            )
            
            with open(f'{self.filename.get()}.pkl', 'wb') as f:
                pickle.dump({
                    'l': l,
                    'r': r,
                    'x': self.get_int_value(self.x_shift),
                    'y': self.get_int_value(self.y_shift),
                    'rotation': self.get_int_value(self.rotation)
                }, f)
            
            self.status_var.set("Processing complete! Files saved successfully.")
        except Exception as e:
            self.status_var.set(f"Error during processing: {str(e)}")

if __name__ == '__main__':
    root = tk.Tk()
    app = PPGPreprocessGUI(root)
    root.mainloop()

