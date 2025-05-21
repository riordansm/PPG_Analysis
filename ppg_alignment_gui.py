import cv2
import numpy as np
from scipy import ndimage
import json
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

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
    return get_frame_4b(BINING)

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

def get_frame(proto = '1', BINING = 2):
    if (proto == '2a'):
        rr, ll = get_frame_2a(BINING)
    elif (proto == '2b'):
        rr, ll = get_frame_2b(BINING)
    elif (proto == '4b'):
        rr, ll = get_frame_4b(BINING)
    elif (proto == '4a'):
        rr, ll = get_frame_4a(BINING)
    elif (proto == '3b'):
        rr, ll = get_frame_3b(BINING)
    elif (proto == '3a'):
        rr, ll = get_frame_3a(BINING)
    else:
        rr,ll = get_frame_default(BINING)
    return rr,ll

class PPGAlignmentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG Video Alignment")
        
        # Variables
        self.filename = tk.StringVar()
        self.rotation = tk.StringVar(value="0")
        self.x_shift = tk.StringVar(value="0")
        self.y_shift = tk.StringVar(value="0")
        self.start_frame = tk.StringVar(value="2")
        self.proto = tk.StringVar(value='1')
        
        # Preview scale factor (1/2 size for preview)
        self.preview_scale = 2
        
        # Cache for frames and transformations
        self.current_frame = None
        self.current_frame_number = None
        self.current_frame_full = None
        self.last_rotation = None
        self.last_x_shift = None
        self.last_y_shift = None
        self.last_proto = None
        self.update_pending = False
        
        # Add trace to variables for real-time updates
        for var in [self.rotation, self.x_shift, self.y_shift, self.proto]:
            var.trace_add("write", self.schedule_update)
        self.start_frame.trace_add("write", self.on_frame_change)
        
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
        
        # Rotation
        ttk.Label(params_frame, text="Rotation:").grid(row=0, column=0, sticky=tk.W)
        rotation_frame = ttk.Frame(params_frame)
        rotation_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Scale(rotation_frame, from_=-180, to=180, orient=tk.HORIZONTAL, 
                  variable=self.rotation).grid(row=0, column=0, padx=5)
        entry_frame = ttk.Frame(rotation_frame)
        entry_frame.grid(row=0, column=1)
        ttk.Button(entry_frame, text="-", width=2, 
                  command=lambda: self.increment_value(self.rotation, -1)).grid(row=0, column=0, padx=1)
        ttk.Entry(entry_frame, textvariable=self.rotation, width=5).grid(row=0, column=1, padx=1)
        ttk.Button(entry_frame, text="+", width=2,
                  command=lambda: self.increment_value(self.rotation, 1)).grid(row=0, column=2, padx=1)
        
        # X Shift
        ttk.Label(params_frame, text="X Shift:").grid(row=1, column=0, sticky=tk.W)
        x_frame = ttk.Frame(params_frame)
        x_frame.grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Scale(x_frame, from_=-500, to=500, orient=tk.HORIZONTAL, 
                 variable=self.x_shift).grid(row=0, column=0, padx=5)
        entry_frame = ttk.Frame(x_frame)
        entry_frame.grid(row=0, column=1)
        ttk.Button(entry_frame, text="-", width=2,
                  command=lambda: self.increment_value(self.x_shift, -1)).grid(row=0, column=0, padx=1)
        ttk.Entry(entry_frame, textvariable=self.x_shift, width=5).grid(row=0, column=1, padx=1)
        ttk.Button(entry_frame, text="+", width=2,
                  command=lambda: self.increment_value(self.x_shift, 1)).grid(row=0, column=2, padx=1)
        
        # Y Shift
        ttk.Label(params_frame, text="Y Shift:").grid(row=2, column=0, sticky=tk.W)
        y_frame = ttk.Frame(params_frame)
        y_frame.grid(row=2, column=1, sticky=(tk.W, tk.E))
        ttk.Scale(y_frame, from_=-500, to=500, orient=tk.HORIZONTAL, 
                 variable=self.y_shift).grid(row=0, column=0, padx=5)
        entry_frame = ttk.Frame(y_frame)
        entry_frame.grid(row=0, column=1)
        ttk.Button(entry_frame, text="-", width=2,
                  command=lambda: self.increment_value(self.y_shift, -1)).grid(row=0, column=0, padx=1)
        ttk.Entry(entry_frame, textvariable=self.y_shift, width=5).grid(row=0, column=1, padx=1)
        ttk.Button(entry_frame, text="+", width=2,
                  command=lambda: self.increment_value(self.y_shift, 1)).grid(row=0, column=2, padx=1)
        
        # Frame Selection
        ttk.Label(params_frame, text="Frame:").grid(row=3, column=0, sticky=tk.W)
        frame_frame = ttk.Frame(params_frame)
        frame_frame.grid(row=3, column=1, sticky=(tk.W, tk.E))
        self.frame_scale = ttk.Scale(frame_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                 variable=self.start_frame)
        self.frame_scale.grid(row=0, column=0, padx=5)
        entry_frame = ttk.Frame(frame_frame)
        entry_frame.grid(row=0, column=1)
        ttk.Button(entry_frame, text="-", width=2,
                  command=lambda: self.increment_value(self.start_frame, -1)).grid(row=0, column=0, padx=1)
        ttk.Entry(entry_frame, textvariable=self.start_frame, width=5).grid(row=0, column=1, padx=1)
        ttk.Button(entry_frame, text="+", width=2,
                  command=lambda: self.increment_value(self.start_frame, 1)).grid(row=0, column=2, padx=1)
        
        # Proto Selection
        ttk.Label(params_frame, text="Proto:").grid(row=4, column=0, sticky=tk.W)
        proto_combo = ttk.Combobox(params_frame, textvariable=self.proto, 
                                 values=['1', '2a', '2b', '3a', '3b', '4a', '4b'],
                                 state='readonly', width=5)
        proto_combo.grid(row=4, column=1, padx=5, sticky=tk.W)
        proto_combo.bind('<<ComboboxSelected>>', self.on_value_change)
        
        # Save Parameters Button
        ttk.Button(main_frame, text="Save Parameters", command=self.save_parameters).grid(row=2, column=0, columnspan=3, pady=10)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=3, pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Select a video file to begin")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=4, column=0, columnspan=3)

    def get_int_value(self, var):
        """Get integer value from a StringVar"""
        try:
            return int(float(var.get()))
        except ValueError:
            return 0

    def increment_value(self, variable, amount):
        """Increment or decrement a variable by the specified amount"""
        try:
            current = int(float(variable.get()))
            variable.set(str(current + amount))
        except ValueError:
            pass

    def schedule_update(self, *args):
        """Schedule an update if one isn't already pending"""
        if not self.update_pending:
            self.update_pending = True
            self.root.after(50, self.update_preview)

    def on_frame_change(self, *args):
        """Handle frame number changes"""
        self.current_frame = None
        self.current_frame_full = None
        self.last_rotation = None
        self.schedule_update()

    def on_value_change(self, *args):
        """Handle value changes"""
        self.schedule_update()

    def browse_file(self):
        """Open a file browser dialog to select a video file"""
        filename = filedialog.askopenfilename(
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if filename:
            try:
                # Clear all caches and state
                self.current_frame = None
                self.current_frame_number = None
                self.current_frame_full = None
                self.last_rotation = None
                self.last_x_shift = None
                self.last_y_shift = None
                self.last_proto = None
                
                # Clear the plot
                self.ax.clear()
                self.canvas.draw()
                
                # Read video info
                cap = cv2.VideoCapture(filename)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                self.filename.set(filename)
                self.frame_scale.configure(to=frame_count)
                self.status_var.set(f"File loaded. Total frames: {frame_count}")
                
                # Force a preview update
                self.update_pending = True
                self.update_preview()
            except Exception as e:
                self.status_var.set(f"Error reading video: {str(e)}")

    def update_preview(self):
        """Update the preview display with current settings"""
        if not self.filename.get():
            self.update_pending = False
            return
            
        try:
            # Clear update flag first
            self.update_pending = False
            
            # Get current values
            frame_pos = self.get_int_value(self.start_frame)
            rotation = self.get_int_value(self.rotation)
            x_shift = self.get_int_value(self.x_shift)
            y_shift = self.get_int_value(self.y_shift)
            proto = self.proto.get()
            
            # Always load frame to ensure we have the latest data
            small_frame, full_frame = self.load_frame(frame_pos)
            if small_frame is None:
                return
                
            # Clear the previous plot
            self.ax.clear()
            
            # Apply transformations to small frame for display
            transformed_frame = ndimage.rotate(small_frame, rotation)
            transformed_frame = ndimage.shift(transformed_frame, 
                                           (x_shift//self.preview_scale, 
                                            y_shift//self.preview_scale, 
                                            0))
            
            # Display downsampled frame
            self.ax.imshow(cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2RGB))
            
            # Set axis limits to match the downsampled frame size
            h, w = transformed_frame.shape[:2]
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
            
            # Draw grid overlay at original scale, but convert coordinates to preview scale
            self.ax.plot([0, 2216//self.preview_scale], 
                       [1200//self.preview_scale, 1200//self.preview_scale], 'b--')
            self.ax.plot([1200//self.preview_scale, 1200//self.preview_scale], 
                       [0, 2216//self.preview_scale], 'b--')
            
            # Get grid points based on proto (using original BINING)
            rr, ll = get_frame(proto, BINING=2)
            
            # Draw rectangles (converting coordinates to preview scale)
            for y_offset in range(10):
                for x_offset in range(10):
                    # Right side
                    ya = rr[x_offset][y_offset]['ya']//self.preview_scale
                    yb = rr[x_offset][y_offset]['yb']//self.preview_scale
                    xa = rr[x_offset][y_offset]['xa']//self.preview_scale
                    xb = rr[x_offset][y_offset]['xb']//self.preview_scale
                    self.ax.plot([xa,xb,xb,xa,xa], [ya,ya,yb,yb,ya], 'r', alpha=0.7, linewidth=1)
                    
                    # Left side
                    ya = ll[x_offset][y_offset]['ya']//self.preview_scale
                    yb = ll[x_offset][y_offset]['yb']//self.preview_scale
                    xa = ll[x_offset][y_offset]['xa']//self.preview_scale
                    xb = ll[x_offset][y_offset]['xb']//self.preview_scale
                    self.ax.plot([xa,xb,xb,xa,xa], [ya,ya,yb,yb,ya], 'r', alpha=0.7, linewidth=1)
            
            # Turn off axis labels
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Update the canvas
            self.canvas.draw()
            
            # Store current values
            self.last_rotation = rotation
            self.last_x_shift = x_shift
            self.last_y_shift = y_shift
            self.last_proto = proto
            
            self.status_var.set(f"Previewing frame {frame_pos}")
            
        except Exception as e:
            self.status_var.set(f"Error updating preview: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_frame(self, frame_number):
        """Load a frame and cache it"""
        if (self.current_frame is not None and 
            self.current_frame_number == frame_number):
            return self.current_frame, self.current_frame_full
            
        try:
            cap = cv2.VideoCapture(self.filename.get())
            if not cap.isOpened():
                self.status_var.set("Error: Could not open video file")
                return None, None
                
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                self.status_var.set(f"Error: Could not read frame {frame_number}")
                return None, None
            
            # Store full-size frame for grid calculations
            self.current_frame_full = frame.copy()
            
            # Downsample frame for display only
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (w//self.preview_scale, h//self.preview_scale), 
                                   interpolation=cv2.INTER_NEAREST)  # Faster interpolation
            
            # Cache the frames
            self.current_frame = small_frame
            self.current_frame_number = frame_number
            
            return small_frame, self.current_frame_full
        except Exception as e:
            self.status_var.set(f"Error loading frame: {str(e)}")
            return None, None

    def save_parameters(self):
        """Save the current parameters to a JSON file"""
        if not self.filename.get():
            self.status_var.set("Please select a video file first")
            return
            
        try:
            # Save parameters to JSON
            params = {
                "r": self.get_int_value(self.rotation),
                "x": self.get_int_value(self.x_shift),
                "y": self.get_int_value(self.y_shift),
                "p": self.proto.get(),
                "s": self.get_int_value(self.start_frame)
            }
            
            json_filename = f'{self.filename.get()}.json'
            with open(json_filename, "w") as outfile:
                json.dump(params, outfile, indent=4)
            
            self.status_var.set(f"Parameters saved to {json_filename}")
        except Exception as e:
            self.status_var.set(f"Error saving parameters: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    root = tk.Tk()
    app = PPGAlignmentGUI(root)
    root.mainloop() 