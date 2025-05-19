import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from ppg_preprocess import plot_frame

class PPGPreprocessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PPG Video Preprocessor")
        
        # Variables
        self.filename = tk.StringVar()
        self.rotation = tk.IntVar(value=0)
        self.x_shift = tk.IntVar(value=0)
        self.y_shift = tk.IntVar(value=0)
        self.start_frame = tk.IntVar(value=2)
        self.proto = tk.StringVar(value='1')
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        ttk.Label(main_frame, text="Video File:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.filename, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Alignment Parameters", padding="5")
        params_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Rotation
        ttk.Label(params_frame, text="Rotation:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.rotation, width=10).grid(row=0, column=1, padx=5)
        
        # X Shift
        ttk.Label(params_frame, text="X Shift:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.x_shift, width=10).grid(row=1, column=1, padx=5)
        
        # Y Shift
        ttk.Label(params_frame, text="Y Shift:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.y_shift, width=10).grid(row=2, column=1, padx=5)
        
        # Start Frame
        ttk.Label(params_frame, text="Start Frame:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(params_frame, textvariable=self.start_frame, width=10).grid(row=3, column=1, padx=5)
        
        # Proto Selection
        ttk.Label(params_frame, text="Proto:").grid(row=4, column=0, sticky=tk.W)
        proto_combo = ttk.Combobox(params_frame, textvariable=self.proto, 
                                 values=['1', '2a', '2b', '3a', '3b', '4a', '4b'])
        proto_combo.grid(row=4, column=1, padx=5)
        
        # Preview Button
        ttk.Button(main_frame, text="Preview Alignment", command=self.preview_alignment).grid(row=2, column=0, columnspan=3, pady=10)
        
        # Process Button
        ttk.Button(main_frame, text="Process Video", command=self.process_video).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Figure for preview
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=3, pady=10)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if filename:
            self.filename.set(filename)
            
    def preview_alignment(self):
        if not self.filename.get():
            return
            
        self.ax.clear()
        plot_frame(
            input_video_path=self.filename.get(),
            rotation=self.rotation.get(),
            shift=(self.x_shift.get(), self.y_shift.get(), 0),
            start_frame=self.start_frame.get(),
            proto=self.proto.get()
        )
        self.canvas.draw()
        
    def process_video(self):
        if not self.filename.get():
            return
            
        # Import the main processing function from ppg_preprocess
        from ppg_preprocess import cal_frame
        import json
        import pickle
        
        # Save parameters to JSON
        params = {
            "r": self.rotation.get(),
            "x": self.x_shift.get(),
            "y": self.y_shift.get(),
            "p": self.proto.get(),
            "s": self.start_frame.get()
        }
        
        with open(f'{self.filename.get()}.json', "w") as outfile:
            json.dump(params, outfile, indent=4)
            
        # Process video and save PKL
        l, r = cal_frame(
            input_video_path=self.filename.get(),
            rotation=self.rotation.get(),
            shift=(self.x_shift.get(), self.y_shift.get(), 0),
            proto=self.proto.get()
        )
        
        with open(f'{self.filename.get()}.pkl', 'wb') as f:
            pickle.dump({
                'l': l,
                'r': r,
                'x': self.x_shift.get(),
                'y': self.y_shift.get(),
                'rotation': self.rotation.get()
            }, f)

if __name__ == "__main__":
    root = tk.Tk()
    app = PPGPreprocessGUI(root)
    root.mainloop() 