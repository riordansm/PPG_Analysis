import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram
import json
import pickle
import os
import pandas as pd

LIGHT_ON_BUFFER = 50 * 3

def save_figure_data(data, filename, figure_name):
    """Save figure data to CSV"""
    # For spectrograms (fig4 and fig5), data is already properly structured
    if figure_name in ['fig4', 'fig5']:
        df = pd.DataFrame(data)
    else:
        # For time series data (fig1-3), transpose the data for proper CSV format
        # Each row will be a time point, each column a channel
        df = pd.DataFrame(data).T
        df.columns = [f'channel_{i}' for i in range(df.shape[1])]
    
    csv_filename = f'{filename}.{figure_name}.csv'
    df.to_csv(csv_filename, index=False)

def save_figures(filename, figs):
    """Save figures to a directory named after the file"""
    # Create directory if it doesn't exist
    save_dir = f'{filename}_figures'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each figure
    for i, fig in enumerate(figs, 1):
        fig.savefig(os.path.join(save_dir, f'figure_{i}.png'))
        plt.close(fig)  # Close figure to free memory

def process_single_file(filename, add_plots=False):
    try:
        with open(f'{filename}.pkl', 'rb') as f:
            x = pickle.load(f)
        
        print(f"\nProcessing {filename}")
        print(np.shape(x['l']))
        l = x['l'][:, LIGHT_ON_BUFFER:]
        r = x['r'][:, LIGHT_ON_BUFFER:]
        
        figures = []
        if add_plots:
            # Figure 1
            fig1 = plt.figure()
            for i in range(10):
                plt.plot(r[10*i,::3], label = f'({i},0)')
            plt.xlabel('time (s)')
            plt.ylabel('count')
            plt.legend()
            figures.append(fig1)
            
            # Figure 2
            fig2 = plt.figure()
            for i in range(10):
                plt.plot(r[10*i,1::3])
            plt.xlabel('time (s)')
            plt.ylabel('count')
            plt.legend()
            figures.append(fig2)
            
            # Figure 3
            fig3 = plt.figure()
            for i in range(10):
                plt.plot(r[10*i,2::3])
            plt.xlabel('time (s)')
            plt.ylabel('count')
            plt.legend()
            figures.append(fig3)
        
        optical = np.sum(r[:,::3], axis = 0)
        Pxx, freqs, t = specgram(optical, NFFT=256, Fs=25, noverlap=100, pad_to=1024)

        if add_plots:
            # Figure 4
            fig4 = plt.figure()
            plt.pcolormesh(t, freqs[35:180] * 60, 10 * np.log10(Pxx[35:180]))
            plt.ylim([50,250])
            plt.plot(t, freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 2*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 3*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.xlabel('time (s)')
            plt.ylabel('frq (BPM)')
            figures.append(fig4)

        frq = freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60

        print("Min", np.min(frq), "Max", np.max(frq))
        print("Range", np.max(frq) - np.min(frq), "\n !!BAD DATA Quality!! " if np.max(frq) - np.min(frq) > 10 else "")
        print("Max Pxx",  10 * np.log10(np.max(Pxx[35:180],axis= 0)))
        print("Mean Pxx", 10 * np.log10(np.median(Pxx[35:180],axis= 0)))

        dictionary = {"Min": np.min(frq), 
                      "Max": np.max(frq), 
                      "Range": np.max(frq) - np.min(frq),
                      "Max Pxx":  10 * np.log10(np.max(Pxx[35:180],axis= 0)),
                      "Mean Pxx": 10 * np.log10(np.median(Pxx[35:180],axis= 0))}

        with open(f'{filename}.result.pkl', 'wb') as f:
                    pickle.dump(dictionary, f)

        optical = r[0,::3]
        
        if add_plots:
            Pxx, freqs, t = specgram(optical, NFFT=256, Fs=25, noverlap=100, pad_to=1024)
            
            # Figure 5
            fig5 = plt.figure()
            plt.pcolormesh(t, freqs[35:180] * 60, 10 * np.log10(Pxx[35:180]))
            plt.ylim([50,250])
            plt.plot(t, freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 2*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 3*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.xlabel('time (s)')
            plt.ylabel('frq (BPM)')
            figures.append(fig5)
            
            # Save all figures
            save_figures(filename, figures)
        
        return True
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Video Analysis")
        parser.add_argument("-p", action='store_true', default=False, help="Add Plots")
        parser.add_argument("directory", help="Directory containing .avi.pkl files to process")
        
        args = parser.parse_args()

        # Get all .avi.pkl files in directory
        directory = args.directory
        pkl_files = [f for f in os.listdir(directory) 
                    if f.endswith('.avi.pkl')]
        
        print(f"Found {len(pkl_files)} .avi.pkl files to process")
        for pkl_file in pkl_files:
            # Get the base filename without .pkl extension but keeping .avi
            base_filename = os.path.join(directory, pkl_file[:-4])  # removes only '.pkl'
            print(f"\nProcessing {pkl_file}")
            process_single_file(base_filename, args.p)
