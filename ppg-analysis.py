import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram
import json
import pickle

LIGHT_ON_BUFFER = 50 * 3

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description="Video Analysis")
        parser.add_argument("-p",  action='store_true', default=False, help="Add Plots")
        parser.add_argument("filename", help="Name of the file to process")
        
        args = parser.parse_args()


        with open(f'{args.filename}.pkl', 'rb') as f:
            x = pickle.load(f)
        
        print(np.shape(x['l']))
        l = x['l'][:, LIGHT_ON_BUFFER:]
        r = x['r'][:, LIGHT_ON_BUFFER:]
        
        
        if args.p:
            for i in range(10):
                plt.plot(r[10*i,::3], label = f'({i},0)')
            plt.xlabel('time (s)')
            plt.ylabel('count')
            plt.legend()
                
            plt.legend()
            plt.figure()
            for i in range(10):
                plt.plot(r[10*i,1::3])
            plt.xlabel('time (s)')
            plt.ylabel('count')
            plt.legend()
            plt.figure()
            for i in range(10):
                plt.plot(r[10*i,2::3])
            plt.xlabel('time (s)')
            plt.ylabel('count')
            plt.legend()
            
            
        
        optical = np.sum(r[:,::3], axis = 0)
        Pxx, freqs, t = specgram(optical, NFFT=256, Fs=25, noverlap=100, pad_to=1024)

        if args.p:
            plt.figure()
            plt.pcolormesh(t, freqs[35:180] * 60, 10 * np.log10(Pxx[35:180]))
            plt.ylim([50,250])

            plt.plot(t, freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 2*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 3*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')

            plt.xlabel('time (s)')
            plt.ylabel('frq (BPM)')

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

        with open(f'{args.filename}.result.pkl', 'wb') as f:
                    pickle.dump(dictionary, f)

        optical = r[0,::3]
        
        if args.p:
            Pxx, freqs, t = specgram(optical, NFFT=256, Fs=25, noverlap=100, pad_to=1024)
            plt.figure()
            plt.pcolormesh(t, freqs[35:180] * 60, 10 * np.log10(Pxx[35:180]))
            plt.ylim([50,250])

            plt.plot(t, freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 2*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')
            plt.plot(t, 3*freqs[35:180][np.argmax(Pxx[35:180],axis= 0)] * 60, 'r--')

            plt.xlabel('time (s)')
            plt.ylabel('frq (BPM)')
        
        plt.show()
