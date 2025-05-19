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
            
            
            
if __name__ == '__main__':

        parser = argparse.ArgumentParser(description="Video Analysis")
        parser.add_argument("filename", help="Name of the file to process")
        parser.add_argument("-r",  type=int, default=None, help="rotation")
        parser.add_argument("-x",  type=int, default=None, help="shift x")
        parser.add_argument("-y",  type=int, default=None, help="shift y")
        parser.add_argument("-s",  type=int, default=2, help="start frame for alignment")
        parser.add_argument("-p",  default='1', help="start frame for alignment")
        
        
        args = parser.parse_args()
        
        if args.x is not None:
            dictionary = {
                "r": args.r,
                "x": args.x,
                "y": args.y,
                "p": args.p,
                "s":args.s
            }
             
            json_object = json.dumps(dictionary, indent=4)
             
            with open(f'{args.filename}.json', "w") as outfile:
                outfile.write(json_object)
                
            plot_frame(input_video_path  = args.filename, rotation = args.r, shift = (args.x, args.y,0), start_frame = args.s, proto = args.p)
            plt.show()

            
        else:
            with open(f'{args.filename}.json', 'r') as openfile:
             
                # Reading from json file
                json_object = json.load(openfile)
                args.r = json_object['r']
                args.x = json_object['x']
                args.y = json_object['y']
                args.p = json_object['p']
                args.s = json_object['s']
                    

        l, r = cal_frame(input_video_path  = args.filename, rotation = args.r, shift = (args.x, args.y,0), proto = args.p)
        
        with open(f'{args.filename}.pkl', 'wb') as f:
            pickle.dump({'l':l, 'r':r, 'x':args.x, 'y':args.y, 'rotation':args.r},f)

