# vim: expandtab:ts=4:sw=4
import argparse

import os
import cv2
import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

from math import factorial
from deep_sort.iou_matching import iou
from application_util import visualization
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def gather_sequence_info(video_name, video_path, feat_path):
    detections = np.load(feat_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    image_size = np.array(frame).shape[:-1] 
    # print(image_size)

    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": video_name,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": None
    }
    return seq_info

# ========================================================================================================


def capture(video_path, cap_dir, results, seq_info, is_plot=False):

    if os.path.exists(cap_dir):
        shutil.rmtree(cap_dir)
    os.makedirs(cap_dir)

    cap = cv2.VideoCapture(video_path)

    N_track = int(max(results[:,1]))
    subplot_x = 6
    subplot_y = int(math.ceil(N_track/subplot_x))
    print('Total Tracks:', N_track)
    print('Subplot', subplot_y, subplot_x)

    image_size = seq_info['image_size']
    points = {}
    captured = []

    with tf.Session() as sess:
        for frame_idx in tqdm(range(
                            seq_info['min_frame_idx'], 
                            seq_info['max_frame_idx'] + 1), 'capturing output'):
        
            image_np = np.array(cap.read()[1])

            mask = results[:, 0].astype(np.int) == frame_idx
            track_ids = results[mask, 1].astype(np.int)
            boxes = results[mask, 2:6]

            for track_id, box in zip(track_ids, boxes):
                if(track_id not in captured):
                    captured.append(track_id)

                    l,t,w,h = np.array(box).astype(int)
                    if(l<0): l=0 # if xmin is negative 
                    if(t<0): t=0 # if ymin is negative

                    if(l+w > image_size[1]): w=image_size[1]-l # if xmax exceeds width
                    if(t+h > image_size[0]): h=image_size[0]-t # if ymax exceeds height

                    cropped_image = sess.run(tf.image.crop_to_bounding_box(image_np, t, l, h, w))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                    img = Image.fromarray(cropped_image)
                    img.save(os.path.join(cap_dir, str(track_id)+'.jpg'))

                    if(is_plot):
                        plt.subplot(subplot_y, subplot_x, len(captured))
                        plt.imshow(cropped_image)
                        plt.title(str(track_id)+', '+str(frame_idx))

    cap.release()

    if(is_plot):
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.5, wspace=0.8)
        plt.show()

# ========================================================================================================
   
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
   
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2: # order should be less than or equal window-2
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve( m[::-1], y, mode='valid')

# ========================================================================================================

def golay_filter(df_track, window_size=45, order=5):
    if(len(df_track) <= window_size):
        return df_track
    df_track[2] = savitzky_golay(df_track[2].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[3] = savitzky_golay(df_track[3].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[4] = savitzky_golay(df_track[4].values, window_size=window_size, order=order, deriv=0, rate=1)
    df_track[5] = savitzky_golay(df_track[5].values, window_size=window_size, order=order, deriv=0, rate=1)
    return df_track
    
def poly_interpolate(df_track):
    model = make_pipeline(PolynomialFeatures(5), Ridge(solver='svd'))
    X = np.array(df_track.index).reshape(-1, 1)
    df_track[2] = model.fit(X, df_track[2]).predict(X)
    df_track[3] = model.fit(X, df_track[3]).predict(X)
    df_track[4] = model.fit(X, df_track[4]).predict(X)
    df_track[5] = model.fit(X, df_track[5]).predict(X)
    return df_track

def moving_avg(df_track, window=5):
    df_haed = df_track[[2,3,4,5]][:window-1]
    df_tail = df_track[[2,3,4,5]].rolling(window=window).mean()[window-1:]
    df_track[[2,3,4,5]] = pd.concat([df_haed, df_tail], axis=0)
    return df_track

def smooth(df, smooth_method):
    polynomials = []
    From, To = min(df[1]), max(df[1])+1
    for track_id in range(From, To):
        df_track = df.loc[df[1]==track_id].copy()

        if(smooth_method == 'poly'): df_track = poly_interpolate(df_track)
        elif(smooth_method == 'moving'): df_track = moving_avg(df_track)
        elif(smooth_method == 'golay'): df_track = golay_filter(df_track)
            
        polynomials.append(df_track)

    df_smooth = pd.concat(polynomials)
    df_smooth = df_smooth.sort_index()
    return df_smootVideoWriterh.values

# ========================================================================================================

def run(video_path, track_path, feat_path, save_output, out_dir, cap_dir, concat, smoothing, save_fig, is_plot, lag = 30):
    video_name = os.path.basename(video_path)
    seq_info = gather_sequence_info(video_name, video_path, feat_path)
   
    if(concat): track_path = track_path[:-4]+'_join.csv'

    df = pd.read_csv(track_path, header=None)

    if(smoothing): results = smooth(df, smooth_method='golay')
    else: results = df.values

    if(save_fig):
        capture(video_path, cap_dir, results, seq_info, is_plot=is_plot)
        return


    cap = cv2.VideoCapture(video_path)
    print('Video Path:', video_path,'\tFeatures:', feat_path)

    print(lag)
    points = {}

    def frame_callback(vis, frame_idx):
        # print("Frame idx", frame_idx)
        image_np = np.array(cap.read()[1])
        vis.set_image(image_np)

        mask = results[:, 0].astype(np.int) == frame_idx
        track_ids = results[mask, 1].astype(np.int)
        boxes = results[mask, 2:6]

        points[frame_idx] = []
        for track_id, box in zip(track_ids, boxes):
            l,t,w,h = np.array(box).astype(int)
            x, y = int(l+w/2), int(t+h)
            points[frame_idx].append([track_id, x, y])

        if(frame_idx > lag):
            remove_idx = frame_idx-lag
            if remove_idx in points:
                del points[remove_idx]

        vis.draw_groundtruth(track_ids, boxes, points)

    visualizer = visualization.Visualization(seq_info, update_ms=50)

    if save_output:
        if(concat):
            visualizer.viewer.enable_videowriter(os.path.join(out_dir, video_name[:-4]+'_opt.mp4'))
        else:
            visualizer.viewer.enable_videowriter(os.path.join(out_dir, video_name[:-4]+'_reg.mp4'))

    visualizer.run(frame_callback)

    cap.release()
    cv2.destroyAllWindows()


# ========================================================================================================


def parse_args():

    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--test_video", 
        help="To run specific one", 
        default=''
    )
    parser.add_argument(
        "--video_dir", 
        help="Path to video directory.", 
        default="../dataset/videos"
    )
    parser.add_argument(
        "--tracks_dir", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="../dataset/tracks"
    )
    parser.add_argument(
        "--feat_dir", 
        help="Features directory.",
        default="../dataset/features"
    )
    parser.add_argument(
        "--save_output", help="Save output of the tracking video (bool).",
        default=False, type=bool)
    parser.add_argument(
        "--out_dir", help="Output directory",
        default='../dataset/outputs'
    )
    parser.add_argument(
        "--cap_dir", help="Captures directory",
        default='../dataset/captures'
    )
    parser.add_argument(
        "--concat", help="Show concatenated points",
        default=False, type=bool
    ),
    parser.add_argument(
        "--smoothing", help="Show concatenated points",
        default=False, type=bool
    )
    parser.add_argument(
        "--save_fig", help="Save captured human",
        default=False, type=bool
    )
    parser.add_argument(
        "--is_plot", help="Plot captured human",
        default=False, type=bool
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    videos = os.listdir(args.video_dir)
    videos.sort()
    for video_name in videos:
        if(video_name != args.test_video and args.test_video != '' ): 
            continue
        try:                
            run(video_path = os.path.join(args.video_dir, video_name), 
                track_path = os.path.join(args.tracks_dir, video_name[:-3]+'csv'),
                feat_path  = os.path.join(args.feat_dir, video_name[:-3]+'npy'), 
                save_output = args.save_output, # save tracking video 
                out_dir = args.out_dir,
                cap_dir = os.path.join(args.cap_dir, video_name[:-4]),
                concat = args.concat,
                smoothing = args.smoothing,
                save_fig = args.save_fig, # save captured human
                is_plot = args.is_plot) # subplot of captured human

        except FileNotFoundError:
            print(video_name + " has not yet been generated.")