# vim: expandtab:ts=4:sw=4
import argparse

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def join(df_track):
    prev_frame_idx = min(df_track['track_id'].index)
    results = []
    for frame_idx, currrent_row in df_track.iterrows():
        gap = frame_idx - prev_frame_idx
        if(gap > 1):
            results.append(str(prev_frame_idx)+' -> '+ str(frame_idx))
            currrent_row = np.array(currrent_row)
            previous_row = np.array(df_track.loc[prev_frame_idx].values)
            steps = (currrent_row - previous_row) / gap

            for i, frame in enumerate(range(prev_frame_idx+1,frame_idx)):
                df_track.loc[frame] = np.array(previous_row + (i+1) * steps).astype(int)

        prev_frame_idx = frame_idx
    df_track = df_track.sort_index()

    misses = np.squeeze(list(set(range(min(df_track.index), 
                                       max(df_track.index) + 1)).difference(df_track.index)))
    if(len(misses)==0 and len(results) > 0):
        print('Track:', int(df_track['track_id'].iloc[0]),', concatenation complete, ',results)
    elif(len(misses)!=0):
        print('Warning!! Frame:', int(df_track['track_id'].iloc[0]), ', concatenation incomplete\n')
    return df_track

def run(track_path):
    concat_track_file = track_path[:-4]+'_join.csv'
    try: os.remove(concat_track_file)
    except OSError: pass

    df = pd.read_csv(track_path, header=None)
    df.columns = ['frame_id','track_id','xmin','ymin','width','height', 
                  'confidence','neg_1', 'neg_2', 'neg_3']
    df.index = df['frame_id']
    df = df.drop(['frame_id'], axis=1)

    concat = []
    From, To = min(df['track_id']), max(df['track_id'])+1
    for track_id in range(From, To):
        concat.append(join(df.loc[df['track_id']==track_id].copy()))
        
    df_concat = pd.concat(concat)
    df_concat = df_concat.sort_index() 
    df_concat.to_csv(concat_track_file, header=None)
    print('=================')

def parse_args():

    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--test_video", 
        help="To run specific one. If None, it will process all tracks.", 
        default=''
    )
    parser.add_argument(
        "--tracks_dir", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="../dataset/tracks"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tracks = os.listdir(args.tracks_dir)
    tracks.sort()

    test_track = ''
    if(args.test_video != ''):
        test_track = args.test_video[:-3]+'csv'

    for track in tracks:

        if('join.csv' in track):
            continue
        if(track != test_track and test_track != '' ): 
            continue
        print('\nProcessing:', track)
        run(track_path = os.path.join(args.tracks_dir, track))