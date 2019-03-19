# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from time import sleep

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


def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(video_name, video_path, feat_path, track_path, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):

    seq_info = gather_sequence_info(video_name, video_path, feat_path)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=50, n_init=5)
    results = []
    cap = cv2.VideoCapture(video_path)
    print('Video Path:', video_path,'\tFeatures:', feat_path)

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            vis.set_image(cap.read()[1])
            
            # vis.draw_detections(detections)
            count_human = vis.draw_trackers(tracker.tracks)
        else:
            count_human = vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()

            ID = count_human.index(track.track_id) + 1
            results.append([frame_idx, ID, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=50)
    else:
        visualizer = visualization.NoVisualization(seq_info)

    visualizer.run(frame_callback)

    cap.release()
    # Store results.
    f = open(track_path, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")

    parser.add_argument(
        "--test_video", 
        help="To run specific one", 
        default=''
    )
    parser.add_argument(
        "--video_dir", 
        help="Video directory.", 
        default="../dataset/videos"
    )
    parser.add_argument(
        "--det_dir", help="Path to detection directory",
        default='../dataset/detections'
    )
    parser.add_argument(
        "--feat_dir", 
        help="Features directory.",
        default="../dataset/features"
    )
    parser.add_argument(
        "--tracks_dir", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="../dataset/tracks"
    )
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.7, type=float
    )
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int
    )
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float
    )
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.25
    )
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None
    )
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=False, type=bool
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # print(args.display)
    videos = os.listdir(args.video_dir)
    videos.sort()
    for video_name in videos:
        if(video_name != args.test_video and args.test_video != '' ): 
            continue
        try:
            run(
                video_name = video_name,
                video_path = os.path.join(args.video_dir, video_name), 
                feat_path  = os.path.join(args.feat_dir, video_name[:-3]+'npy'), 
                track_path = os.path.join(args.tracks_dir, video_name[:-3]+'csv'),
                min_confidence  = args.min_confidence, 
                nms_max_overlap = args.nms_max_overlap, 
                min_detection_height = args.min_detection_height,
                max_cosine_distance  = args.max_cosine_distance, 
                nn_budget = args.nn_budget, 
                display   = args.display
                )
        except FileNotFoundError:
            print(video_name + " has not yet been generated.")
            
    # print('================')