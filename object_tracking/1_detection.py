# vim: expandtab:ts=4:sw=4
import argparse
import numpy as np
import os
import sys
import cv2
import csv
import tensorflow as tf
import pandas as pd
import time
from tqdm import tqdm

from collections import defaultdict
from io import StringIO

sys.path.insert(0, os.path.abspath(".."))
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# ================================================================================

def detection(sess, video_path, det_path):
    points_objs = []
    start = time.time()
    
    skip = 0
    id_frame = 1
    id_center = 0
    
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length)
    while(True):
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        image_np = np.array(frame)
        if(image_np.shape == ()): break

        # print('Frame ID:', id_frame, '\tTime:', '{0:.2f}'.format(time.time()-start), 'seconds')

        image_np_expanded = np.expand_dims(image_np, axis=0)


        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        count_boxes = 0
        thresh = 0.2
        max_boxes = 50

        for i, c in enumerate(classes):
            if (c == 1 and (scores[i] > thresh) and (count_boxes < max_boxes)):
                im_height = image_np.shape[0]
                im_width = image_np.shape[1]
                ymin, xmin, ymax, xmax = boxes[i]

                (left, right, top, bottom) = (int(xmin*im_width),  int(xmax*im_width),
                                              int(ymin*im_height), int(ymax*im_height))
                points_objs.append([
                    id_frame, -1,
                    left, top, right-left, bottom-top,
                    scores[i],
                    -1, -1, -1
                ])
                count_boxes += 1

        id_frame += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    cv2.destroyAllWindows()

    # write detection

    with open(det_path[:-3]+'csv', 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(points_objs)

# ==============================================================================

"""Parse command line arguments."""
def parse_args():
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
        "--det_dir", 
        help="Detection directory. Will be created if it does not exist.", 
        default="../dataset/detections"
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    # ========================  Model initialization  ==============================
    print ('loading model..')
    PATH_TO_CKPT = '../object_detection/faster_rcnn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(os.path.join('..', 'object_detection','data', 'mscoco_label_map.pbtxt'))
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # ==============================================================================
   
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            videos = os.listdir(args.video_dir)
            videos.sort()
            for video_name in videos:

                if(video_name != args.test_video and args.test_video != '' ): 
                    continue

                print('Processing Video:', video_name + '..')
                detection(sess, 
                            video_path=os.path.join(args.video_dir, video_name),
                            det_path=os.path.join(args.det_dir, video_name)
                        )
