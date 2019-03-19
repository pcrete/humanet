import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import pandas as pd
import difflib
import time
import plotly.plotly as py
import plotly.graph_objs as go
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm_notebook

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(".."))
from utils import label_map_util
from utils import visualization_utils as vis_util

def detection(path):
    points_objs = {}
    frame_objs = {}
    
    start = time.time()
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            skip = 0
            cap = cv2.VideoCapture(path)

            id_frame = 0;
            id_center = 0;

            while(True):
                ret, frame = cap.read()
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                if(skip == 2):
                    skip = 0
                    image_np = np.array(frame)
                    if(image_np.shape == ()): break

                    print('Frame ID:', id_frame, '\tTime:', '{0:.2f}'.format(time.time()-start), 'seconds')
                    frame_objs[id_frame] = image_np
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
                    thresh = 0.75
                    max_boxes = 40

                    for i, c in enumerate(classes):
                        if (c == 1 and (scores[i] > thresh) and (count_boxes < max_boxes)):

                            im_height = image_np.shape[0]
                            im_width = image_np.shape[1]

                            ymin, xmin, ymax, xmax = boxes[i]
                            x_center, y_center = (xmin+xmax)/2, (ymin+ymax)/2

                            points_objs[id_center] = {
                                'frame': id_frame,
                                'center': [x_center, y_center],
                                'bbox': [ymin, xmin, ymax, xmax],
                                'score': scores[i]
                            }
                            (left, right, top, bottom) = (int(xmin*im_width),  int(xmax*im_width),
                                                          int(ymin*im_height), int(ymax*im_height))

                            (xmin, xmax, ymin, ymax) = (left, right, top, bottom)
                            points_objs[id_center]['bbox_real'] =  [xmin, xmax, ymin, ymax]

                            id_center += 1
                            count_boxes += 1

                    id_frame += 1

                skip+=1

            cap.release()
            cv2.destroyAllWindows()

        return points_objs, frame_objs


def create_graph(points_objs, frame_objs, df_points):
    graphs = {-1:[]}
    for i in points_objs:
        graphs[i] = []

    df_prev = df_points.loc[df_points['frame'] == 0]
    for i,row in df_prev.iterrows():
        graphs[-1].append(i)

    for k in tqdm_notebook(range(len(frame_objs)-1)):
        df_prev = df_points.loc[df_points['frame'] == k]
        df_curr = df_points.loc[df_points['frame'] == k+1]

        for i, row_i in df_curr.iterrows():
            curr_point = np.array(row_i['center'])
            Min = 9999999.0
            ind = -1
            for j, row_j in df_prev.iterrows():
                prev_point = np.array(row_j['center'])
                dist = np.linalg.norm(prev_point - curr_point)
                if(dist < Min):
                    Min = dist
                    ind = j
            Sim = 1  
#             Sim = compare_images(i, ind, points_objs, frame_objs, df_points)

            if(Min < 0.03 and Sim > 0.8):
                graphs[ind].append(i)
            else:
                graphs[-1].append(i)

    # Add End point
    for i in points_objs:
        if(len(graphs[i]) == 0):
            graphs[i].append(-99)
    
    return graphs


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

def graph_paths(graphs):
    paths = (find_all_paths(graphs, -1, -99))
    for path in paths:
        print(path)
    print('Total path:', len(paths),'\n')
    
    to_removes = []
    removed = 0
    for i in range(len(paths)-1):
        sm = difflib.SequenceMatcher(None, paths[i], paths[i+1])
        if(sm.ratio() > 0.7):
            to_removes.append(paths[i+1])
        print(i,i-1,sm.ratio())
    print('Removed:', len(to_removes))

    for to_remove in to_removes:
        paths.remove(to_remove)
    return paths


def visualize_selected(paths, df_points, directory):
    os.makedirs(os.path.join(directory, 'output'), exist_ok=True)

    inds = []
    duplicates = []
    tmp_boxes = []
    tmp_scores = []
    for path in paths:
        
        inds_path = path[1:len(path)-1]
        if(len(inds_path) <= 3): continue
            
        for ind in inds_path:
            if ind not in duplicates:
                inds.append(ind)
                tmp_boxes.append(df_points.loc[ind]['bbox'])
                tmp_scores.append(df_points.loc[ind]['score'])
                break
        duplicates += path
        duplicates = list(set(duplicates))
    inds = np.array(inds)
    print('Total Objects:', len(inds))

    nms = None
    with tf.Session() as sess:    
        nms = sess.run(tf.image.non_max_suppression( tmp_boxes, tmp_scores, max_output_size = len(inds), iou_threshold = 0.6))

        inds = inds[nms]
        print(inds)
        print(len(inds))
        
        for ind in inds:
            print('ID:', ind, '\tscore:', df_points.loc[ind]['score'])
            obj = df_points.loc[ind]
            xmin, xmax, ymin, ymax = obj['bbox_real']
            cropped_image = tf.image.crop_to_bounding_box(frame_objs[obj['frame']], ymin, xmin, ymax-ymin, xmax-xmin)
            cropped_image = np.array(sess.run(cropped_image))
            plt.imshow( cropped_image[:,:,::-1])
            plt.axis('off')
            plt.savefig(os.path.join(directory, 'output', str(ind)+'.png'), bbox_inches='tight', pad_inches = 0)
            # plt.show()

# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

print ('loading model..')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print('model loaded')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


DIR = '../data'
VIDEO = '3_BEST'


points_objs, frame_objs = detection(path=os.path.join(DIR,VIDEO,VIDEO+'.mp4'))

df_points = pd.DataFrame.from_dict(points_objs, orient='index')

graphs = create_graph(points_objs, frame_objs, df_points)
print('Graphs:',len(graphs),'nodes')

paths = graph_paths(graphs)

visualize_selected(paths, df_points, directory=os.path.join(DIR,VIDEO))