import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0,os.path.abspath(".."))
from utils import label_map_util
from utils import visualization_utils as vis_util

# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

PATH_TO_TEST_IMAGES_DIR = 'test_images'
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 30)

print 'loading model..'

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
	
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
	im_width, im_height = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		skip = 0
		# cap = cv2.VideoCapture(0)
		# cap = cv2.VideoCapture('branson.mp4')
		# cap = cv2.VideoCapture('/home/poom/Desktop/Driving_CA_USA.mp4')
		cap = cv2.VideoCapture()
		# Opening the link
		cap.open("http://192.168.1.34:8080/video?.mjpeg")    
		while(True):
			ret, frame = cap.read()
			if(skip == 3):
				skip = 0
				image_np = np.array(frame)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
	           
				image_np_expanded = np.expand_dims(image_np, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				
				(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
				    feed_dict={image_tensor: image_np_expanded})
							
				vis_util.visualize_boxes_and_labels_on_image_array(
				image_np,
				    np.squeeze(boxes),
				    np.squeeze(classes).astype(np.int32),
				    np.squeeze(scores),
				    category_index,
				    use_normalized_coordinates=True,
				    line_thickness=1,
				    max_boxes_to_draw=20,
				    min_score_thresh=0.6)

				cv2.imshow('frame',image_np)
			skip+=1

		cap.release()
		cv2.destroyAllWindows()