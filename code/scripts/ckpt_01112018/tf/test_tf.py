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
import time
import ntpath

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

PATH_TO_TF_MODELS = sys.argv[1]
PATH_TO_TEST_IMAGES_DIR = sys.argv[2]
OUTPUTFILE = sys.argv[3]
MODEL_NAME=sys.argv[4]
SAMPLING=float(sys.argv[5])
SIZE=int(sys.argv[6])
MODEL_DIR=sys.argv[7]

print "***********************************"
print "Input images:\t"+PATH_TO_TEST_IMAGES_DIR
print "Output file:\t"+OUTPUTFILE
print "Model name:\t"+MODEL_NAME
print "Image size:\t"+str(SIZE)
print "Frame rate:\t"+str(SAMPLING)
print "Model dir:\t"+str(MODEL_DIR)
print "***********************************"

sys.path.append(PATH_TO_TF_MODELS+"/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

def detect(image_path, sess, origin_path, output):
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = cv2.imread(image_path)
      image_np = image_np[...,::-1]

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)

      # Actual detection.
      start_time = time.time()
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
#      print("found "+str(len(boxes[0]))+" objects in "+str(time.time()-start_time)+" seconds\n")

      output.write("FrameID="+ntpath.basename(origin_path)+"\n")
      output.write(origin_path+": "+str(time.time()-start_time)+" seconds\n")
      for j in range(len(boxes[0])):
        if scores[0][j] > 0.1:
          minX=boxes[0][j][1]*1920
          minY=boxes[0][j][0]*1200
          w = boxes[0][j][3]*1920-boxes[0][j][1]*1920
          h = boxes[0][j][2]*1200-boxes[0][j][0]*1200
#          print category_index[classes[0][j]]['name']+": "+str(scores[0][j]*100)+"%"+ \
#                       "\t"+str(boxes[0][j][0]*1200)+"\t"+str(boxes[0][j][1]*1920)+ \
#                       "\t"+str(boxes[0][j][2]*1200)+"\t"+str(boxes[0][j][3]*1920)
          print "=== "+category_index[classes[0][j]]['name']+": "+str(scores[0][j]*100)+"%"+ \
                       "\t"+str(minX)+"\t"+str(minY)+ \
                       "\t"+str(minX+w)+"\t"+str(minY+h)
          output.write(category_index[classes[0][j]]['name']+ \
                       ": "+str("{:.2f}".format(scores[0][j]*100))+"%"+ \
                       "\t"+str("{:.6f}".format(minX/1920))+"\t"+str("{:.6f}".format(minY/1200))+ \
                       "\t"+str("{:.6f}".format(w/1920))+"\t"+str("{:.6f}".format(h/1200))+"\n")
      print(origin_path+" takes "+str(time.time()-start_time)+" seconds")

      # Visualization of the results of a detection.
      IMAGE_SIZE = (12, 8)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          min_score_thresh=.1,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show()


# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_CKPT = MODEL_DIR + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('models/research/object_detection/data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join(PATH_TO_TF_MODELS+"/research/object_detection/data", 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

'''
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
'''

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


TEST_IMAGE_PATHS = []
for filename in os.listdir(PATH_TO_TEST_IMAGES_DIR):
        if filename.endswith('.jpg'):
                TEST_IMAGE_PATHS.append(PATH_TO_TEST_IMAGES_DIR+filename)
TEST_IMAGE_PATHS.sort()

output = open(OUTPUTFILE, "w")
output.write("")
output.close()

output = open(OUTPUTFILE, "a")

sampling_interval = int(1.0/SAMPLING)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    count = 0
    for i in range(len(TEST_IMAGE_PATHS)):
      if i%sampling_interval != 10:
        continue
      if count % 50 == 0:
        print '======'+MODEL_NAME+': '+str(i)+'/'+str(len(TEST_IMAGE_PATHS))
      image_path = TEST_IMAGE_PATHS[i]
      #detect(image_path, sess)
      os.system('rm tmp.jpg')
      os.system('ffmpeg -loglevel quiet -i '+image_path+' -vf scale='+str(SIZE)+':-1 tmp.jpg ')
      detect('tmp.jpg', sess, image_path, output)
      count = count+1

