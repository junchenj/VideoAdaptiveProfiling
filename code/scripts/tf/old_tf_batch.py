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
from utils import label_map_util
from utils import visualization_utils as vis_util

model=sys.argv[1]
#PATH_TO_TEST_IMAGES_DIR = '/home/junchenj/crowdai/object-detection-crowdai-samples/'
PATH_TO_TEST_IMAGES_DIR = sys.argv[2]
OUTPUTFILE = sys.argv[3]

print "====== "+model+"\t"+OUTPUTFILE

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = model
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/junchenj/code/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


print 'TEST1'
detection_graph = tf.Graph()
with detection_graph.as_default():
  print 'TEST2'
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    print 'TEST3'
    serialized_graph = fid.read()
    print 'TEST4'
    od_graph_def.ParseFromString(serialized_graph)
    print 'TEST5'
    tf.import_graph_def(od_graph_def, name='')
    print 'TEST6'


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  
  start_time = time.time()
  tmp0 = image.getdata()
  print("W "+str(time.time()-start_time)+" seconds")
  start_time = time.time()
  tmp1 = np.array(tmp0)
  print("X "+str(time.time()-start_time)+" seconds")
  start_time = time.time()
  tmp2 = tmp1.reshape((im_height, im_width, 3))
  print("Y "+str(time.time()-start_time)+" seconds")
  start_time = time.time()
  tmp3 = tmp2.astype(np.uint8)
  print("Z "+str(time.time()-start_time)+" seconds")
  return tmp3
#  return np.array(image.getdata()).reshape(
#      (im_height, im_width, 3)).astype(np.uint8)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 5) ]
TEST_IMAGE_PATHS = []
for filename in os.listdir(PATH_TO_TEST_IMAGES_DIR):
	if filename.endswith('.jpg'):
		TEST_IMAGE_PATHS.append(PATH_TO_TEST_IMAGES_DIR+filename)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


output = open(OUTPUTFILE, "w")
output.write("")
output.close()

output = open(OUTPUTFILE, "a")

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
    for i in range(len(TEST_IMAGE_PATHS)):
      if i % 50 == 0:
        print '======'+model+': '+str(i)+'/'+str(len(TEST_IMAGE_PATHS))
      #if i%3==module: 
      #  continue
      image_path = TEST_IMAGE_PATHS[i]
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      start_time = time.time()
#      image = Image.open(image_path)
#      print("step-1 "+str(time.time()-start_time)+" seconds")
      
#      start_time = time.time()
#      image_np = load_image_into_numpy_array(image)
      image_np = cv2.imread(image_path)
      image_np = image_np[...,::-1]
#      print("step-2 "+str(time.time()-start_time)+" seconds")

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      start_time = time.time()
      image_np_expanded = np.expand_dims(image_np, axis=0)
#      print("step-3 "+str(time.time()-start_time)+" seconds")

      # Actual detection.
#      start_time = time.time()
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
#      print("found "+str(len(boxes[0]))+" objects in "+str(time.time()-start_time)+" seconds\n")

      output.write("FrameID="+ntpath.basename(image_path)+"\n")
      output.write(image_path+": "+str(time.time()-start_time)+" seconds\n")
      for j in range(len(boxes[0])):
        if scores[0][j] > 0.1:
          minX=boxes[0][j][1]*1920
          minY=boxes[0][j][0]*1200
          w = boxes[0][j][3]*1920-boxes[0][j][1]*1920
          h = boxes[0][j][2]*1200-boxes[0][j][0]*1200
#          print category_index[classes[0][j]]['name']+": "+str(scores[0][j]*100)+"%"+ \
#                       "\t"+str(boxes[0][j][0]*1200)+"\t"+str(boxes[0][j][1]*1920)+ \
#                       "\t"+str(boxes[0][j][2]*1200)+"\t"+str(boxes[0][j][3]*1920)
#          print "=== "+category_index[classes[0][j]]['name']+": "+str(scores[0][j]*100)+"%"+ \
#                       "\t"+str(minX)+"\t"+str(minY)+ \
#                       "\t"+str(minX+w)+"\t"+str(minY+h)
#          print "### "+category_index[classes[0][j]]['name']+": "+str(scores[0][j]*100)+"%"+ \
#                       "\t"+str(minX)+"\t"+str(minY)+ \
#                       "\t"+str(w)+"\t"+str(h)
#          output.write(category_index[classes[0][j]]['name']+": "+str(scores[0][j]*100)+"%"+ \
#                       "\t"+str(boxes[0][j][0])+"\t"+str(boxes[0][j][1])+ \
#                       "\t"+str(boxes[0][j][2])+"\t"+str(boxes[0][j][3])+"\n")
          output.write(category_index[classes[0][j]]['name']+ \
                       ": "+str("{:.2f}".format(scores[0][j]*100))+"%"+ \
                       "\t"+str("{:.6f}".format(minX/1920))+"\t"+str("{:.6f}".format(minY/1200))+ \
                       "\t"+str("{:.6f}".format(w/1920))+"\t"+str("{:.6f}".format(h/1200))+"\n")
      print(image_path+" takes "+str(time.time()-start_time)+" seconds")

'''
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          min_score_thresh=.5,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show()
'''

output.close()







