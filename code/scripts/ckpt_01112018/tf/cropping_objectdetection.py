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
import sys, getopt, os
import ntpath

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

PATH_TO_TF_MODELS = None
PATH_TO_TEST_IMAGES_DIR = None
OUTPUT_FILE = None
MODEL_NAME = None
SAMPLING = None
SIZE = None

opts, args = getopt.getopt(sys.argv[1:],"t:i:o:m:r:s:")
for o, a in opts:
    if o == '-t':
        PATH_TO_TF_MODELS = a
    elif o == '-i':
        PATH_TO_TEST_IMAGES_DIR = a
    elif o == '-o':
        OUTPUT_FILE = a
    elif o == '-m':
        MODEL_NAME = a
    elif o == '-r':
        SAMPLING = float(a)
    elif o == '-s':
        SIZE = int(a)
    else:
        print("Usage: %s -t path_tf_model -i images -o output -m model -r sampling -s size" % sys.argv[0])
        sys.exit()
if (PATH_TO_TF_MODELS is None):
    print("Missing arguments -t")
    sys.exit()
if (PATH_TO_TEST_IMAGES_DIR is None):
    print("Missing arguments -i")
    sys.exit()
if (OUTPUT_FILE is None):
    print("Missing arguments -o")
    sys.exit()
if (MODEL_NAME is None):
    print("Missing arguments -m")
    sys.exit()
if (SAMPLING is None):
    print("Missing arguments -r")
    sys.exit()
if (SIZE is None):
    print("Missing arguments -s")
    sys.exit()


print "***********************************"
print "TFModel path:\t"+PATH_TO_TF_MODELS
print "Input images:\t"+PATH_TO_TEST_IMAGES_DIR
print "Output file:\t"+OUTPUT_FILE
print "Model name:\t"+MODEL_NAME
print "Image size:\t"+str(SIZE)
print "Frame rate:\t"+str(SAMPLING)
print "***********************************"

sys.path.append(PATH_TO_TF_MODELS+"/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

def detect(image, sess):
    predictions = []
    runtime = 0
    img_height, img_width = image.shape[:2]

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_expanded = np.expand_dims(image, axis=0)

    # Actual detection.
    start_time = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
#    print("found "+str(len(boxes[0]))+" objects in "+str(time.time()-start_time)+" seconds\n")

    runtime = time.time()-start_time
    for j in range(len(boxes[0])):
        if scores[0][j] > 0.1:
            minX=boxes[0][j][1]*img_width
            minY=boxes[0][j][0]*img_height
            w = boxes[0][j][3]*img_width-boxes[0][j][1]*img_width
            h = boxes[0][j][2]*img_height-boxes[0][j][0]*img_height
            name = category_index[classes[0][j]]['name']
            pr = scores[0][j]
            x = minX/img_width
            y = minY/img_height
            w = w/img_width
            h = h/img_height
            prediction = (name, pr, x, y, w, h)
            predictions.append(prediction)
    print(" takes "+str(runtime)+" seconds")
    return predictions, runtime

'''
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image,
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

def detect_cropping(image, sess, target_size, model_default_size):
    predictions_global = []
    runtime_global = 0
    img_height, img_width = image.shape[:2]
    new_w = target_size
    new_h = int(float(target_size)*float(img_height)/float(img_width))
    default_w = model_default_size
    default_h = int(float(model_default_size)*float(img_height)/float(img_width))

    image_resized = cv2.resize(image, (new_w, new_h))
    step_len_w = int(default_w/1)
    step_len_h = int(default_h/1)
    num_crops_w = int(new_w-default_w)/step_len_w
    num_crops_h = int(new_h-default_h)/step_len_h
    for i in range(num_crops_w+1):
        for j in range(num_crops_h+1):
            x_crop = i*step_len_w
            y_crop = j*step_len_h
            w_crop = default_w
            h_crop = default_h
            #print str(x_crop)+", "+str(y_crop)+", "+str(w_crop)+", "+str(h_crop)
            #print "resize:\t"+str(image_resized.shape)
            image_cropped = image_resized[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
            #print "croped:\t"+str(image_cropped.shape)
            predictions, runtime = detect(image_cropped, sess)
            for prediction in predictions:
                name = prediction[0]
                pr = prediction[1]
                x = prediction[2]
                y = prediction[3]
                w = prediction[4]
                h = prediction[5]
                x_global = (x*w_crop+x_crop)/new_w
                y_global = (y*h_crop+y_crop)/new_h
                w_global = w*w_crop/new_w
                h_global = h*h_crop/new_h
                prediction_global = (name, pr, x_global, y_global, w_global, h_global)
                predictions_global.append(prediction_global)
                runtime_global = runtime_global+runtime
    return predictions_global, runtime_global


# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('models/research/object_detection/data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join(PATH_TO_TF_MODELS+"/research/object_detection/data", 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

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

output = open(OUTPUT_FILE, "w")
output.write("")
output.close()

output = open(OUTPUT_FILE, "a")

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
            if i%sampling_interval != 0:
                continue
            if count % 50 == 0:
                print '======'+MODEL_NAME+': '+str(i)+'/'+str(len(TEST_IMAGE_PATHS))
            image_path = TEST_IMAGE_PATHS[i]
            image = cv2.imread(image_path)
            image = image[...,::-1]
            height, width = image.shape[:2]
            target_size = min(width, SIZE)
            model_default_size = 400
            predictions, runtime = detect_cropping(image, sess, target_size, model_default_size)
            output.write("FrameID="+ntpath.basename(image_path)+"\n")
            output.write(image_path+": "+str(runtime)+" seconds\n")
            for prediction in predictions:
                name = prediction[0]
                pr = prediction[1]
                x = prediction[2]
                y = prediction[3]
                w = prediction[4]
                h = prediction[5]
                output.write(name+": "+"{:.2f}".format(pr*100)+"%"+ \
                    "\t"+"{:.6f}".format(x)+"\t"+"{:.6f}".format(y)+ \
                    "\t"+"{:.6f}".format(w)+"\t"+"{:.6f}".format(h)+"\n")

        count = count+1

