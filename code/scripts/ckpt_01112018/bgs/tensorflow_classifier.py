#import matplotlib
#import matplotlib.pyplot as plt

import cv2
print("CV2 version: "+cv2.__version__)
import numpy as np

frame = cv2.imread("/home/junchenj/workspace/scripts/frames/out-000001.jpg")
print frame.shape


import os
import tensorflow as tf
try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from nets import resnet_v1
from nets import mobilenet_v1
from nets import vgg
from nets import inception_resnet_v2
from nets.nasnet import nasnet
from nets import nets_factory
from preprocessing import vgg_preprocessing
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

from datasets import dataset_utils

import sys, getopt, os
import ntpath
import time
import matplotlib.pyplot as plt
import PIL.Image
import StringIO
from six.moves import urllib
import tarfile

EXTRACTION_LOG = ""
IMAGES_FOLDER = ""
OUTPUT_FILE = ""
MODEL = ""

opts, args = getopt.getopt(sys.argv[1:],"e:i:o:m:")
for o, a in opts:
    if o == '-e':
        EXTRACTION_LOG = a
    elif o == '-i':
        IMAGES_FOLDER = a
    elif o == '-o':
        OUTPUT_FILE = a
    elif o == '-m':
        MODEL = a
    else:
        print("Usage: %s -e extraction -i images -o output -m model" % sys.argv[0])
        sys.exit()
if (not EXTRACTION_LOG):
    print("Missing arguments -e")
    sys.exit()
if (not IMAGES_FOLDER):
    print("Missing arguments -i")
    sys.exit()
if (not OUTPUT_FILE):
    print("Missing arguments -o")
    sys.exit()
if (not MODEL):
    print("Missing arguments -m")
    sys.exit()

print "***********************************"
print "Extraction:\t"+EXTRACTION_LOG
print "Images path:\t"+IMAGES_FOLDER
print "Output file:\t"+OUTPUT_FILE
print "Model name:\t"+MODEL
print "***********************************"

model_to_url = {
                'inception_v1':                 'inception_v1_2016_08_28', \
                'inception_v2':                 'inception_v2_2016_08_28', \
                'inception_v3':                 'inception_v3_2016_08_28', \
                'inception_v4':                 'inception_v4_2016_09_09', \
                'resnet_v1_50':                 'resnet_v1_50_2016_08_28', \
                'resnet_v1_101':                'resnet_v1_101_2016_08_28', \
                'resnet_v1_152':                'resnet_v1_152_2016_08_28', \
                'mobilenet_v1_025':             'mobilenet_v1_0.25_128_2017_06_14', \
                'mobilenet_v1_050':             'mobilenet_v1_0.50_160_2017_06_14', \
                'mobilenet_v1':                 'mobilenet_v1_1.0_224_2017_06_14', \
                'inception_resnet_v2':          'inception_resnet_v2_2016_08_30', \
                'nasnet_mobile':                'nasnet-a_mobile_04_10_2017', \
                'nasnet_large':                 'nasnet-a_large_04_10_2017', \
                'vgg_16':                       'vgg_16_2016_08_28'}
model_to_image_size = {
                'inception_v1':                 inception.inception_v1.default_image_size, \
                'inception_v2':                 inception.inception_v2.default_image_size, \
                'inception_v3':                 inception.inception_v3.default_image_size, \
                'inception_v4':                 inception.inception_v3.default_image_size, \
                'resnet_v1_50':                 resnet_v1.resnet_v1_50.default_image_size, \
                'resnet_v1_101':                resnet_v1.resnet_v1_101.default_image_size, \
                'resnet_v1_152':                resnet_v1.resnet_v1_152.default_image_size, \
                'mobilenet_v1_025':             mobilenet_v1.mobilenet_v1.default_image_size, \
                'mobilenet_v1_050':             mobilenet_v1.mobilenet_v1.default_image_size, \
                'mobilenet_v1':                 mobilenet_v1.mobilenet_v1.default_image_size, \
                'inception_resnet_v2':          inception_resnet_v2.inception_resnet_v2.default_image_size, \
                'nasnet_mobile':                nasnet.build_nasnet_mobile.default_image_size, \
                'nasnet_large':                 nasnet.build_nasnet_large.default_image_size, \
                'vgg_16':                       vgg.vgg_16.default_image_size}
model_to_batch_size = {
                'inception_v1':                 400, \
                'inception_v2':                 400, \
                'inception_v3':                 400, \
                'inception_v4':                 400, \
                'resnet_v1_50':                 400, \
                'resnet_v1_101':                400, \
                'resnet_v1_152':                400, \
                'mobilenet_v1_025':             400, \
                'mobilenet_v1_050':             400, \
                'mobilenet_v1':                 400, \
                'inception_resnet_v2':          400, \
                'nasnet_mobile':                400, \
                'nasnet_large':                 200, \
                'vgg_16':                       100}
model_to_num_classes = {
                'inception_v1':                 1001, \
                'inception_v2':                 1001, \
                'inception_v3':                 1001, \
                'inception_v4':                 1001, \
                'resnet_v1_50':                 1000, \
                'resnet_v1_101':                1000, \
                'resnet_v1_152':                1000, \
                'mobilenet_v1_025':             1001, \
                'mobilenet_v1_050':             1001, \
                'mobilenet_v1':                 1001, \
                'inception_resnet_v2':          1001, \
                'nasnet_mobile':                1001, \
                'nasnet_large':                 1001, \
                'vgg_16':                       1000}
model_to_checkpoint_name = {
                'inception_v1':                 'inception_v1.ckpt', \
                'inception_v2':                 'inception_v2.ckpt', \
                'inception_v3':                 'inception_v3.ckpt', \
                'inception_v4':                 'inception_v4.ckpt', \
                'resnet_v1_50':                 'resnet_v1_50.ckpt', \
                'resnet_v1_101':                'resnet_v1_101.ckpt', \
                'resnet_v1_152':                'resnet_v1_152.ckpt', \
                'mobilenet_v1_025':             'mobilenet_v1_0.25_128.ckpt', \
                'mobilenet_v1_050':             'mobilenet_v1_0.50_160.ckpt', \
                'mobilenet_v1':                 'mobilenet_v1_1.0_224.ckpt', \
                'inception_resnet_v2':          'inception_resnet_v2_2016_08_30.ckpt', \
                'nasnet_mobile':                'model.ckpt', \
                'nasnet_large':                 'model.ckpt', \
                'vgg_16':                       'vgg_16.ckpt'}



def preprocess_for_inception(image, height, width, sess,
                        central_fraction=0.875, scope=None, debug=False):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if debug:
      plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(image, tf.uint8))))))
      print "test 1"; plt.show()
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if debug:
      plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(tf.multiply(image, 255), tf.uint8))))))
      print "test 2"; plt.show()
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)
    if debug:
      plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(tf.multiply(image, 255), tf.uint8))))))
      print "test 3"; plt.show()

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    if debug:
      plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(tf.multiply(image, 255), tf.uint8))))))
      print "test 4"; plt.show()
    image = tf.subtract(image, 0.5)
    if debug:
      plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(tf.multiply(image, 255), tf.uint8))))))
      print "test 5"; plt.show()
    image = tf.multiply(image, 2.0)
    if debug:
      plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(tf.multiply(image, 255), tf.uint8))))))
      print "test 6"; plt.show()
    return image

def preprocess_for_vgg(image, output_height, output_width, resize_side, sess, debug=False):
  image = vgg_preprocessing._aspect_preserving_resize(image, resize_side)
  if debug:
    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(image, tf.uint8))))))
    print "test 1"; plt.show()
#  image = vgg_preprocessing._central_crop([image], output_height, output_width)[0]
  if debug:
    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(image, tf.uint8))))))
    print "test 2"; plt.show()
  image.set_shape([output_height, output_width, 3])
  if debug:
    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(image, tf.uint8))))))
    print "test 3"; plt.show()
  image = tf.to_float(image)
  if debug:
    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(image, tf.uint8))))))
    print "test 4"; plt.show()
  image = vgg_preprocessing._mean_image_subtraction(image, [vgg_preprocessing._R_MEAN, vgg_preprocessing._G_MEAN, vgg_preprocessing._B_MEAN])
  if debug:
    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(tf.image.encode_jpeg(tf.cast(image, tf.uint8))))))
    print "test 5"; plt.show()
  return image

def prediction_basic(processed_images, model, frame, sess):
    start_time = time.time()
    num_classes = model_to_num_classes[model]
    network_fn = nets_factory.get_network_fn(model, num_classes, is_training=False)
    logits, _ = network_fn(processed_images)
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, model_to_checkpoint_name[model]),
        slim.get_model_variables())
    print "Prediction2.1: "+str(time.time()-start_time)+" seconds"
    start_time = time.time()
    init_fn(sess)
    print "Prediction2.2: "+str(time.time()-start_time)+" seconds"
    probabilities = tf.nn.softmax(logits)

    start_time = time.time()
    np_image, probabilities = sess.run([frame, probabilities])
    runtime = time.time()-start_time
    print "Prediction: "+str(runtime)+" seconds"
    return probabilities, runtime

_prediction_debug = False
def batch_prediction(frame_id_to_path, frame_id_to_image_ids, image_id_to_coordinates, model, sess,
                    debug=_prediction_debug):
    print "batch processing: "+str(len(image_id_to_coordinates))
    if model == 'inception_v1' or model == 'inception_v2' or \
            model == 'inception_v3' or model == 'inception_v4' or \
            model == 'mobilenet_v1_025' or model == 'mobilenet_v1_050' or model == 'mobilenet_v1' or \
            model == 'inception_resnet_v2' or model == 'nasnet_mobile' or model == 'nasnet_large':
        preprocessing_type = 'inception'
    elif model == 'vgg_16' or model == 'resnet_v1_50' or model == 'resnet_v1_101' or model == 'resnet_v1_152':
        preprocessing_type = 'vgg'
    image_id_to_predictions = {}
    image_ids = []
    count = 0
    start_time_1 = time.time()
    image_size = model_to_image_size[model]
    for frame_id, path in frame_id_to_path.iteritems():
        frame_string = open(path, 'rb').read()
        frame = tf.image.decode_jpeg(frame_string, channels=3)
        frame_np = cv2.imread(path, cv2.IMREAD_COLOR)
        frame_height, frame_width = frame_np.shape[:2]
        if preprocessing_type == 'inception':
            processed_frame = preprocess_for_inception(frame, frame_height, frame_width, sess, \
                                                       central_fraction=1.0, debug=_prediction_debug)
        elif preprocessing_type == 'vgg':
            processed_frame = preprocess_for_vgg(frame, frame_height, frame_width, frame_height, \
                                                 sess, debug=_prediction_debug)
        start_time = time.time()
        height, width = processed_frame.shape[:2].as_list()
        for image_id in frame_id_to_image_ids[frame_id]:
            fields = image_id_to_coordinates[image_id].split('\t')
            x = int(width*float(fields[0]))
            y = int(height*float(fields[1]))
            w = int(width*float(fields[2]))
            h = int(height*float(fields[3]))
            processed_image = tf.image.crop_to_bounding_box(processed_frame, y, x, h, w)
            if debug:
                print "object at "+str(fields)
                print str(x)+", "+str(y)+", "+str(w)+", "+str(h)+", "+str(frame_height-y-h)
                if preprocessing_type == 'vgg':
                    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(\
                                tf.image.encode_jpeg(tf.cast(processed_image, tf.uint8))))))
                elif preprocessing_type == 'inception':
                    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(\
                                tf.image.encode_jpeg(tf.cast(tf.multiply(processed_image, 255), tf.uint8))))))
                plt.show()
            processed_image = tf.image.resize_images(processed_image, (image_size, image_size))
            if debug:
                print "resized"
                if preprocessing_type == 'vgg':
                    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(\
                                tf.image.encode_jpeg(tf.cast(processed_image, tf.uint8))))))
                elif preprocessing_type == 'inception':
                    plt.imshow(PIL.Image.open(StringIO.StringIO(sess.run(\
                                tf.image.encode_jpeg(tf.cast(tf.multiply(processed_image, 255), tf.uint8))))))
                plt.show()
            if count == 0:
                processed_images = tf.expand_dims(processed_image, 0)
            else:
                local_matrix = tf.expand_dims(processed_image, 0)
                processed_images = tf.concat([processed_images, local_matrix], 0)
            image_ids.append(image_id)
            count = count+1
    print "Preparation: "+str(time.time()-start_time_1)+" seconds"
    probabilities, runtime = prediction_basic(processed_images, model, frame, sess)
    for k in range(len(image_ids)):
        image_id = image_ids[k]
        predictions = []
        prob = probabilities[k, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-prob), key=lambda x:x[1])]
        for i in range(5):
            index = sorted_inds[i]
            if model == 'resnet_v1_50' or model == 'resnet_v1_101' or model == 'resnet_v1_152' or \
                model == 'vgg_16':
                name = names[index+1]
            else:
                name = names[index]
            pr = prob[index]
            pair = (name, pr)
            predictions.append(pair)
        image_id_to_predictions[image_id] = predictions
    return image_id_to_predictions, runtime


def process_super_batch(frame_ids, frame_id_to_path, frame_id_to_image_ids, image_id_to_path, \
                        image_id_to_coordinates, output, model):
    tf.Graph().as_default()
    with tf.Session(graph=tf.Graph()) as sess:
        image_id_to_predictions, runtime = batch_prediction(frame_id_to_path, frame_id_to_image_ids,\
                                                            image_id_to_coordinates, model, sess)
        tf.reset_default_graph()
    for m in range(len(frame_ids)):
        frame_id = frame_ids[m]
        frame_path = frame_id_to_path[frame_id]
        output.write("FrameID="+frame_id+"\n")
        runtime_per_frame = float(runtime)/float(len(image_id_to_path))\
                            *float(len(frame_id_to_image_ids[frame_id]))
        output.write(frame_path+": "+str(runtime_per_frame)+" seconds"+"\n")
        for n in range(len(frame_id_to_image_ids[frame_id])):
            image_id = frame_id_to_image_ids[frame_id][n]
            coordinates = image_id_to_coordinates[image_id]
            predictions = image_id_to_predictions[image_id]
            name = predictions[0][0]
            prob = float(predictions[0][1])
            #if prob < 0.3:
            #    continue
            #output.write(image_id_to_path[image_id]+"\n")
            output.write(name+": "+"{0:.2f}".format(prob*100)+"%\t"+coordinates+"\n")
    

def process(lines, split_index_list, output_file, model):
    image_size = model_to_image_size[model]
    batch_size = model_to_batch_size[model]
    frame_ids = []
    frame_id_to_path = {}
    frame_id_to_image_ids = {}
    image_id_to_path = {}
    image_id_to_coordinates = {}
    for i in range(len(split_index_list)-1):
        frame_path = lines[split_index_list[i]].rstrip()
        frame_id = ntpath.basename(frame_path)
        frame_ids.append(frame_id)
        frame_id_to_path[frame_id] = frame_path
        num_images = split_index_list[i+1]-split_index_list[i]-1
        image_ids = []
        for j in range(num_images):
            line = lines[split_index_list[i]+j+1]
            fields = line.rstrip().split("\t")
            image_path = fields[0]
            image_id = ntpath.basename(image_path)
            coordinates = fields[1]+"\t"+fields[2]+"\t"+fields[3]+"\t"+fields[4]
            image_path = os.path.join(IMAGES_FOLDER,image_id)
            image_id_to_path[image_id] = image_path
            image_id_to_coordinates[image_id] = coordinates
            image_ids.append(image_id)
        frame_id_to_image_ids[frame_id] = image_ids
        if (len(image_id_to_path) < batch_size and i+1 < len(split_index_list)-1) or len(frame_ids) == 0:
            continue
        print frame_id
        output = open(output_file, "a")
        tf.Graph().as_default()
        process_super_batch(frame_ids, frame_id_to_path, frame_id_to_image_ids, image_id_to_path, \
                            image_id_to_coordinates, output, model)
        sys.stdout.flush()
        output.close()
        frame_ids = []
        frame_id_to_path = {}
        frame_id_to_image_ids = {}
        image_id_to_path = {}
        image_id_to_coordinates = {}

def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.
  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  def _progress(count, block_size, total_size):
    pass
  print 'Start downloading'
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  sys.stdout.flush()
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


log = open(EXTRACTION_LOG, "r")
split_index_list = []
lines = []
count = 0
for line in log:
    if len(line.split("\t")) == 1:
        split_index_list.append(count)
    lines.append(line)
    count += 1
split_index_list.append(count)

output = open(OUTPUT_FILE, "w")
output.write("")
output.close()

url = "http://download.tensorflow.org/models/"+model_to_url[MODEL]+".tar.gz"
checkpoints_dir = '/tmp/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

#dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)
download_and_uncompress_tarball(url, checkpoints_dir)
names = imagenet.create_readable_names_for_imagenet_labels()

global_start_time = time.time()

process(lines, split_index_list, OUTPUT_FILE, MODEL)
print "Prediction: "+str(time.time()-global_start_time)+" seconds"
sys.stdout.flush()

 
