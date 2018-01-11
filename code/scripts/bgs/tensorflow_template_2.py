import cv2
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
import tensorflow as tf

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

import time


##############


from datasets import dataset_utils

url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
checkpoints_dir = '/tmp/checkpoints'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

###############

image_size = inception.inception_v1.default_image_size

#with tf.Graph().as_default():
with tf.Session(graph=tf.Graph()) as sess:
  with slim.arg_scope(inception.inception_v1_arg_scope()):
    tf.Graph().as_default()
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    #image_string = urllib.urlopen(url).read()

    names = imagenet.create_readable_names_for_imagenet_labels()
    
    start_time = time.time()
    image_string = open("/home/junchenj/workspace/scripts/bgs/output_test/images/out-000002_c_0.jpg", 'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    print("step 1: "+str(time.time()-start_time)+" seconds")

    start_time = time.time()
    # Create the model, use the default arg scope to configure the batch norm parameters.
    #with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    print("step 2: "+str(time.time()-start_time)+" seconds")
    
    start_time = time.time()
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
        slim.get_model_variables('InceptionV1'))
    init_fn(sess)
    print("step X: "+str(time.time()-start_time)+" seconds")
    
    start_time = time.time()
    np_image, probabilities = sess.run([image, probabilities])
    probabilities = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    print("step 3: "+str(time.time()-start_time)+" seconds")
    
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))

    for k in range(5):
        start_time = time.time()
        image_string = open("/home/junchenj/workspace/scripts/bgs/output_test/images/out-000010_c_"+str(k)+".jpg", 'rb').read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        print image.shape
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        local_matrix  = tf.expand_dims(processed_image, 0)
        processed_images = tf.concat([processed_images, local_matrix], 0)
        print processed_images.shape

    start_time = time.time()
    # Create the model, use the default arg scope to configure the batch norm parameters.
    #with slim.arg_scope(inception.inception_v1_arg_scope()):
    logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False, reuse=True)
    print("step 2: "+str(time.time()-start_time)+" seconds")
    probabilities = tf.nn.softmax(logits)
    
    print probabilities.shape
    print type(image)
    start_time = time.time()
    np_image, probabilities = sess.run([image, probabilities])
    print("step 3: "+str(time.time()-start_time)+" seconds")
    print probabilities.shape
    
    for k in range(6):
        prob = probabilities[k, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-prob), key=lambda x:x[1])]
        print "--------- image "+str(k)
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (prob[index] * 100, names[index]))

'''
    for k in range(10):
        start_time = time.time()
        image_string = open("/home/junchenj/workspace/scripts/bgs/output_test/images/out-000002_c_"+str(k)+".jpg", 'rb').read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        print processed_image.shape
        processed_images  = tf.expand_dims(processed_image, 0)
        print processed_images.shape
        print type(processed_images)
        print "step 1: "+str(time.time()-start_time)+" seconds"
    
        start_time = time.time()
        # Create the model, use the default arg scope to configure the batch norm parameters.
        #with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False, reuse=True)
        print("step 2: "+str(time.time()-start_time)+" seconds")
        probabilities = tf.nn.softmax(logits)
        
        start_time = time.time()
        np_image, probabilities = sess.run([image, probabilities])
        print("step 3: "+str(time.time()-start_time)+" seconds")
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
    
        for i in range(5):
            index = sorted_inds[i]
            print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))
'''





