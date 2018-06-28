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
import sys, getopt, os, argparse
import ntpath

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


def detect(image, sess,
           detection_boxes, detection_scores, detection_classes, num_detections, image_tensor):
    
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
    print(" takes "+str(runtime)+" seconds")
    # Visualization of the results of a detection.
    '''
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        min_score_thresh=.5,
        use_normalized_coordinates=True,
        line_thickness=8)
    if show_frames:
        im = plt.imshow(image)
        plt.show(block=False)
        f.canvas.draw()
    '''
    return boxes, scores, classes, num, runtime
    '''
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
    return predictions, runtime
    '''

def detect_cropping(image, sess, target_size, model_default_size,
                    detection_boxes, detection_scores, detection_classes, num_detections, image_tensor):
    boxes_global = None
    scores_global = None
    classes_global = None
    num_global = 0
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
            image_cropped = image_resized[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
            #predictions, runtime = detect(image_cropped, sess)
            start_time = time.time()
            boxes, scores, classes, num, runtime = detect(image_cropped, sess,
                                                          detection_boxes, detection_scores,
                                                          detection_classes, num_detections, image_tensor)
            print("detection takes "+str(time.time()-start_time)+" seconds")
            '''
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
            '''
            for j in range(len(boxes[0])):
                boxes[0][j][1] = (boxes[0][j][1]*w_crop+x_crop)/new_w
                boxes[0][j][3] = (boxes[0][j][3]*w_crop+x_crop)/new_w
                boxes[0][j][0] = (boxes[0][j][0]*h_crop+y_crop)/new_h
                boxes[0][j][2] = (boxes[0][j][2]*h_crop+y_crop)/new_h
            if boxes_global is None:
                boxes_global = boxes
            else:
                boxes_global = np.concatenate((boxes_global, boxes), axis=1)
            if scores_global is None:
                scores_global = scores
            else:
                scores_global = np.concatenate((scores_global, scores), axis=1)
            if classes_global is None:
                classes_global = classes
            else:
                classes_global = np.concatenate((classes_global, classes), axis=1)
            num_global = num_global+num
            runtime_global = runtime_global+runtime
    #return predictions_global, runtime_global
    return boxes_global, scores_global, classes_global, num_global, runtime_global


def main(PATH_TO_TF_MODELS, PATH_TO_TEST_IMAGES_DIR, OUTPUT_FILE,
         MODEL_PATH, MODEL_NAME, SIZE, SAMPLING,
         save_frames, SAVE_TO_FRAMES_PATH, show_frames,
         detection_graph = None,
         sess = None):
    
    print "***********************************"
    print "TFModel path:\t"+PATH_TO_TF_MODELS
    print "Input images:\t"+PATH_TO_TEST_IMAGES_DIR
    print "Output file:\t"+OUTPUT_FILE
    print "Model path:\t"+MODEL_PATH
    print "Model name:\t"+MODEL_NAME
    print "Image size:\t"+str(SIZE)
    print "Frame rate:\t"+str(SAMPLING)
    print "Save frames:\t"+str(save_frames)
    print "Save frames to:\t"+str(SAVE_TO_FRAMES_PATH)
    print "Show frames:\t"+str(show_frames)
    print "Graph preloaded:\t"+str(detection_graph)
    print "Sess preloaded:\t"+str(sess)
    print "***********************************"
    
    sys.path.append(PATH_TO_TF_MODELS+"/research/object_detection")
    
    from utils import label_map_util
    from utils import visualization_utils as vis_util
    import collections
    
    if show_frames:
        IMAGE_SIZE = (15, 10)
        f = plt.figure(figsize=IMAGE_SIZE)

    # What model to download.
    #MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # List of the strings that is used to add correct label for each box.
    #PATH_TO_LABELS = os.path.join('models/research/object_detection/data', 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = os.path.join(PATH_TO_TF_MODELS+"/research/object_detection/data", 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    #PATH_TO_CKPT = '/home/junchenj/VideoAdaptiveProfiling/code/scripts/tf/resized_'+str(SIZE)+'/'+ 'frozen_inference_graph.pb'

    '''
    NEW_CKPT_DIR = "ResizedModel_"+MODEL_NAME+"_"+str(SIZE)
    if (os.path.isdir(NEW_CKPT_DIR)):
        print "================== Config already exist ================"
    else:
        print "================== Creating new config ================"
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        #PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        #opener = urllib.request.URLopener()
        #opener.retrieve(DOWNLOAD_BASE + MODEL_NAME + '.tar.gz', MODEL_FILE)
        #tar_file = tarfile.open(MODEL_FILE)
        #for file in tar_file.getmembers():
        #  file_name = os.path.basename(file.name)
        #  tar_file.extract(file, os.getcwd())
        #  if 'frozen_inference_graph.pb' in file_name:
        #    tar_file.extract(file, os.getcwd())
        
        #os.system("sh create_bp.sh "+str(SIZE)+" "+NEW_CKPT_DIR+" "+MODEL_NAME)
        #os.system("rm -r "+MODEL_NAME+"/")
        
        os.system("cp -r /mnt/detectors/"+MODEL_NAME+" "+NEW_CKPT_DIR)
        print "================== Created new config ================"
    PATH_TO_CKPT = NEW_CKPT_DIR + '/frozen_inference_graph.pb'
    '''
    PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'

    if detection_graph is None:
        timestamp = time.time()
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        print("loading the graph takes "+str(time.time()-timestamp)+" seconds")

    timestamp = time.time()
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print("loading the labelmap takes "+str(time.time()-timestamp)+" seconds")

    TEST_IMAGE_PATHS = []
    for filename in os.listdir(PATH_TO_TEST_IMAGES_DIR):
            if filename.endswith('.jpg'):
                    TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, filename))
    TEST_IMAGE_PATHS.sort()

    output = open(OUTPUT_FILE, "w")
    output.write("")
    output.close()

    output = open(OUTPUT_FILE, "a")

    sampling_interval = int(1.0/SAMPLING)

    timestamp = time.time()

#    with detection_graph.as_default():
#        with tf.Session(graph=detection_graph) as sess:
#    sess = tf.Session(graph=detection_graph)
    print("preparing session takes "+str(time.time()-timestamp)+" seconds")
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
        #if count % 50 == 0:
        print '======'+MODEL_NAME+': '+str(i)+'/'+str(len(TEST_IMAGE_PATHS))
        image_path = TEST_IMAGE_PATHS[i]
        print image_path
        image = cv2.imread(image_path)
        image = image[...,::-1]
        img_height, img_width = image.shape[:2]
        target_size = min(img_width, SIZE)
        #model_default_size = 400
        model_default_size = target_size
        #predictions, runtime = detect_cropping(image, sess, target_size, model_default_size)
        boxes, scores, classes, num, runtime = detect_cropping(image, sess, target_size, model_default_size,
                                                               detection_boxes, detection_scores,
                                                               detection_classes, num_detections, image_tensor)
        output.write("FrameID="+ntpath.basename(image_path)+"\n")
        output.write(image_path+": "+str(runtime)+" seconds\n")
        for j in range(len(boxes[0])):
            if scores[0][j] < 0.1:
                continue
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
            output.write(name+": "+"{:.2f}".format(pr*100)+"%"+ \
                "\t"+"{:.6f}".format(x)+"\t"+"{:.6f}".format(y)+ \
                "\t"+"{:.6f}".format(w)+"\t"+"{:.6f}".format(h)+"\n")
        if save_frames or show_frames:
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(image,
                                                               np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               category_index,
                                                               min_score_thresh=.5,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)
            if save_frames:
                cv2.imwrite(SAVE_TO_FRAMES_PATH+"{0:0>3}".format(i)+".png", image)
            if show_frames:
                im = plt.imshow(image)
                plt.show(block=False)
                f.canvas.draw()
        '''
        IMAGE_SIZE = (9, 6)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            max_boxes_to_draw=1000000,
            min_score_thresh=.5,
            use_normalized_coordinates=True,
            line_thickness=4)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image)
        plt.title(MODEL_NAME)
        plt.show()
        #os.system("mkdir /home/junchenj/workspace/tmp_resize_"+str(target_size)+"/")
        #new_image_path = "/home/junchenj/workspace/tmp_resize_"+str(target_size)+"/"+str(i)+".jpg"
        #print new_image_path
        #cv2.imwrite(new_image_path, image)
        '''
    #count = count+1


if __name__ == '__main__':
    PATH_TO_TF_MODELS = None
    PATH_TO_TEST_IMAGES_DIR = None
    OUTPUT_FILE = None
    MODEL_PATH = None
    MODEL_NAME = None
    SAMPLING = None
    SIZE = None
    save_frames = False
    SAVE_TO_FRAMES_PATH = None
    show_frames = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfpath")
    parser.add_argument("--frames")
    parser.add_argument("--output")
    parser.add_argument("--modelpath")
    parser.add_argument("--modelname")
    parser.add_argument("--framerate")
    parser.add_argument("--size")
    parser.add_argument("--saveframes")
    parser.add_argument("--saveframespath")
    parser.add_argument("--showframes")
    args = parser.parse_args()
    PATH_TO_TF_MODELS = args.tfpath
    PATH_TO_TEST_IMAGES_DIR = args.frames
    OUTPUT_FILE = args.output
    MODEL_PATH = args.modelpath
    MODEL_NAME = args.modelname
    SAMPLING = float(args.framerate)
    SIZE = int(args.size)
    if args.saveframes == "True": save_frames = True
    SAVE_TO_FRAMES_PATH = args.saveframespath
    if args.showframes == "True": show_frames = True
    if (PATH_TO_TF_MODELS is None):
        print("Missing arguments --tfpath")
        sys.exit()
    if (PATH_TO_TEST_IMAGES_DIR is None):
        print("Missing arguments --frames")
        sys.exit()
    if (OUTPUT_FILE is None):
        print("Missing arguments --output")
        sys.exit()
    if (MODEL_PATH is None):
        print("Missing arguments --modelpath")
        sys.exit()
    if (MODEL_NAME is None):
        print("Missing arguments --modelname")
        sys.exit()
    if (SAMPLING is None):
        print("Missing arguments --framerate")
        sys.exit()
    if (SIZE is None):
        print("Missing arguments --size")
        sys.exit()

    main(PATH_TO_TF_MODELS, PATH_TO_TEST_IMAGES_DIR, OUTPUT_FILE,
         MODEL_PATH, MODEL_NAME, SIZE, SAMPLING,
         save_frames, SAVE_TO_FRAMES_PATH, show_frames)




