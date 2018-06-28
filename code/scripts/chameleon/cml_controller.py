import sys
import os, shutil
import subprocess
import cml_inference
import tensorflow as tf
import time

frames_root = "/mnt/tmp/frames/"
new_model_root = "/mnt/tmp/models/"
output_root = "/mnt/tmp/output/"
analysis_root = "/mnt/tmp/analysis/"
save_images_root = "/mnt/tmp/saved_images/"

modelpath_to_graph = {}
modelpath_to_sess = {}

def cleanup_frames():
	try: shutil.rmtree(frames_root, ignore_errors=True)
	except OSError: print "Error deleting "+frames_root

def cleanup_models():
	try: shutil.rmtree(new_model_root, ignore_errors=True)
	except OSError: print "Error deleting "+frames_root

def cleanup_outputs():
    try: shutil.rmtree(output_root, ignore_errors=True)
    except OSError: print "Error deleting "+output_root

def cleanup_analysis():
    try: shutil.rmtree(analysis_root, ignore_errors=True)
    except OSError: print "Error deleting "+analysis_root

def cleanup_savedimages():
    try: shutil.rmtree(save_images_root, ignore_errors=True)
    except OSError: print "Error deleting "+save_images_root

def output_file_name(video_name, start_ms, end_ms, model, img_size, frame_rate):
    return output_root+"Result_Video_"+video_name+\
        "_From_"+"{0:0>9}".format(start_ms)+\
        "_To_"+"{0:0>9}".format(end_ms)+\
        "_Size_"+str(img_size)+"_Rate_"+str(frame_rate)+"_Model_"+model+".txt"

def prepare(path_to_video, video_name, start_ms, end_ms, model, img_size, frame_rate):
    old_model_path = "/mnt/detectors/"+model
    new_model_path = new_model_root+"Model_"+model+"_Size_"+str(img_size)+"/"
    frames_path = frames_root+"Frame_Video_"+video_name+\
                    "_From_"+"{0:0>9}".format(start_ms)+\
                    "_To_"+"{0:0>9}".format(end_ms)
    output_file = output_file_name(video_name, start_ms, end_ms,
                                   model, img_size, frame_rate)
    cmd = "sh cml_preprocess.sh --video "+path_to_video+" \
        --start "+str(start_ms)+" --end "+str(end_ms)+" \
        --frames "+frames_path+" \
        --model "+old_model_path+" \
        --newmodel "+new_model_path+" \
        --size "+str(img_size)+" \
        --framerate "+str(frame_rate)+" \
        --modelname "+model+" \
        --output "+output_file
    subprocess.call(cmd.split())

def inference(path_to_video, video_name, start_ms, end_ms,
              model, img_size, frame_rate,
              save_images, save_images_path, show_images):
    old_model_path = "/mnt/detectors/"+model
    new_model_path = new_model_root+"Model_"+model+"_Size_"+str(img_size)+"/"
    frames_path = frames_root+"Frame_Video_"+video_name+\
                "_From_"+"{0:0>9}".format(start_ms)+\
                "_To_"+"{0:0>9}".format(end_ms)
    output_file = output_file_name(video_name, start_ms, end_ms,
                                   model, img_size, frame_rate)
    #if new_model_path not in modelpath_to_graph:
    if new_model_path not in modelpath_to_sess:
        print "****************** Pre-Loading TF Graph..."
        print "****************** "+new_model_path+"\n"
        PATH_TO_CKPT = new_model_path + '/frozen_inference_graph.pb'
        timestamp = time.time()
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
        print("loading the graph takes "+str(time.time()-timestamp)+" seconds\n")
        modelpath_to_graph[new_model_path] = detection_graph
        modelpath_to_sess[new_model_path] = sess
    else:
        print "****************** TF Graph already pre-loaded!"
        print "****************** "+new_model_path+"\n"
    graph = modelpath_to_graph[new_model_path]
    sess = modelpath_to_sess[new_model_path]

    if not os.path.isfile(output_file):
        print "****************** Start actual inference and save results to..."
        print "****************** "+output_file+"\n"
        open(output_file, 'a').close()
        cml_inference.main("/home/junchenj/workspace/tensorflow/models/",
                           frames_path, output_file, new_model_path,
                           model, img_size, frame_rate,
                           save_images, save_images_path, show_images,
                           graph, sess)
    else:
        print "****************** Results already exist..."
        print "****************** "+output_file+"\n"
    print "****************** Done ****************** \n\n\n"

class detected_object:
    def __init__(self, x, y, w, h, label, prob):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._label = label
        self._prob = prob

class output_per_frame:
    def __init__(self, frame_name):
        self._name = frame_name
        self._obj_list = []
    def set_run_time(self, run_time):
        self._run_time = run_time
    def add_object(self, object):
        self._obj_list.append(object)

def parse_output_from_lines(lines):
    frameoutput_list = []
    output = None
    for line in lines:
        line = line.strip()
        if line.startswith('FrameID='):
            if output is not None:
                frameoutput_list.append(output)
            frame_name = line[len('FrameID='):-4]
            output = output_per_frame(frame_name)
        if line.endswith('seconds'):
            run_time = float(line[line.index('.jpg: ')+6:line.index(' seconds')])
            output.set_run_time(run_time)
        else:
            try: index1 = line.index(': ')
            except: continue
            label = line[:index1]
            remain = line[index1+len(': '):]
            fields = remain.split("\t")
            prob = float(fields[0][:-1])/100.0
            x = float(fields[1])
            y = float(fields[2])
            w = float(fields[3])
            h = float(fields[4])
            output.add_object(detected_object(x, y, w, h, label, prob))
    if output is not None:
        frameoutput_list.append(output)
    return frameoutput_list

def cal_overlap(obj_1, obj_2):
    overlap_w = max(0,
                    min(obj_1._x+obj_1._w, obj_2._x+obj_2._w)-max(obj_1._x, obj_2._x))
    overlap_h = max(0,
                    min(obj_1._y+obj_1._h, obj_2._y+obj_2._h)-max(obj_1._y, obj_2._y))
    overlap_area = overlap_w*overlap_h
    total_area = obj_1._w*obj_1._h+obj_2._w*obj_2._h-overlap_area
    return overlap_area/total_area

def cal_precision(true_pos, false_pos, false_neg):
    return float(true_pos)/max(0.000001, float(true_pos+false_pos))

def cal_recall(true_pos, false_pos, false_neg):
    return float(true_pos)/max(0.000001, float(true_pos+false_neg))

def cal_f1(true_pos, false_pos, false_neg):
    precision = cal_precision(true_pos, false_pos, false_neg)
    recall = cal_recall(true_pos, false_pos, false_neg)
    return 2.0/(1.0/max(0.000001,precision)+1.0/max(0.000001,recall))

def mean(l):
    return sum(l) / float(len(l))

def get_perf_of_frame(frameoutput_gt, frameoutput_nn, overlap_thresh, confident_thresh):
    run_time = frameoutput_nn._run_time
    true_pos = 0
    false_pos = 0
    false_neg = 0
    valid_objects_gt = set()
    valid_objects_nn = set()
    matched_objects_gt = set()
    for obj in frameoutput_gt._obj_list:
        if (obj._label == 'person') and (obj._prob >= confident_thresh):
            #if obj._prob >= confident_thresh:
            valid_objects_gt.add(obj)
    for obj in frameoutput_nn._obj_list:
        if (obj._label == 'person') and (obj._prob >= confident_thresh):
            #if obj._prob >= confident_thresh:
            valid_objects_nn.add(obj)
    for obj_nn in valid_objects_nn:
        obj_matched = None
        max_overlap = -1
        for obj_gt in valid_objects_gt:
            if obj_gt in matched_objects_gt: continue
            overlap = cal_overlap(obj_gt, obj_nn)
            if overlap >= overlap_thresh and obj_gt._label == obj_nn._label:
                if overlap > max_overlap:
                    obj_matched = obj_gt
                    max_overlap = overlap
        if obj_matched != None:
            true_pos = true_pos+1
            matched_objects_gt.add(obj_matched)
        else: false_pos = false_pos+1
    false_neg = len(valid_objects_gt)-len(matched_objects_gt)
    # print str(len(valid_objects_gt))+"\t"+str(len(valid_objects_nn))+"\t"+\
    #        str(len(frameoutput_gt._obj_list))+"\t"+str(len(frameoutput_nn._obj_list))+"\t"+\
    #        str(true_pos)+"\t"+str(false_pos)+"\t"+str(false_neg)
    return [true_pos, false_pos, false_neg, run_time]

def compare(video_name, start_ms, end_ms,
            model_1, img_size_1, frame_rate_1,
            model_2, img_size_2, frame_rate_2,
            overlap_thresh, confident_thresh):
    default_frame_rate = 1.0
    path_to_output_1 = output_file_name(video_name, start_ms, end_ms,
                                        model_1, img_size_1, default_frame_rate)
    path_to_output_2 = output_file_name(video_name, start_ms, end_ms,
                                        model_2, img_size_2, default_frame_rate)
    with open(path_to_output_1) as f: lines_1 = f.readlines()
    with open(path_to_output_2) as f: lines_2 = f.readlines()
    frameoutput_list_1 = parse_output_from_lines(lines_1)
    frameoutput_list_2 = parse_output_from_lines(lines_2)
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0
    run_time_list = []
    for i in range(len(frameoutput_list_1)):
        rounded_index_1 = (int(float(i)/default_frame_rate*frame_rate_1)+1)%len(frameoutput_list_1)
        rounded_index_2 = (int(float(i)/default_frame_rate*frame_rate_2)+1)%len(frameoutput_list_2)
        frameoutput_1 = frameoutput_list_1[rounded_index_1]
        frameoutput_2 = frameoutput_list_2[rounded_index_2]
        [true_pos, false_pos, false_neg, run_time] = get_perf_of_frame(frameoutput_1,
                                                                       frameoutput_2,
                                                                       overlap_thresh,
                                                                       confident_thresh)
        precision = cal_precision(true_pos, false_pos, false_neg)
        recall = cal_recall(true_pos, false_pos, false_neg)
        # print "########### "+frameoutput_1._name+":\t"+str(precision)+","+str(recall)
        total_true_pos = total_true_pos+true_pos
        total_false_pos = total_false_pos+false_pos
        total_false_neg = total_false_neg+false_neg
        if run_time < 1:
            run_time_list.append(run_time)
    return [total_true_pos, total_false_pos, total_false_neg, mean(run_time_list)]

def run_on_segment(path_to_video, video_name,
                   model_gt, img_size_gt, frame_rate_gt,
                   model_list, img_size_list, frame_rate_list):
    for i in range(len(model_list)):
        for j in range(len(img_size_list)):
            frame_rate = 1.0
            prepare(path_to_video, video_name, start_ms, end_ms,
                    model_list[i], img_size_list[j], frame_rate)
    if not os.path.exists(save_images_root): os.makedirs(save_images_root)
    for i in range(len(model_list)):
        for j in range(len(img_size_list)):
            model = model_list[i]
            img_size = img_size_list[j]
            frame_rate = 1.0
            save_images_path = save_images_root+"Images_"+video_name+\
                                "_From_"+"{0:0>9}".format(start_ms)+\
                                "_To_"+"{0:0>9}".format(end_ms)+\
                                "_Size_"+str(img_size)+"_Rate_"+str(frame_rate)+"_Model_"+model+"/"
            if i == 0 and j == 0:
                save_images = True
                show_images = False
                if not os.path.exists(save_images_path): os.makedirs(save_images_path)
            else:
                save_images = False
                show_images = False
            inference(path_to_video, video_name, start_ms, end_ms,
                      model, img_size, frame_rate,
                      save_images, save_images_path, show_images)
    overlap_thresh = 0.3
    confident_thresh = 0.3
    print "############### EVALUATION ##################"
    analysis_file = analysis_root+"Result"+\
                    "_From_"+"{0:0>9}".format(start_ms)+\
                    "_To_"+"{0:0>9}".format(end_ms)+\
                    "_Video_"+video_name+".txt"
    if not os.path.exists(analysis_root): os.makedirs(analysis_root)
    file = open(analysis_file, "w+")
    for i in range(len(model_list)):
        for j in range(len(img_size_list)):
            for k in range(len(frame_rate_list)):
                model = model_list[i]
                img_size = img_size_list[j]
                frame_rate = frame_rate_list[k]
                [total_true_pos, total_false_pos,
                 total_false_neg, avg_runtime] = compare(video_name, start_ms, end_ms,
                                                         model_gt, img_size_gt, frame_rate_gt,
                                                         model, img_size, frame_rate,
                                                         overlap_thresh, confident_thresh)
                precision = cal_precision(total_true_pos, total_false_pos, total_false_neg)
                recall = cal_recall(total_true_pos, total_false_pos, total_false_neg)
                f1 = cal_f1(total_true_pos, total_false_pos, total_false_neg)
                if f1 >= 0.9: sign = "\t**"
                elif f1 >= 0.8: sign = "\t*"
                else: sign = ""
                line = '{0:.4g}'.format(f1)+"\t"+'{0:.4g}'.format(precision)+"\t"+'{0:.4g}'.format(recall)+\
                        "\t|\t"+str(total_true_pos)+"\t"+str(total_false_pos)+"\t"+str(total_false_neg)+\
                        "\t"+'{0:.4g}'.format(avg_runtime)+\
                        "\t|\t"+str(img_size)+"_"+str(frame_rate)+"_"+str(model)+sign
                print line
                file.write(line+"\n")
    print "############### DONE ##################"

def get_ms(mm, ss):
    return (mm*60+ss)*1000

def sec_to_ms(ss):
    return ss*1000


#cleanup_models()
#cleanup_frames()
#cleanup_outputs()
#cleanup_analysis()
#cleanup_savedimages()
#path_to_video = "/home/junchenj/videos/msra.mp4"
#video_name = "msra"

model_gt = "faster_rcnn_resnet101_coco_2017_11_08"
img_size_gt = 960
frame_rate_gt = 1.0

model_list = ["faster_rcnn_resnet101_coco_2017_11_08",
              "faster_rcnn_resnet50_coco_2017_11_08",
              "faster_rcnn_inception_v2_coco_2017_11_08"]
img_size_list = [960, 840, 720, 600, 480]
frame_rate_list = [1.00, 0.50, 0.20, 0.12, 0.04]
#model_list = ["faster_rcnn_resnet101_coco_2017_11_08"]
#img_size_list = [960]
#frame_rate_list = [1.0, 0.2, 0.12, 0.04]

#time_list = [get_ms(31,45), get_ms(38,45)]
#time_list = [get_ms(31,47), get_ms(38,47)]

path_to_video_list_list = []
time_list_list = []

path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-09_08.00.25.mp4",
                                "/mnt/videos/192.168.1.102-18-01-09_08.00.25.mp4",
                                "/mnt/videos/192.168.1.103-18-01-09_08.00.25.mp4"])
time_list_list.append([sec_to_ms(1081)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-09_11.20.29.mp4",
                                "/mnt/videos/192.168.1.102-18-01-09_11.20.29.mp4",
                                "/mnt/videos/192.168.1.103-18-01-09_11.20.29.mp4"])
time_list_list.append([sec_to_ms(2187)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-09_17.20.21.mp4",
                                "/mnt/videos/192.168.1.102-18-01-09_17.20.21.mp4",
                                "/mnt/videos/192.168.1.103-18-01-09_17.20.21.mp4"])
time_list_list.append([sec_to_ms(1892)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-10_08.00.36.mp4",
                                "/mnt/videos/192.168.1.102-18-01-10_08.00.37.mp4",
                                "/mnt/videos/192.168.1.103-18-01-10_08.00.36.mp4"])
time_list_list.append([sec_to_ms(2776)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-10_11.20.16.mp4",
                                "/mnt/videos/192.168.1.102-18-01-10_11.20.18.mp4",
                                "/mnt/videos/192.168.1.103-18-01-10_11.20.16.mp4"])
time_list_list.append([sec_to_ms(2298)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-10_17.20.26.mp4",
                                "/mnt/videos/192.168.1.102-18-01-10_17.20.26.mp4",
                                "/mnt/videos/192.168.1.103-18-01-10_17.20.25.mp4"])
time_list_list.append([sec_to_ms(1230)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-11_08.00.12.mp4",
                                "/mnt/videos/192.168.1.102-18-01-11_08.00.13.mp4",
                                "/mnt/videos/192.168.1.103-18-01-11_08.00.12.mp4"])
time_list_list.append([sec_to_ms(2350)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-11_11.39.52.mp4",
                                "/mnt/videos/192.168.1.102-18-01-11_11.39.52.mp4",
                                "/mnt/videos/192.168.1.103-18-01-11_11.39.52.mp4"])
time_list_list.append([sec_to_ms(850)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.100-18-01-11_17.20.18.mp4",
                                "/mnt/videos/192.168.1.102-18-01-11_17.20.17.mp4",
                                "/mnt/videos/192.168.1.103-18-01-11_17.20.18.mp4"])
time_list_list.append([sec_to_ms(1880)]) # static



path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-09_08.00.26.mp4",
                                "/mnt/videos/192.168.1.105-18-01-09_08.00.27.mp4"])
time_list_list.append([sec_to_ms(1899)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-09_11.20.29.mp4",
                                "/mnt/videos/192.168.1.105-18-01-09_11.20.29.mp4"])
time_list_list.append([sec_to_ms(2331)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-09_17.20.21.mp4",
                                "/mnt/videos/192.168.1.105-18-01-09_17.20.21.mp4"])
time_list_list.append([sec_to_ms(1613)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-10_08.00.36.mp4",
                                "/mnt/videos/192.168.1.105-18-01-10_08.00.38.mp4"])
time_list_list.append([sec_to_ms(2804)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-10_11.20.16.mp4",
                                "/mnt/videos/192.168.1.105-18-01-10_11.20.18.mp4"])
time_list_list.append([sec_to_ms(2180)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-10_17.20.26.mp4",
                                "/mnt/videos/192.168.1.105-18-01-10_17.20.27.mp4"])
time_list_list.append([sec_to_ms(1370)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-11_08.00.12.mp4",
                                "/mnt/videos/192.168.1.105-18-01-11_08.00.13.mp4"])
time_list_list.append([sec_to_ms(1670)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-11_11.39.52.mp4",
                                "/mnt/videos/192.168.1.105-18-01-11_11.39.52.mp4"])
time_list_list.append([sec_to_ms(1086)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.104-18-01-11_17.20.18.mp4",
                                "/mnt/videos/192.168.1.105-18-01-11_17.20.18.mp4"])
time_list_list.append([sec_to_ms(900)]) # static



path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-09_08.00.26.mp4",
                                "/mnt/videos/192.168.1.111-18-01-09_08.00.25.mp4",
                                "/mnt/videos/192.168.1.112-18-01-09_08.00.25.mp4"])
time_list_list.append([sec_to_ms(1620)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-09_11.20.29.mp4",
                                "/mnt/videos/192.168.1.111-18-01-09_11.20.28.mp4",
                                "/mnt/videos/192.168.1.112-18-01-09_11.20.29.mp4"])
time_list_list.append([sec_to_ms(2059)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-09_17.20.21.mp4",
                                "/mnt/videos/192.168.1.111-18-01-09_17.20.21.mp4",
                                "/mnt/videos/192.168.1.112-18-01-09_17.20.21.mp4"])
time_list_list.append([sec_to_ms(2189)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-10_08.00.36.mp4",
                                "/mnt/videos/192.168.1.111-18-01-10_08.00.36.mp4",
                                "/mnt/videos/192.168.1.112-18-01-10_08.00.37.mp4"])
time_list_list.append([sec_to_ms(1590)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-10_11.20.16.mp4",
                                "/mnt/videos/192.168.1.111-18-01-10_11.20.16.mp4",
                                "/mnt/videos/192.168.1.112-18-01-10_11.20.18.mp4"])
time_list_list.append([sec_to_ms(475)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-10_17.20.27.mp4",
                                "/mnt/videos/192.168.1.111-18-01-10_17.20.25.mp4",
                                "/mnt/videos/192.168.1.112-18-01-10_17.20.27.mp4"])
time_list_list.append([sec_to_ms(850)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-11_08.00.14.mp4",
                                "/mnt/videos/192.168.1.111-18-01-11_08.00.13.mp4",
                                "/mnt/videos/192.168.1.112-18-01-11_08.00.13.mp4"])
time_list_list.append([sec_to_ms(3379)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-11_11.39.52.mp4",
                                "/mnt/videos/192.168.1.111-18-01-11_11.39.52.mp4",
                                "/mnt/videos/192.168.1.112-18-01-11_11.39.52.mp4"])
time_list_list.append([sec_to_ms(961)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.110-18-01-11_17.20.18.mp4",
                                "/mnt/videos/192.168.1.111-18-01-11_17.20.17.mp4",
                                "/mnt/videos/192.168.1.112-18-01-11_17.20.18.mp4"])
time_list_list.append([sec_to_ms(530)]) # static



path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-09_08.00.26.mp4",
                                "/mnt/videos/192.168.1.116-18-01-09_08.00.26.mp4"])
time_list_list.append([sec_to_ms(1678)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-09_11.20.29.mp4",
                                "/mnt/videos/192.168.1.116-18-01-09_11.20.29.mp4"])
time_list_list.append([sec_to_ms(1280)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-09_17.20.22.mp4",
                                "/mnt/videos/192.168.1.116-18-01-09_17.20.21.mp4"])
time_list_list.append([sec_to_ms(1025)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-10_08.00.36.mp4",
                                "/mnt/videos/192.168.1.116-18-01-10_08.00.36.mp4"])
time_list_list.append([sec_to_ms(1950)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-10_11.20.18.mp4",
                                "/mnt/videos/192.168.1.116-18-01-10_11.20.16.mp4"])
time_list_list.append([sec_to_ms(1975)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-10_17.20.28.mp4",
                                "/mnt/videos/192.168.1.116-18-01-10_17.20.27.mp4"])
time_list_list.append([sec_to_ms(1000)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-11_08.00.13.mp4",
                                "/mnt/videos/192.168.1.116-18-01-11_08.00.12.mp4"])
time_list_list.append([sec_to_ms(750)]) # static
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-11_11.39.52.mp4",
                                "/mnt/videos/192.168.1.116-18-01-11_11.39.52.mp4"])
time_list_list.append([sec_to_ms(748)]) # changing
path_to_video_list_list.append(["/mnt/videos/192.168.1.115-18-01-11_17.20.18.mp4",
                                "/mnt/videos/192.168.1.116-18-01-11_17.20.18.mp4"])
time_list_list.append([sec_to_ms(1317)]) # changing


for j in range(len(path_to_video_list_list)):
    path_to_video_list = path_to_video_list_list[j]
    time_list = time_list_list[j]
    for x in time_list:
        for y in range(0,2):
            start_ms = x+1000*y
            end_ms = start_ms+1000
            for i in range(len(path_to_video_list)):
                path_to_video = path_to_video_list[i]
                video_name = (os.path.basename(path_to_video))[:-4]
                run_on_segment(path_to_video, video_name,
                               model_gt, img_size_gt, frame_rate_gt,
                               model_list, img_size_list, frame_rate_list)

for j in range(len(path_to_video_list_list)):
    path_to_video_list = path_to_video_list_list[j]
    time_list = time_list_list[j]
    for x in time_list:
        for y in range(2,4):
            start_ms = x+1000*y
            end_ms = start_ms+1000
            for i in range(len(path_to_video_list)):
                path_to_video = path_to_video_list[i]
                video_name = (os.path.basename(path_to_video))[:-4]
                run_on_segment(path_to_video, video_name,
                               model_gt, img_size_gt, frame_rate_gt,
                               model_list, img_size_list, frame_rate_list)

for j in range(len(path_to_video_list_list)):
    path_to_video_list = path_to_video_list_list[j]
    time_list = time_list_list[j]
    for x in time_list:
        for y in range(-2,0):
            start_ms = x+1000*y
            end_ms = start_ms+1000
            for i in range(len(path_to_video_list)):
                path_to_video = path_to_video_list[i]
                video_name = (os.path.basename(path_to_video))[:-4]
                run_on_segment(path_to_video, video_name,
                               model_gt, img_size_gt, frame_rate_gt,
                               model_list, img_size_list, frame_rate_list)


