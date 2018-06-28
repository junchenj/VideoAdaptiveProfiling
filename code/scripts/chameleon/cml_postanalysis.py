import sys
import os, shutil, argparse
import subprocess
import tensorflow as tf
import time

output_root = "/mnt/tmp/output/"
analysis_root = "/mnt/tmp/analysis/"

def output_file_name(video_name, start_ms, end_ms, model, img_size, frame_rate):
    return output_root+"Result_Video_"+video_name+\
        "_From_"+"{0:0>9}".format(start_ms)+\
        "_To_"+"{0:0>9}".format(end_ms)+\
        "_Size_"+str(img_size)+"_Rate_"+str(frame_rate)+"_Model_"+model+".txt"

def analysis_file_name(video_name, start_ms, end_ms):
    return analysis_root+"Result"+\
            "_From_"+"{0:0>9}".format(start_ms)+\
            "_To_"+"{0:0>9}".format(end_ms)+\
            "_Video_"+video_name+".txt"

############ From cml_controller.py #############
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

###########################

key_to_value = {}

def get_config_key(config):
    [model, img_size, frame_rate] = config
    return "m:"+str(model)+"_l:"+str(img_size)+"_r:"+str(frame_rate)

def get_key(video_name, start_ms, end_ms, config):
    [model, img_size, frame_rate] = config
    return str(model)+"_"+str(img_size)+"_"+str(frame_rate)+\
        "_"+str(video_name)+"_"+str(start_ms)+"_"+str(end_ms)

def compare_arbitrary_configs(video_name, start_ms, end_ms, config_1, config_2):
    requested_key_1 = get_key(video_name, start_ms, end_ms, config_1)
    requested_key_2 = get_key(video_name, start_ms, end_ms, config_2)
    requested_key = requested_key_1+"___"+requested_key_2
    if requested_key not in key_to_value:
        [model_1, img_size_1, frame_rate_1] = config_1
        [model_2, img_size_2, frame_rate_2] = config_2
        overlap_thresh = 0.3
        confident_thresh = 0.3
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
            perf = get_perf_of_frame(frameoutput_1, frameoutput_2, overlap_thresh, confident_thresh)
            [true_pos, false_pos, false_neg, run_time] = perf
            precision = cal_precision(perf)
            recall = cal_recall(perf)
            # print "########### "+frameoutput_1._name+":\t"+str(precision)+","+str(recall)
            total_true_pos = total_true_pos+true_pos
            total_false_pos = total_false_pos+false_pos
            total_false_neg = total_false_neg+false_neg
            if run_time < 1:
                run_time_list.append(run_time)
            key_to_value[requested_key] = [total_true_pos, total_false_pos,
                                           total_false_neg, mean(run_time_list)]
    return key_to_value[requested_key]

def get_performance(video_name, time_ms, config):
    start_ms = time_ms
    end_ms = time_ms+1000
    requested_key = get_key(video_name, start_ms, end_ms, config)
    if requested_key not in key_to_value:
        output_file = analysis_file_name(video_name, start_ms, end_ms)
        with open(output_file) as f: lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields = line.split('\t')
            true_positives = int(fields[4])
            false_positives = int(fields[5])
            false_negatives = int(fields[6])
            run_time = float(fields[7])
            tag = fields[9]
            values = tag.split('_')
            img_size = int(values[0])
            frame_rate = float(values[1])
            run_time = run_time*float(frame_rate)
            model = tag[len(values[0])+1+len(values[1])+1:]
            key = get_key(video_name, start_ms, end_ms, [model, img_size, frame_rate])
            key_to_value[key] = [true_positives, false_positives, false_negatives, run_time]
    return key_to_value[requested_key]

def cal_precision(perf):
    [true_pos, false_pos, false_neg, run_time] = perf
    return float(true_pos)/max(0.000001, float(true_pos+false_pos))

def cal_recall(perf):
    [true_pos, false_pos, false_neg, run_time] = perf
    return float(true_pos)/max(0.000001, float(true_pos+false_neg))

def cal_f1(perf):
    precision = cal_precision(perf)
    recall = cal_recall(perf)
    return cal_f1_from_pair(precision, recall)

def cal_f1_from_pair(precision, recall):
    if precision == 0 and recall == 0:
        return 1.0
    else: return 2.0/(1.0/max(0.000001,precision)+1.0/max(0.000001,recall))

def cal_cost(perf):
    [true_pos, false_pos, false_neg, run_time] = perf
    return run_time

def sec_to_ms(ss):
    return ss*1000

def mean(l):
    return sum(l) / float(len(l))

def get_camera_from_video_name(video_name):
    return video_name[:video_name.index('-')]

def get_avg_eval(eval_list, accuracy_thresh):
    total_cost = 0
    total_accuracy = 0
    total_pass_count = 0
    total_count = 0
    for eval in eval_list:
        [avg_cost, avg_accuracy, pass_count, count] = eval
        total_cost = total_cost+avg_cost*count
        total_accuracy = total_accuracy+avg_accuracy*count
        total_pass_count = total_pass_count+pass_count
        total_count = total_count+count
    if float(total_accuracy)/float(total_count) > accuracy_thresh: label1 = "*"
    else: label1 = ""
    if float(total_pass_count)/float(total_count) > 0.8: label2 = "$"
    else: label2 = ""
    return '{0:.4g}'.format(float(total_cost)/float(total_count))+", "\
           +'{0:.4g}'.format(float(total_accuracy)/float(total_count))+", "\
           +'{0:.4g}'.format(float(total_pass_count)/float(total_count))+", "\
           +'{0:.4g}'.format(total_count)+", "\
           +label1+", "+label2

def get_fake_cost(config):
    [model, img_size, frame_rate] = config
    model_index = float(model_list.index(model))
    max_img_size = max(img_size_list)
    value = (1.0/pow(model_index+1,0.2))*pow(max_img_size/img_size, 0.5)*frame_rate
    # print str(model_index)+"\t"+str(max_img_size)+"\t"+str(frame_rate)+"\t"+str(value)
    return value

def valid(config, model_list, img_size_list, frame_rate_list):
    return True
    #[model, img_size, frame_rate] = config
    #if (model_list.index(model) < len(model_list)-1) or \
    #    (img_size_list.index(img_size) < len(img_size_list)-1):
    #    return True
    #else: return False

def get_best_config(accuracy_thresh, video_name, time_ms,
                    model_list, img_size_list, frame_rate_list):
    best_cost = 10000
    best_config = None
    # print ""
    for model in model_list:
        for img_size in img_size_list:
            for frame_rate in frame_rate_list:
                config = [model, img_size, frame_rate]
                if not valid(config, model_list, img_size_list, frame_rate_list):
                    continue
                #cost = get_fake_cost(config)
                cost = cal_cost(get_performance(video_name, time_ms, config))
                f1 = cal_f1(get_performance(video_name, time_ms, config))
                if (best_config is None) or (cost < best_cost and f1 >= accuracy_thresh):
                    best_config = config
                    best_cost = cost
    return best_config

def get_chemeleon_config(accuracy_thresh, video_name, time_ms,
                         model_list, img_size_list, frame_rate_list):
    dt_model = model_list[len(model_list)-1]
    dt_img_size = img_size_list[len(img_size_list)-1]
    dt_frame_rate = frame_rate_list[len(frame_rate_list)-1]
    #dt_model = model_list[0]
    #dt_img_size = img_size_list[0]
    #dt_frame_rate = frame_rate_list[0]
    best_cost = 10000
    best_config = None
    #print "\nchameleon"
    for model in model_list:
        for img_size in img_size_list:
            for frame_rate in frame_rate_list:
    #for i in range(len(model_list)):
    #    for j in range(len(img_size_list)):
    #        for k in range(len(frame_rate_list)):
                config = [model, img_size, frame_rate]
                if not valid(config, model_list, img_size_list, frame_rate_list):
                    continue
                cost = cal_cost(get_performance(video_name, time_ms, config))
                '''
                f1_model = cal_f1(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                  [model_list[0], dt_img_size, dt_frame_rate],
                                  [model, dt_img_size, dt_frame_rate]))
                f1_img_size = cal_f1(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                     [dt_model, img_size_list[0], dt_frame_rate],
                                     [dt_model, img_size, dt_frame_rate]))
                f1_frame_rate = cal_f1(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                       [dt_model, dt_img_size, frame_rate_list[0]],
                                       [dt_model, dt_img_size, frame_rate]))
                f1 = f1_model * f1_img_size * f1_frame_rate
                real_f1 = cal_f1(get_performance(video_name, time_ms, config))
                print '{0:.4g}'.format(f1_model)+"\t"+\
                      '{0:.4g}'.format(f1_img_size)+"\t"+\
                      '{0:.4g}'.format(f1_frame_rate)+"\t=\t"+\
                      '{0:.4g}'.format(f1)+"\t<>\t"+\
                      '{0:.4g}'.format(real_f1)+"\t"
                '''
                precision_model = cal_precision(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                  [model_list[0], dt_img_size, dt_frame_rate],
                                  [model, dt_img_size, dt_frame_rate]))
                precision_img_size = cal_precision(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                     [dt_model, img_size_list[0], dt_frame_rate],
                                     [dt_model, img_size, dt_frame_rate]))
                precision_frame_rate = cal_precision(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                       [dt_model, dt_img_size, frame_rate_list[0]],
                                       [dt_model, dt_img_size, frame_rate]))
                precision = precision_model * precision_img_size * precision_frame_rate
                recall_model = cal_recall(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                  [model_list[0], dt_img_size, dt_frame_rate],
                                  [model, dt_img_size, dt_frame_rate]))
                recall_img_size = cal_recall(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                     [dt_model, img_size_list[0], dt_frame_rate],
                                     [dt_model, img_size, dt_frame_rate]))
                recall_frame_rate = cal_recall(compare_arbitrary_configs(video_name, time_ms, time_ms+1000,
                                       [dt_model, dt_img_size, frame_rate_list[0]],
                                       [dt_model, dt_img_size, frame_rate]))
                recall = recall_model * recall_img_size * recall_frame_rate
                #precision = pow(precision, 0.5)
                #recall = pow(recall, 0.5)
                f1 = cal_f1_from_pair(precision, recall)
                real_f1 = cal_f1(get_performance(video_name, time_ms, config))
                #print '{0:.4g}'.format(f1)+"\t"+'{0:.4g}'.format(pow(f1, 0.5))+\
                #        "\t"+'{0:.4g}'.format(real_f1)+"\t"+'{0:.4g}'.format(pow(f1, 0.5)/real_f1)
                f1 = pow(f1, 2.0)
                #print '{0:.4g}'.format(f1)+"\t"+str(accuracy_thresh)+\
                #        "\t"+'{0:.4g}'.format(cost)+"\t"+'{0:.4g}'.format(best_cost)+\
                #        "\t"+str(config)+"\t"+str(best_config)
                if (best_config is None) or (cost < best_cost and f1 >= accuracy_thresh):
                    #print "update"
                    best_config = config
                    best_cost = cost
    return best_config

def get_eval(video_names, time_windows, accuracy_thresh, video_to_time_to_decision):
    cost_list = []
    accuracy_list = []
    pass_count = 0
    count = 0
    for video_name in video_names:
        for time_ms in time_windows:
            [best_config, tried_configs] = video_to_time_to_decision[video_name][time_ms]
            cost = 0
            for config_key in tried_configs:
                config = tried_configs[config_key]
                #cost = get_fake_cost(config)
                cost = cost + cal_cost(get_performance(video_name, time_ms, config))
            f1 = cal_f1(get_performance(video_name, time_ms, best_config))
            cost_list.append(cost)
            accuracy_list.append(f1)
            if f1 < accuracy_thresh:
                label = "x"
            else:
                label = "v"
                pass_count = pass_count+1
            #print str(cost)+"\t"+str(f1)+"\t"+label
            count = count+1
    return [mean(cost_list), mean(accuracy_list), pass_count, count]

def get_chameleon_profile_configs(model_list,
                                  img_size_list,
                                  frame_rate_list):
    result = {}
    default_model = model_list[len(model_list)-1]
    default_img_size = img_size_list[len(img_size_list)-1]
    default_frame_rate = frame_rate_list[len(frame_rate_list)-1]
    for model in model_list:
        config = [model, default_img_size, default_frame_rate]
        result[get_config_key(config)] = config
    for img_size in img_size_list:
        config = [default_model, img_size, default_frame_rate]
        result[get_config_key(config)] = config
    for frame_rate in frame_rate_list:
        config = [default_model, default_img_size, frame_rate]
        result[get_config_key(config)] = config
    return result

def simulate_chameleon(video_names, time_windows,
                       accuracy_thresh, distorted_accuracy_thresh,
                       camera_to_best_config,
                       model_list, img_size_list, frame_rate_list):
    #print "==== REAL CHAMELEON ===="
    video_to_time_to_decision = {}
    leader_video = video_names[0]
    first_time = time_windows[0]
    best_config = None
    for video_name in video_names:
        time_to_decision = {}
        time_to_config = {}
        for time_ms in time_windows:
            time_to_config[time_ms] = best_config
            tried_configs = {}
            if best_config is None:
                best_config = get_chemeleon_config(distorted_accuracy_thresh,
                                                   leader_video,
                                                   first_time,
                                                   model_list,
                                                   img_size_list,
                                                   frame_rate_list)
                #tried_configs[get_config_key(best_config)] = best_config
                tried_configs = get_chameleon_profile_configs(model_list,
                                                              img_size_list,
                                                              frame_rate_list)
            tried_configs[get_config_key(best_config)] = best_config
            time_to_decision[time_ms] = [best_config, tried_configs]
        video_to_time_to_decision[video_name] = time_to_decision
    #print str(best_config)
    eval_real_chameleon = get_eval(video_names, time_windows,
                                   accuracy_thresh, video_to_time_to_decision)
    '''
    eval_real_chameleon = [1, 1, 1, 1]
    
    '''
    #print "==== LEADER CHAMELEON ===="
    # optimal leader
    video_to_time_to_decision = {}
    leader_video = video_names[0]
    first_time = time_windows[0]
    best_config = None
    for video_name in video_names:
        time_to_decision = {}
        time_to_config = {}
        for time_ms in time_windows:
            time_to_config[time_ms] = best_config
            tried_configs = {}
            if best_config is None:
                best_config = get_best_config(distorted_accuracy_thresh,
                                              leader_video,
                                              first_time,
                                              model_list,
                                              img_size_list,
                                              frame_rate_list)
                tried_configs = get_chameleon_profile_configs(model_list,
                                                              img_size_list,
                                                              frame_rate_list)
            tried_configs[get_config_key(best_config)] = best_config
            time_to_decision[time_ms] = [best_config, tried_configs]
        video_to_time_to_decision[video_name] = time_to_decision
    #print str(best_config)
    eval_leader_chameleon = get_eval(video_names, time_windows,
                                     accuracy_thresh, video_to_time_to_decision)
    '''
    eval_leader_chameleon = [1, 1, 1, 1]

    '''
    #print "==== LEADER OPTIMAL ===="
    # optimal leader
    video_to_time_to_decision = {}
    leader_video = video_names[0]
    first_time = time_windows[0]
    best_config = get_best_config(distorted_accuracy_thresh,
                                  leader_video, first_time,
                                  model_list, img_size_list, frame_rate_list)
    for video_name in video_names:
        time_to_decision = {}
        time_to_config = {}
        for time_ms in time_windows:
            tried_configs = {}
            tried_configs[get_config_key(best_config)] = best_config
            time_to_decision[time_ms] = [best_config, tried_configs]
        video_to_time_to_decision[video_name] = time_to_decision
    #print str(best_config)
    eval_leader = get_eval(video_names, time_windows,
                           accuracy_thresh, video_to_time_to_decision)
    '''
    eval_leader = [1, 1, 1, 1]

    '''
    #print "==== 1EPOCH ===="
    # 1epoch
    video_to_time_to_decision = {}
    for video_name in video_names:
        camera = get_camera_from_video_name(video_name)
        config = camera_to_best_config[camera]
        time_to_config = {}
        for time_ms in time_windows:
            tried_configs = {}
            tried_configs[get_config_key(config)] = config
            time_to_decision[time_ms] = [config, tried_configs]
        video_to_time_to_decision[video_name] = time_to_decision
        #print str(config)
    eval_1epoch = get_eval(video_names, time_windows,
                           accuracy_thresh, video_to_time_to_decision)
    '''
    eval_1epoch = [1, 1, 1, 1]

    '''
    #print "==== STATIC ===="
    # static
    video_to_time_to_decision = {}
    static_config = [model_list[0], img_size_list[0], frame_rate_list[0]]
    for video_name in video_names:
        time_to_decision = {}
        time_to_config = {}
        for time_ms in time_windows:
            tried_configs = {}
            tried_configs[get_config_key(static_config)] = static_config
            time_to_decision[time_ms] = [static_config, tried_configs]
        video_to_time_to_decision[video_name] = time_to_decision
    #print str(static_config)
    eval_static = get_eval(video_names, time_windows,
                           accuracy_thresh, video_to_time_to_decision)
    '''
    eval_static = [1, 1, 1, 1]
    '''
    return [eval_real_chameleon, eval_leader_chameleon,
            eval_leader, eval_1epoch, eval_static]


def analyze(path_to_video_list_list, time_list_list,
            model_gt, img_size_gt, frame_rate_gt,
            model_list, img_size_list, frame_rate_list,
            accuracy_thresh, timeindex_shift):
    #accuracy_thresh = 0.85
    
    print '###############################'
    print '###############################'
    #distorted_accuracy_thresh = accuracy_thresh
    #distorted_accuracy_thresh = 1-1.0*(1-accuracy_thresh)
    #distorted_accuracy_thresh = 1-0.3*(1-accuracy_thresh)
    
    for distorted_accuracy_thresh in [1-1.0*(1-accuracy_thresh),
                                      1-0.75*(1-accuracy_thresh),
                                      1-0.5*(1-accuracy_thresh),
                                      1-0.25*(1-accuracy_thresh),
                                      1-0.0*(1-accuracy_thresh)]:
        print "\n\n\n distorted_accuracy_thresh="+str(distorted_accuracy_thresh)
        eval_real_chameleon_list = []
        eval_leader_chameleon_list = []
        eval_leader_list = []
        eval_1epoch_list = []
        eval_static_list = []
        camera_to_best_config = {}
        ylist = [3,2,1,0,-1,-2]
        #timeindex_shift = 5
        for j in range(len(path_to_video_list_list)):
            path_to_video_list = path_to_video_list_list[j]
            time_list = time_list_list[j]
            for x in time_list:
                video_names = []
                time_windows = []
                for i in range(len(path_to_video_list)):
                    path_to_video = path_to_video_list[i]
                    video_name = (os.path.basename(path_to_video))[:-4]
                    video_names.append(video_name)
                #for y in range(-2,4):
                #for y in range(0,4):
                for i in range(len(ylist)):
                    y = ylist[(i+timeindex_shift)%len(ylist)]
                    start_ms = x+1000*y
                    time_windows.append(start_ms)
                for video_name in video_names:
                    camera = get_camera_from_video_name(video_name)
                    #print camera
                    for start_ms in time_windows:
                        best_config = get_best_config(distorted_accuracy_thresh, video_name, start_ms,
                                                      model_list, img_size_list, frame_rate_list)
                    if camera not in camera_to_best_config:
                        camera_to_best_config[camera] = best_config
            for x in time_list:
                #print x
                video_names = []
                time_windows = []
                for i in range(len(path_to_video_list)):
                    path_to_video = path_to_video_list[i]
                    video_name = (os.path.basename(path_to_video))[:-4]
                    video_names.append(video_name)
                #for y in range(-2,4):
                #for y in range(0,4):
                for i in range(len(ylist)):
                    y = ylist[(i+timeindex_shift)%len(ylist)]
                    start_ms = x+1000*y
                    time_windows.append(start_ms)
                [eval_real_chameleon,
                 eval_leader_chameleon,
                 eval_leader,
                 eval_1epoch,
                 eval_static] = simulate_chameleon(video_names, time_windows,
                                                   accuracy_thresh, distorted_accuracy_thresh,
                                                   camera_to_best_config,
                                                   model_list, img_size_list, frame_rate_list)
                eval_real_chameleon_list.append(eval_real_chameleon)
                eval_leader_chameleon_list.append(eval_leader_chameleon)
                eval_leader_list.append(eval_leader)
                eval_1epoch_list.append(eval_1epoch)
                eval_static_list.append(eval_static)
                #'''
        print str(get_avg_eval(eval_real_chameleon_list, accuracy_thresh))
        #print str(get_avg_eval(eval_leader_chameleon_list, accuracy_thresh))
        #print str(get_avg_eval(eval_leader_list, accuracy_thresh))
        print str(get_avg_eval(eval_1epoch_list, accuracy_thresh))
        #print str(get_avg_eval(eval_static_list, accuracy_thresh))
        print ""



parser = argparse.ArgumentParser()
parser.add_argument("--offset")
parser.add_argument("--thresh")
args = parser.parse_args()
timeindex_shift = int(args.offset)
accuracy_thresh = float(args.thresh)

model_gt = "faster_rcnn_resnet101_coco_2017_11_08"
img_size_gt = 960
frame_rate_gt = 1.0

model_list = ["faster_rcnn_resnet101_coco_2017_11_08",
              "faster_rcnn_resnet50_coco_2017_11_08",
              "faster_rcnn_inception_v2_coco_2017_11_08"]
img_size_list = [960, 840, 720, 600, 480]
frame_rate_list = [1.00, 0.50, 0.20, 0.12, 0.04]



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
#'''
analyze(path_to_video_list_list, time_list_list,
        model_gt, img_size_gt, frame_rate_gt,
        model_list, img_size_list, frame_rate_list,
        accuracy_thresh, timeindex_shift)



path_to_video_list_list = []
time_list_list = []
#'''
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
#'''
analyze(path_to_video_list_list, time_list_list,
        model_gt, img_size_gt, frame_rate_gt,
        model_list, img_size_list, frame_rate_list,
        accuracy_thresh, timeindex_shift)



path_to_video_list_list = []
time_list_list = []
#'''
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
#'''
analyze(path_to_video_list_list, time_list_list,
        model_gt, img_size_gt, frame_rate_gt,
        model_list, img_size_list, frame_rate_list,
        accuracy_thresh, timeindex_shift)



path_to_video_list_list = []
time_list_list = []
#'''
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

analyze(path_to_video_list_list, time_list_list,
        model_gt, img_size_gt, frame_rate_gt,
        model_list, img_size_list, frame_rate_list,
        accuracy_thresh, timeindex_shift)














