import sys
import os, shutil
import subprocess

frames_root = "/mnt/tmp/frames/"
new_model_root = "/mnt/tmp/models/"
output_root = "/mnt/tmp/output/"

def cleanup_frames():
	try: shutil.rmtree(frames_root, ignore_errors=True)
	except OSError: print "Error deleting "+frames_root

def cleanup_models():
	try: shutil.rmtree(new_model_root, ignore_errors=True)
	except OSError: print "Error deleting "+frames_root

def inference(video, start_ms, end_ms, model, img_size, frame_rate):
	old_model_path = "/mnt/detectors/"+model
	new_model_path = new_model_root+"Model_"+model+"_Size_"+str(img_size)+"/"
	frames_path = frames_root+"Frame_From_"+str(start_ms)+"_To_"+str(end_ms)
	output_file = output_root+"Result_From_"+str(start_ms)+"_To_"+str(end_ms)+\
		"_Size_"+str(img_size)+"_Rate_"+str(frame_rate)+"_Model_"+model+".txt"
	cmd = "sh cml_preprocess.sh --video "+video+" \
		--start "+str(start_ms)+" --end "+str(end_ms)+" \
		--frames "+frames_path+" \
		--model "+old_model_path+" \
		--newmodel "+new_model_path+" \
		--size "+str(img_size)+" \
		--framerate "+str(frame_rate)+" \
		--modelname "+model+" \
		--output "+output_file
	print cmd
	subprocess.call(cmd.split())



#cleanup_models()
#cleanup_frames()
video = "/home/junchenj/videos/msra.mp4"

from cml_test import test

test("x", "y")

time = 600000
while time < 700000:
	print time
	model = "faster_rcnn_nas_coco_2017_11_08"
	img_size = 720
	frame_rate = 0.033
	start_ms = time
	end_ms = time + 10000

	inference(video, start_ms, end_ms, model, img_size, frame_rate)

	time = time + 10000

