import sys
import subprocess

def inference(video, start_ms, end_ms, model, img_size, frame_rate):
	model_path = "/home/junchenj/workspace/scripts/chameleon/RESIZED_"+model+"_SIZE_"+str(img_size)+"/"
	cmd = "sh cml_preprocess.sh --video "+video+" \
		--start "+str(start_ms)+" --end "+str(end_ms)+" \
		--frames /home/junchenj/videos/frames \
		--model /mnt/detectors/faster_rcnn_nas_coco_2017_11_08 \
		--newmodel "+model_path+" \
		--size "+str(img_size)+" --framerate "+str(frame_rate)+" \
		--modelname "+model
	subprocess.call(cmd.split())


#video = "/home/junchenj/videos/CnnNews.mp4"
#video = "/home/junchenj/videos/bridge.ts"
video = "/home/junchenj/videos/Bellevue_Bellevue_NE8th__2017-04-08_09-52-19.mp4"
model = "faster_rcnn_nas_coco_2017_11_08"
#img_size = 300
img_size = 720
frame_rate = 0.33
start_ms = 0
#start_ms = 3004
#end_ms = 6000
end_ms = 1000
#end_ms = 6003

#start_ms = 3000
#end_ms = 6000

inference(video, start_ms, end_ms, model, img_size, frame_rate)
