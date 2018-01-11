sh start_tensorflow.sh

#python test_tf.py ~/workspace/tensorflow/models ssd_mobilenet_v1_coco_2017_11_17 ~/object-detection-crowdai/ output.txt

SAMPLING=0.0000003

rm -rf results
mkdir results

#for SIZE in 1920 1280 960 640 480 320 240; do
for SIZE in 1920; do
#for MODEL in ssd_mobilenet_v1_coco_2017_11_17 ssd_inception_v2_coco_2017_11_17 faster_rcnn_inception_v2_coco_2017_11_08 faster_rcnn_resnet50_coco_2017_11_08 faster_rcnn_resnet50_lowproposals_coco_2017_11_08 rfcn_resnet101_coco_2017_11_08 faster_rcnn_resnet101_coco_2017_11_08 faster_rcnn_resnet101_lowproposals_coco_2017_11_08 faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08 faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2017_11_08 faster_rcnn_nas_coco_2017_11_08 faster_rcnn_nas_lowproposals_coco_2017_11_08; do
for MODEL in faster_rcnn_resnet50_coco_2017_11_08; do
#for MODELSIZE in 1200 1000 800 600 400; do
for MODELSIZE in 1200 600; do
	OUTPUT='results/output_'$MODEL'_'$SAMPLING'_'$SIZE'_'$MODELSIZE'.txt'
	#python run_tensorflow_objectdetection.py ~/workspace/tensorflow/models ~/object-detection-crowdai/ $OUTPUT $MODEL $SAMPLING $SIZE
	python test_tf.py ~/workspace/tensorflow/models ~/object-detection-crowdai/ $OUTPUT $MODEL $SAMPLING $SIZE /home/junchenj/workspace/scripts/fine_tuned_model_$MODELSIZE
	#rm -rf ${MODEL}*
done
done
done
