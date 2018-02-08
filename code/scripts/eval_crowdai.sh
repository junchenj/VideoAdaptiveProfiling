id=$1

FOLDER=$(pwd)
echo $FOLDER

ROOT=/home/junchenj/workspace/eval_crowdai_${id}
rm -r $ROOT
mkdir $ROOT
echo $ROOT

cd /home/junchenj/workspace/scripts/tf/

DEFAULT_SIZE=400
DEFAULT_SAMPLING=0.1
DEFAULT_MODEL=faster_rcnn_resnet50

FRAMES=~/crowdai_sample/
mkdir $FRAMES
cp ~/object-detection-crowdai/*${id}.jpg $FRAMES

#for x in faster_rcnn_nas faster_rcnn_resnet101 faster_rcnn_inception_v2 ssd_mobilenet_v1 faster_rcnn_inception_resnet rfcn_resnet101 faster_rcnn_resnet50 ssd_inception_v2 faster_rcnn_resnet50_lowproposals; do
for x in faster_rcnn_inception_resnet_lowproposals; do
    model=$x
    sampling=${DEFAULT_SAMPLING}
    size=${DEFAULT_SIZE}
    output=${ROOT}/Detections_Size_${size}_Sampling_${sampling}_Model_${model}.txt
    #python cropping_objectdetection.py -t ~/workspace/tensorflow/models/ -i ${FRAMES} -o ${output} -m ${model} -r ${sampling} -s $size
    python cropping_test.py -t ~/workspace/tensorflow/models/ -i ${FRAMES} -o ${output} -m ${model} -r ${sampling} -s $size
    rm -rf ${model}*
done

#for z in 400 500 600 700 800 900 1000 1100 1200; do
for z in 1300 1400 1500 1600 ; do
    model=${DEFAULT_MODEL}
    sampling=${DEFAULT_SAMPLING}
    size=$z
    output=${ROOT}/Detections_NewSize_${size}_Sampling_${sampling}_Model_${model}.txt
    #python cropping_objectdetection.py -t ~/workspace/tensorflow/models/ -i ${FRAMES} -o ${output} -m ${model} -r ${sampling} -s $size
    #python cropping_test.py -t ~/workspace/tensorflow/models/ -i ${FRAMES} -o ${output} -m ${model} -r ${sampling} -s $size
    rm -rf ${model}*
done

cd $FOLDER
