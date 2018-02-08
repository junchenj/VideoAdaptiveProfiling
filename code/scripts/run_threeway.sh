video=$1

echo $video

FOLDER=$(pwd)
echo $FOLDER

ROOT=/home/junchenj/workspace/threeway_result_${video}
rm -r $ROOT
mkdir $ROOT
echo $ROOT

FRAMES=${ROOT}/frames
rm -r ${FRAMES}
mkdir ${FRAMES}

ffmpeg -loglevel quiet -i ~/videos/${video}*.mp4 ${FRAMES}/out-%06d.jpg

cd /home/junchenj/workspace/scripts/tf/

rm -r ResizedModel*

DEFAULT_SIZE=400
DEFAULT_SAMPLING=0.033
DEFAULT_MODEL=faster_rcnn_resnet50

for z in 1200 800 400 200 ; do
for x in ssd_mobilenet_v1 ssd_inception_v2 faster_rcnn_inception_v2 faster_rcnn_resnet50  faster_rcnn_resnet101; do
for y in 0.33; do
    model=$x
    sampling=$y
    size=$z
    output=${ROOT}/Detections_Size_${size}_Sampling_${sampling}_Model_${model}.txt
    python cropping_test.py -t ~/workspace/tensorflow/models/ -i ${FRAMES} -o ${output} -m ${model} -r ${sampling} -s $size
done
done
done



cd $FOLDER

cd /home/junchenj/workspace/scripts/bgs/
DEFAULT_SAMPLING='0.033'

IMAGES=${ROOT}/images

for x in 0.0001; do
minarea=$x
python extraction.py -f ${FRAMES} -o ${IMAGES} -m ${minarea}
for y in nasnet_large resnet_v1_101 resnet_v1_50 inception_v2 mobilenet_v1; do
    model=$y
    sampling=${DEFAULT_SAMPLING}
    output=${ROOT}/Classifications_MinArea_${minarea}_Sampling_${sampling}_Model_${model}.txt
    python tensorflow_classifier.py -e ${IMAGES}/log.txt -i ${IMAGES} -o ${output} -m ${model} -r ${sampling}
    rm -rf /tmp/checkpoints/*
done
done

rm -r ${FRAMES}
rm -r ${IMAGES}

cd $FOLDER
