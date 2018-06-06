video=$1

if [ -z "$video" ]
then
        echo Please specify video
        exit 0
fi

echo $video

FOLDER=$(pwd)
echo $FOLDER

ROOT=/home/junchenj/workspace/visualize_${video}
FRAMES=${ROOT}/frames
echo $ROOT

#rm -r $ROOT
mkdir $ROOT
#rm -r ${FRAMES}
mkdir ${FRAMES}
#ffmpeg -loglevel quiet -i ~/videos/${video}*.mp4 ${FRAMES}/out-%06d.jpg
#ffmpeg -i ~/videos/${video}*.ts ${FRAMES}/out-%06d.jpg
echo 'Frames saved in '${FRAMES}
rm -r ResizedModel*

cd /home/junchenj/workspace/scripts/chameleon/


SIZE=1080
SAMPLING=0.033
MODEL=faster_rcnn_resnet101
#SIZE=250
#SAMPLING=0.033
#MODEL=ssd_mobilenet_v1

OUTPUT=${ROOT}/Detections_Size_${SIZE}_Sampling_${SAMPLING}_Model_${MODEL}.txt
echo $OUTPUT
python tensorflow_inference.py -t ~/workspace/tensorflow/models/ -i ${FRAMES} -o ${OUTPUT} -m ${MODEL} -r ${SAMPLING} -s ${SIZE}

cd $FOLDER
