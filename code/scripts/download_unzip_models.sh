export list="
faster_rcnn_nas_coco_2017_11_08 
faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08 
faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2017_11_08 
faster_rcnn_resnet101_coco_2017_11_08 
rfcn_resnet101_coco_2017_11_08 
faster_rcnn_resnet50_coco_2017_11_08 
faster_rcnn_resnet50_lowproposals_coco_2017_11_08 
faster_rcnn_inception_v2_coco_2017_11_08 
ssd_inception_v2_coco_2017_11_17 
ssd_mobilenet_v1_coco_2017_11_17 "

FOLDER=$(pwd)
echo $FOLDER
rm -r /mnt/detectors
mkdir /mnt/detectors
cd /mnt/detectors
for name in $list; do
    echo Loading and unzip $name
    wget http://download.tensorflow.org/models/object_detection/${name}.tar.gz
    tar -xzvf ${name}.tar.gz
done
wget -O /mnt/detectors/ssd_inception_v2_coco_2017_11_17/pipeline.config https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config
wget -O /mnt/detectors/ssd_mobilenet_v1_coco_2017_11_17/pipeline.config https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config

chmod 777 -R /mnt/detectors
cd $FOLDER
