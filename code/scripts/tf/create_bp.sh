#!/bin/bash
size=$1
newdir=$2
model=$3
rm -r ${newdir}
mkdir ${newdir}
root=/mnt/detectors
model_dir=${root}/${model}
cp ${model_dir}/pipeline.config tmp.config
if [ "$model" = "faster_rcnn_nas_coco_2017_11_08" ]
then
    sed -i 's$height: 1200$height: '$size'$g' tmp.config
    sed -i 's$width: 1200$width: '$size'$g' tmp.config
elif [ "$model" = "ssd_inception_v2_coco_2017_11_17" ]
then
    sed -i 's$height: 300$height: '$size'$g' tmp.config
    sed -i 's$width: 300$width: '$size'$g' tmp.config
elif [ "$model" = "ssd_mobilenet_v1_coco_2017_11_17" ]
then
    sed -i 's$height: 300$height: '$size'$g' tmp.config
    sed -i 's$width: 300$width: '$size'$g' tmp.config
else
    sed -i 's$min_dimension: 600$min_dimension: '$size'$g' tmp.config
    sed -i 's$max_dimension: 1024$max_dimension: '$((2 * $size))'$g' tmp.config
fi
python export_inference_graph.py --input_type image_tensor --pipeline_config_path tmp.config --trained_checkpoint_prefix ${model_dir}/model.ckpt --output_directory ${newdir}
