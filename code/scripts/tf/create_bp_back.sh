path=/home/junchenj/VideoAdaptiveProfiling/code/scripts/tf
oldmodel_path=/home/junchenj/VideoAdaptiveProfiling/code/scripts/tf/faster_rcnn_resnet50_coco_2017_11_08
for size in 400 600 800 1000 1200; do
#for size in 400 ; do
    newdir=${path}/resized_${size}
    rm -r ${newdir}
    mkdir ${newdir}
    cp ${oldmodel_path}/pipeline.config tmp.config
    sed -i 's$min_dimension: 600$min_dimension: '$size'$g' tmp.config
    sed -i 's$max_dimension: 1024$max_dimension: '$((2 * $size))'$g' tmp.config
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path tmp.config --trained_checkpoint_prefix ${oldmodel_path}/model.ckpt --output_directory ${newdir}
done
