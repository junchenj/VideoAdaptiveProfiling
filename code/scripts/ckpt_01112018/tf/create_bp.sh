for size in 400 600 800 1000 1200; do
	python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./pipeline_$size.config --trained_checkpoint_prefix ./faster_rcnn_resnet50_coco_2017_11_08/model.ckpt --output_directory ./fine_tuned_model_$size
done
