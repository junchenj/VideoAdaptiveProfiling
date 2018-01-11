DATASET_DIR=/tmp/images/
EVAL_DIR=/tmp/eval/
CHECKPOINT_DIR=/tmp/checkpoints/model.ckpt
PWD=`$pwd`
cd ~/workspace/tensorflow/models/
python research/slim/eval_image_classifier.py \
--checkpoint_path=${CHECKPOINT_DIR} \
--eval_dir=${EVAL_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=imagenet \
--dataset_split_name=validation \
--model_name=nasnet_mobile \
--eval_image_size=224 \
--moving_average_decay=0.9999
cd $PWD
