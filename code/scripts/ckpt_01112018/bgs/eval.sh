for model in nasnet_large resnet_v1_101 resnet_v1_50 inception_v2 mobilenet_v1; do
    echo "########### START "${model}" ##############"
    python tensorflow_classifier.py -e ~/workspace/scripts/bgs/output_new/log.txt -i ~/workspace/scripts/bgs/output_new/images/ -o ~/workspace/scripts/bgs/output_new/predictions.txt -m $model > log_${model}.txt
    echo "########### END "${model}" ##############"
done
for model in inception_v2 inception_v3 inception_v4 resnet_v1_152 mobilenet_v1_025 mobilenet_v1_050 inception_resnet_v2 nasnet_mobile; do
    echo "########### START "${model}" ##############"
    python tensorflow_classifier.py -e ~/workspace/scripts/bgs/output_new/log.txt -i ~/workspace/scripts/bgs/output_new/images/ -o ~/workspace/scripts/bgs/output_new/predictions.txt -m $model > log_${model}.txt
    echo "########### END "${model}" ##############"
done
