PWD=`pwd`
cd ../tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
cd $PWD
