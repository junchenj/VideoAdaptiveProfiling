WORKSPACE_PATH=$1
PWD=`pwd`

if [ -z "$WORKSPACE_PATH" ]
then
	echo Please specify WORKSPACE_PATH
	exit 0
fi

echo Creating workspace at $WORKSPACE_PATH

rm -rf $WORKSPACE_PATH
mkdir $WORKSPACE_PATH
cp -r tensorflow $WORKSPACE_PATH/
cp -r darknet $WORKSPACE_PATH/

cd $WORKSPACE_PATH
cd tensorflow
sh install_tensorflow.sh

cd ../darknet
make
wget https://pjreddie.com/media/files/yolo.weights

cd $PWD
