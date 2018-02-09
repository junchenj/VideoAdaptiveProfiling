PWD=`pwd`

sudo apt-get update
sudo apt-get install --assume-yes gcc
sudo apt-get install --assume-yes g++
sudo apt-get install --assume-yes make
sudo apt-get install --assume-yes ffmpeg
sudo apt-get install --assume-yes build-essential

#wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
#sudo sh cuda_8.0.61_375.26_linux-run --override

#wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
#sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install --assume-yes cuda

#echo "export PATH=/usr/local/cuda-8.0/bin:$PATH" >> ~/.bashrc
#echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
#export PATH=/usr/local/cuda-8.0/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
echo "export PATH=/usr/local/cuda-9.0/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

#echo download and install cudnn libcudnn6_6.0.21-1+cuda8.0_amd64.deb!!!
#sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
#echo "export CUDA_HOME=/usr/local/cuda-8.0" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda-9.0" >> ~/.bashrc
#export CUDA_HOME=/usr/local/cuda-8.0
export CUDA_HOME=/usr/local/cuda-9.0

sudo apt-get install --assume-yes libcupti-dev
sudo apt-get install --assume-yes python-pip python-dev
pip install tensorflow-gpu


sudo apt-get install --assume-yes protobuf-compiler python-pil python-lxml
pip install --upgrade pip
sudo pip install jupyter
sudo pip install matplotlib
sudo pip install opencv-contrib-python

git clone https://github.com/tensorflow/models.git
cd models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo "export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim" >> ~/.bashrc

sudo apt-get install --assume-yes python-tk


# RUN THIS CODE UP ON EACH RESTART
protoc object_detection/protos/*.proto --python_out=.

#  TEST
python object_detection/builders/model_builder_test.py

cd $PWD
