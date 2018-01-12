#/bin/bash

camera=$1
begin=$2
step=$3
range=$4

exp_name=$5
mod=10

sudo mkdir /home/junchenj/data/

videos_folder=/home/junchenj/videos/
root=/home/junchenj/data/$exp_name
frames_folder=$root/frames/
frames_input_folder=$root/frames_input/
predictions_logs_folder=$root/predictions_logs/
predictions_imgs_folder=$root/predictions_imgs/
default_video_file=notfound
darknet_dir=/home/junchenj/darknet/

sudo rm -r $root
sudo mkdir $root
sudo rm -r $predictions_logs_folder
sudo mkdir $predictions_logs_folder
sudo rm -r $predictions_imgs_folder
sudo mkdir $predictions_imgs_folder

hourlist=(07_11 07_14 07_15 07_16 07_17 07_18 07_19 07_21 07_22 07_23 08_00 08_01 08_02 08_03 08_04 08_05 08_06 08_07 08_08 08_09)

echo ${#hourlist[@]}

count=0
#for ((i=$begin;i<${#hourlist[@]};i+=$step)); do
for ((i=$begin;i<$begin+$step*$range;i+=$step)); do
	hour=${hourlist[i]}
	video=${camera}_${hour}
	video_file=$default_video_file
	for f in $videos_folder/*mp4; do
		name=$(basename $f)
		if [[ $name == *"${camera}"*"${hour}"* ]]; then video_file=$name; fi
	done
	if [ "$video_file" == "$default_video_file" ]; then continue; fi
	count=$(($count+1))
        echo ~~~~~~~~~~~~~~~~~~~~~~~~ $video_file ~~~~~~~~~~~~~~~~~~~~~~~~
        for b in 0.2; do
                sudo rm -r $frames_folder
                sudo mkdir $frames_folder
                #sudo ffmpeg -loglevel quiet -i $videos_folder/$video_file -vf eq=brightness=$b $frames_folder/tmp.mp4
                sudo ffmpeg -i $videos_folder/$video_file -vf eq=brightness=$b $frames_folder/tmp.mp4
                sudo rm -r $frames_input_folder
                sudo mkdir $frames_input_folder
                #sudo ffmpeg -loglevel quiet -i $frames_folder/tmp.mp4 $frames_input_folder/out-%06d.jpg
                sudo ffmpeg -i $frames_folder/tmp.mp4 $frames_input_folder/out-%06d.jpg
                for ((i=0;i<$mod;i+=1)); do
                        sudo rm -r ${frames_input_folder}/$i
                        sudo mkdir ${frames_input_folder}/$i
                        for ((j=0;j<=9;j+=1)); do
                                if [ $(($j%$mod)) == $i ]; then
                                        sudo mv ${frames_input_folder}/*${j}.jpg ${frames_input_folder}/$i/
                                fi
                        done
                done
                for ((s=1280;s>=320;s-=192)); do
                        for ((i=0;i<$mod;i+=1)); do
                                echo ============= testing $video size $s bright $b batch $i ============
                                cd $darknet_dir
                                cp cfg/yolo.cfg cfg/custom.cfg
                                sed -i 's/width=608/width='$s'/g' cfg/custom.cfg
                                sed -i 's/height=608/height='$s'/g' cfg/custom.cfg
                                rm -r $exp_name
                                mkdir $exp_name
                                cd $darknet_dir
                                ./darknet detector test-batch cfg/coco.data cfg/custom.cfg yolo.weights $frames_input_folder/$i/ -thresh 0.05 -outfolder $exp_name > out_bellevue_${exp_name}.txt
                                sudo mv $exp_name $predictions_imgs_folder/imgs_${video}_bright_${b}_size_${s}_batch_${i}
                                sudo mv out_bellevue_${exp_name}.txt $predictions_logs_folder/logs_${video}_bright_${b}_size_${s}_batch_${i}
                        done
                done
        done
done
echo $count
