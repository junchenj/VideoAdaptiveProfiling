#/bin/bash

PATH_TO_ORIGINAL_VIDEO=""
start_time_ms=0
end_time_ms=0
PATH_TO_FRAMES=""
PATH_TO_ORIGINAL_MODEL=""
PATH_TO_NEW_MODEL=""
img_size=""
frame_rate=""
model_name=""
OUTPUT_FILE=""

if [ "$#" -lt 20 ]; then
	echo "Error: need at least TEN arguments"
	echo "Format: sh cml_preprocess.sh --video PathToOriginalVideo \
	--start StartTimeStamp --end EndTimeStamp \
	--frames PathToFrames --model PathToOriginalModel --newmodel PathToNewModel \
	--size ImageSize --framerate FrameRate --modelname ModelName --output OutputFile"
	exit
fi

while test $# -gt 0; do
	case "$1" in
		-h|--help)
			echo "Format: sh cml_preprocess.sh --video PathToOriginalVideo 
			--start StartTimeStamp --end EndTimeStamp 
			--frames PathToFrames --model PathToOriginalModel --newmodel PathToNewModel 
			--size ImageSize --framerate FrameRate --modelname ModelName --output OutputFile"
			echo "options:"
			echo "-h --help         show brief help"
			echo "-v --video        path to original video"
			echo "-t --start        start time in ms"
			echo "-e --end          end time in ms"
			echo "-f --frames       path to output frames"
			echo "-m --model        path to the original model"
			echo "-u --newmodel     path to the new model"
			echo "-s --size         the new image size"
			echo "-r --framerate    the new frame rate"
			echo "-n --modelname    the model name"
			echo "-o --output       the output file"
			exit 0
			;;
		-v|--video)
			PATH_TO_ORIGINAL_VIDEO=$2
			shift
			;;
		-t|--start)
			start_time_ms=$2
			shift
			;;
		-e|--end)
			end_time_ms=$2
			shift
			;;
		-f|--frames)
			PATH_TO_FRAMES=$2
			shift
			;;
		-m|--model)
			PATH_TO_ORIGINAL_MODEL=$2
			shift
			;;
		-u|--newmodel)
			PATH_TO_NEW_MODEL=$2
			shift
			;;
		-s|--size)
			img_size=$2
			shift
			;;
		-r|--framerate)
			frame_rate=$2
			shift
			;;
		-n|--modelname)
			model_name=$2
			shift
			;;
		-o|--output)
			OUTPUT_FILE=$2
			shift
			;;
		-*)
			echo "invalid option "$1
			exit 0
			;;
		*)
			shift
			;;
	esac
done

echo "**********************************************************************"
echo "*         Preprocessing video segment and inference model            *"
echo "**********************************************************************"
echo "Path to original video:   "$PATH_TO_ORIGINAL_VIDEO
echo "Start time millisecond:   "$start_time_ms
echo "End time millisecond:     "$end_time_ms
echo "Path to output frames:    "$PATH_TO_FRAMES
echo "Path to original model:   "$PATH_TO_ORIGINAL_MODEL
echo "Path to new model:        "$PATH_TO_NEW_MODEL
echo "Image size:               "$img_size
echo "Frame rate:               "$frame_rate
echo "Model name:               "$model_name
echo "Output file:              "$OUTPUT_FILE
echo "**********************************************************************"

PATH_TO_TEMP=/mnt/tmp
mkdir -p $PATH_TO_TEMP



# TRANSFORM VIDEO
echo "****************** Getting video frames...\n"
if [ ! -d "$PATH_TO_FRAMES" ]; then
#rm -rf $PATH_TO_FRAMES
mkdir -p $PATH_TO_FRAMES
PATH_TO_SEGMENT=${PATH_TO_TEMP}/segment.mp4
s_ms=$(( $start_time_ms % 1000 ))
s_ss=$(((( $start_time_ms - $s_ms ) % 60000) /1000 ))
s_mm=$(((( $start_time_ms - $s_ms - $s_ss * 1000 ) % 3600000 ) / 60000 ))
s_hh=$(((( $start_time_ms - $s_ms - $s_ss * 1000 - $s_mm * 60000 )) / 3600000 ))
dur_time_ms=$(( $end_time_ms - $start_time_ms ))
d_ms=$(( $dur_time_ms % 1000 ))
d_ss=$(((( $dur_time_ms - $d_ms ) % 60000) /1000 ))
d_mm=$(((( $dur_time_ms - $d_ms - $d_ss * 1000 ) % 3600000 ) / 60000 ))
d_hh=$(((( $dur_time_ms - $d_ms - $d_ss * 1000 - $d_mm * 60000 )) / 3600000 ))
echo "****************** Saving segment to "${PATH_TO_SEGMENT}"...\n"
#ffmpeg -y -i $PATH_TO_ORIGINAL_VIDEO \
ffmpeg -loglevel quiet -y -i $PATH_TO_ORIGINAL_VIDEO \
	-ss ${s_hh}:${s_mm}:${s_ss}.${s_ms} \
	-strict -2 \
	-t ${d_hh}:${d_mm}:${d_ss}.${d_ms} \
	$PATH_TO_SEGMENT
	#-c copy $PATH_TO_SEGMENT
echo "****************** Saving frames to "${PATH_TO_FRAMES}/"...\n"
#ffmpeg -y -i $PATH_TO_SEGMENT ${PATH_TO_FRAMES}/Frame_TS_${start_time_ms}_%08d.jpg
ffmpeg -loglevel quiet -y -i $PATH_TO_SEGMENT ${PATH_TO_FRAMES}/Frame_TS_${start_time_ms}_%08d.jpg
#rm $PATH_TO_SEGMENT
else
echo "****************** PATH_TO_FRAMES already exists: \n"${PATH_TO_FRAMES}" \n"
fi
echo "****************** Done ******************\n\n\n"



# CREATING THE NEW MODEL
echo "****************** Preparing the new model...\n"
if [ ! -d "$PATH_TO_NEW_MODEL" ]; then
#rm -rf $PATH_TO_NEW_MODEL
mkdir -p $PATH_TO_NEW_MODEL
PATH_TO_TMPCONFIG=${PATH_TO_TEMP}/tmp.config
echo "****************** Saving temp config to "${PATH_TO_TMPCONFIG}"...\n"
cp ${PATH_TO_ORIGINAL_MODEL}/pipeline.config $PATH_TO_TMPCONFIG
if [ "$model_name" = "faster_rcnn_nas_coco_2017_11_08" ]
then
	sed -i 's$height: 1200$height: '$img_size'$g' $PATH_TO_TMPCONFIG
	sed -i 's$width: 1200$width: '$img_size'$g' $PATH_TO_TMPCONFIG
elif [ "$model_name" = "ssd_inception_v2_coco_2017_11_17" ]
then
	sed -i 's$height: 300$height: '$img_size'$g' $PATH_TO_TMPCONFIG
	sed -i 's$width: 300$width: '$img_size'$g' $PATH_TO_TMPCONFIG
elif [ "$model_name" = "ssd_mobilenet_v1_coco_2017_11_17" ]
then
	sed -i 's$height: 300$height: '$img_size'$g' $PATH_TO_TMPCONFIG
	sed -i 's$width: 300$width: '$img_size'$g' $PATH_TO_TMPCONFIG
else
	sed -i 's$min_dimension: 600$min_dimension: '$img_size'$g' $PATH_TO_TMPCONFIG
	sed -i 's$max_dimension: 1024$max_dimension: '$((2 * $img_size))'$g' $PATH_TO_TMPCONFIG
fi
echo "****************** Creating new model by config "${PATH_TO_TMPCONFIG}"...\n"
python export_inference_graph.py --input_type image_tensor \
	--pipeline_config_path $PATH_TO_TMPCONFIG \
	--trained_checkpoint_prefix ${PATH_TO_ORIGINAL_MODEL}/model.ckpt \
	--output_directory ${PATH_TO_NEW_MODEL}
else
echo "****************** PATH_TO_NEW_MODEL already exists: \n"${PATH_TO_NEW_MODEL}" \n"
fi
echo "****************** Done ****************** \n\n\n"




# ACTUAL INFERENCE
#echo "****************** Start inference... \n"
if [ ! -d "$(dirname "$OUTPUT_FILE")" ]; then
mkdir -p "$(dirname "$OUTPUT_FILE")"
fi
#if [ ! -d "$OUTPUT_FILE" ]; then
#touch $OUTPUT_FILE
#python cml_inference.py --tfpath ~/workspace/tensorflow/models/ \
#	--frames $PATH_TO_FRAMES --output $OUTPUT_FILE \
#	--modelpath $PATH_TO_NEW_MODEL \
#	--modelname $model_name --framerate $frame_rate --size $img_size \
#	--showframes False
#fi
#echo "****************** Done ****************** \n\n\n"
