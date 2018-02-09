#/bin/bash

camera=$1
begin=$2
step=$3
range=$4

hourlist=(07_11 07_14 07_15 07_16 07_17 07_18 07_19 07_21 07_22 07_23 08_00 08_01 08_02 08_03 08_04 08_05 08_06 08_07 08_08 08_09)

for ((index=$begin;index<$begin+$step*$range;index+=$step)); do
    hour=${hourlist[index]}
    video=${camera}__2017-04-${hour}
    echo $video
    #sh run_all.sh $video
    #sh run_light.sh $video
    sh run_threeway.sh $video
done
