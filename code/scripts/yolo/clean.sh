#sudo cp -r /home/junchenj/data/Result/ /home/junchenj/data/Result_back/
#suco cp log log_back

for name in Result Result_Odd Result_38; do

dir=/home/junchenj/data/$name/predictions_logs/

find $dir -type f -size -40 -ls | awk ' {print $11} ' > tmp.txt
while read -r line
do
	ls -l $line
	sudo rm $line
done < tmp.txt
done
