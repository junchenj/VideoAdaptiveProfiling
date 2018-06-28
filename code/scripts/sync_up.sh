#for server in 13.82.42.156 13.82.44.34 13.82.44.232 13.82.45.56 13.82.45.231 52.179.86.13 13.82.46.114 13.82.45.103; do
for server in 40.71.24.98; do
    #ssh-copy-id junchenj@$server
    #ssh -t junchenj@$server rm ~/workspace/scripts/
    scp -r ~/VideoAdaptiveProfiling/code/scripts junchenj@$server:~/workspace/
done

#for server in 13.90.86.109 40.71.5.238 40.71.7.119 52.170.250.51 52.226.134.245 40.121.222.123 52.168.35.121; do
#    scp -r ~/videos junchenj@$server:~/
#    scp -r ~/object-detection-crowdai junchenj@$server:~/
#done
