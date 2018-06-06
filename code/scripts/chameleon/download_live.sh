url=$1
save_file=$2

if [ -z "$url" ]
then
        echo Please specify youtube url e.g., https://www.youtube.com/watch?v=qHqIjDoMYAk
        exit 0
fi

if [ -z "$save_file" ]
then
        echo Please specify save file
        exit 0
fi

rm $save_file

line=$(youtube-dl --list-formats $url | tail -1)
echo "$line"
id="$(echo $line | cut -d' ' -f1)"
echo $id
manifest_url=$(youtube-dl -f $id -g $url)
echo $manifest_url
ffmpeg -i $manifest_url -c copy $save_file
