import numpy as np
import cv2
print("CV2 version: "+cv2.__version__)
import sys, getopt, os
import ntpath

EXTRACTION_LOG = ""
IMAGES_FOLDER = ""
OUTPUT_FILE = ""
DARKNET_ROOT = ""
MODEL = ""

opts, args = getopt.getopt(sys.argv[1:],"e:i:o:d:m:")
for o, a in opts:
    if o == '-e':
        EXTRACTION_LOG = a
    elif o == '-i':
        IMAGES_FOLDER = a
    elif o == '-o':
        OUTPUT_FILE = a
    elif o == '-d':
        DARKNET_ROOT = a
    elif o == '-m':
        MODEL = a
    else:
        print("Usage: %s -e extraction -i images -o output -d darknet -m model" % sys.argv[0])
        sys.exit()
if (not EXTRACTION_LOG):
    print("Missing arguments -e")
    sys.exit()
if (not IMAGES_FOLDER):
    print("Missing arguments -i")
    sys.exit()
if (not OUTPUT_FILE):
    print("Missing arguments -o")
    sys.exit()
if (not DARKNET_ROOT):
    print("Missing arguments -d")
    sys.exit()
if (not MODEL):
    print("Missing arguments -m")
    sys.exit()

print "***********************************"
print "Extraction:\t"+EXTRACTION_LOG
print "Images path:\t"+IMAGES_FOLDER
print "Output file:\t"+OUTPUT_FILE
print "Darknet root:\t"+DARKNET_ROOT
print "Model name:\t"+MODEL
print "***********************************"

TEMP_FOLDER = "tmp_dir"
os.system("rm -r "+TEMP_FOLDER)
os.system("mkdir "+TEMP_FOLDER)

def process(image_id_to_coordinates, output):
#    print(os.path.join(DARKNET_ROOT,"darknet")+" classifier predict-batch"\
#            " "+os.path.join(DARKNET_ROOT,"cfg/imagenet1k.data")+\
#            " "+os.path.join(DARKNET_ROOT,"cfg/resnet152.cfg")+\
#            " "+os.path.join(DARKNET_ROOT,"models/resnet152.weights")+\
#            " "+TEMP_FOLDER)  
    current_path = os.getcwd()
    os.system("cd "+DARKNET_ROOT+";"+\
            "./darknet classifier predict-batch"+\
            " "+"cfg/imagenet1k.data"+\
            " "+"cfg/resnet152.cfg"+\
            " "+"models/resnet152.weights"+\
            " "+os.path.join(current_path, TEMP_FOLDER)+"/;"+\
            "cd "+current_path)
    os.system("rm -r "+TEMP_FOLDER)
    os.system("mkdir "+TEMP_FOLDER)

log = open(EXTRACTION_LOG, "r")
split_index_list = []
lines = []
count = 0
for line in log:
    if line.startswith('FrameID'):
        split_index_list.append(count)
    lines.append(line)
    count += 1
split_index_list.append(count)

output = open(OUTPUT_FILE, "w")
output.write("")
output.close()
output = open(OUTPUT_FILE, "a")


batch_size = 5000
count = 0
image_id_to_coordinates = {}
for i in range(len(split_index_list)-1):
    frame_id = ntpath.basename(lines[split_index_list[i]])
    output.write("FrameID="+frame_id)
    num_images = split_index_list[i+1]-split_index_list[i]-1
    if num_images == 0: continue
    if count+num_images > batch_size:
        # run darknet
        print("###### Processing "+str(len(image_id_to_coordinates))+" images ########")
        print("###### Until "+str(i)+"/"+str(len(split_index_list)-1)+" frames ########")
        process(image_id_to_coordinates, output)
        count = 0
        image_id_to_coordinates.clear()
    count = count+num_images
    for j in range(num_images):
        line = lines[split_index_list[i]+j+1]
        fields = line.split("\t")
        image_id = ntpath.basename(fields[0])
        coordinates = fields[1]+"\t"+fields[2]+"\t"+fields[3]+"\t"+fields[4]
        image_id_to_coordinates[image_id] = coordinates
        #print("cp "+os.path.join(IMAGES_FOLDER,image_id)+" "+TEMP_FOLDER)
        os.system("cp "+os.path.join(IMAGES_FOLDER,image_id)+" "+TEMP_FOLDER) 
process(image_id_to_coordinates, output)
        
        
        
        
        




