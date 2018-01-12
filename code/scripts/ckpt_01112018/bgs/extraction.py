import numpy as np
import cv2
print("CV2 version: "+cv2.__version__)
import sys, getopt, os
import ntpath

FRAMES_PATH = ""
OUTPUT_PATH = ""

opts, args = getopt.getopt(sys.argv[1:],"f:o:")
for o, a in opts:
    if o == '-f':
        FRAMES_PATH = a
    elif o == '-o':
	OUTPUT_PATH = a
    else:
        print("Usage: %s -f input -o output" % sys.argv[0])
        sys.exit()
if (not FRAMES_PATH):
    print("Missing arguments -f")
    sys.exit()
if (not OUTPUT_PATH):
    print("Missing arguments -o")
    sys.exit()

print "***********************************"
print "Frames path:\t"+FRAMES_PATH
print "Output path:\t"+OUTPUT_PATH
print "***********************************"

FRAME_PATH_LIST = []
for filename in os.listdir(FRAMES_PATH):
    if filename.endswith('.jpg'):
        FRAME_PATH_LIST.append(FRAMES_PATH+filename)
FRAME_PATH_LIST.sort()

index = 0
cap = cv2.VideoCapture('/home/junchenj/videos/Bellevue_116th_NE12th__2017-04-07_10-54-50.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

os.system('rm -r '+OUTPUT_PATH)
os.system('mkdir '+OUTPUT_PATH)
IMAGES_PATH = os.path.join(OUTPUT_PATH, "images/")
os.system('mkdir '+IMAGES_PATH)

LOG_FILE = os.path.join(OUTPUT_PATH, "log.txt")
output = open(LOG_FILE, "w")
output.write("")
output.close()
output = open(LOG_FILE, "a")

while(index < len(FRAME_PATH_LIST)):
    #ret, frame = cap.read()
    frame_id = ntpath.basename(FRAME_PATH_LIST[index])
    frame_name, ext = os.path.splitext(frame_id)
    print FRAME_PATH_LIST[index]
    frame = cv2.imread(FRAME_PATH_LIST[index])

    [total_height,total_width,color] = frame.shape
    total_height = float(total_height)
    total_width = float(total_width)
    MinBoxArea = int(total_height*total_width*1.0/100.0)
    fgmask = fgbg.apply(frame)
    ret,th1 = cv2.threshold(fgmask,25,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    close_operated_image = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    _, thresholded = cv2.threshold(close_operated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = cv2.medianBlur(thresholded, 5)
    th1 = median
    _,contours,hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    
    output.write(FRAME_PATH_LIST[index]+"\n")
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # get center
        x = x+w/2
        y = y+h/2
        # expand to a square with min(w,h)
        w = max(w, h)
        h = max(w, h)
        w = int(min([w, 2*x, 2*(total_width-x)]))
        h = int(min([h, 2*y, 2*(total_height-y)]))
        # get corner with new h w
        x = int(x-w/2)
        y = int(y-h/2)
        if (w * h < MinBoxArea):
            continue;
        contour_name = frame_name+"_c_"+str(count)
        contour_id = contour_name+".jpg"
        contour_path = os.path.join(IMAGES_PATH, contour_id)
        #cv2.imshow('obj',frame[y:y+h, x:x+w])
        #print '\t\t\t\t\t'+str(x)+', '+str(y)+', '+str(w)+', '+str(h)
        #cv2.imwrite(contour_path, frame[y:y+h, x:x+w])
        output.write(contour_id+\
                    "\t"+"{:.6f}".format(x/total_width)+\
                    "\t"+"{:.6f}".format(y/total_height)+\
                    "\t"+"{:.6f}".format(w/total_width)+\
                    "\t"+"{:.6f}".format(h/total_height)+"\n")
        count += 1
    print "index="+str(index)+" count="+str(count)+" minArea="+str(MinBoxArea)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#    cv2.imshow('Frame',frame)
#    cv2.imshow('Diff',fgmask)
#    cv2.imshow('Final',th1)

    index += 1

output.close()
cap.release()
cv2.destroyAllWindows()







