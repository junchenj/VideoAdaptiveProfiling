import numpy as np
import cv2
print("CV2 version: "+cv2.__version__)
import sys, getopt, os
import ntpath

class Cover_query:
    """Segment tree to maintain a set of integer intervals
    and permitting to query the size of their union.
    """
    def __init__(self, _len):
        """creates a structure, where all possible intervals
        will be included in [0, _len - 1].
        """
        assert _len != []
        self.N = 1
        while self.N < len(_len):
            self.N *= 2
        self.c = [0] * (2 * self.N)
        self.s = [0] * (2 * self.N)
        self.w = [0] * (2 * self.N)
        for i in range(len(_len)):
            self.w[self.N + i] = _len[i]
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def cover(self):
        """:returns: the size of the union of the stored intervals
        """
        return self.s[1]


    def change(self, i, k, delta):
        """when delta = +1, adds an interval [i, k], when delta = -1, removes it
        :complexity: O(log _len)
        """
        self._change(1, 0, self.N, i, k, delta)


    def _change(self, p, start, span, i, k, delta):
        if start + span <= i or k <= start:
            return
        if i <= start and start + span <= k:
            self.c[p] += delta
        else:
            self._change(2*p,     start,             span // 2, i, k, delta)
            self._change(2*p + 1, start + span // 2, span // 2, i, k, delta)
        if self.c[p] == 0:
            if p >= self.N:
                self.s[p] = 0
            else:
                self.s[p] = self.s[2 * p] + self.s[2 * p + 1]
        else:
            self.s[p] = self.w[p]

def union_rectangles(R):
    """Area of union of rectangles

    :param R: list of rectangles defined by (x1, y1, x2, y2)
       where (x1, y1) is top left corner and (x2, y2) bottom right corner
    :returns: area
    :complexity: :math:`O(n^2)`
    """
    if R == []:
        return 0
    X = []
    Y = []
    for j in range(len(R)):
        (x1, y1, x2, y2) = R[j]
        assert x1 <= x2 and y1 <= y2
        X.append(x1)
        X.append(x2)
        Y.append((y1, +1, j))
        Y.append((y2, -1, j))
    X.sort()
    Y.sort()
    X2i = {X[i]: i for i in range(len(X))}
    _len = [X[i + 1] - X[i] for i in range(len(X) - 1)]
    C = Cover_query(_len)
    area = 0
    last = 0
    for (y, delta, j) in Y:
        area += (y - last) * C.cover()
        last = y
        (x1, y1, x2, y2) = R[j]
        i = X2i[x1]
        k = X2i[x2]
        C.change(i, k, delta)
    return area

'''
# Testing union_rectangles. Correct answer = 48750
R = []
R.append((0,0,100,100))
R.append((50,50,150,150))
R.append((50,50,250,250))
R.append((25,50,75,150))
print str(union_rectangles(R))
'''

VIDEO_PATH = None
FRAMES_PATH = None
OUTPUT_PATH = None
MIN_AREA = None

# python finding_interesting_segments.py -i /mnt/videos/192.168.1.111-18-01-09_11.20.28.mp4 -o /mnt/tmp/diffs/ -m 0.0025

opts, args = getopt.getopt(sys.argv[1:],"i:f:o:m:")
for o, a in opts:
    if o == '-i':
        VIDEO_PATH = a
    elif o == '-f':
        FRAMES_PATH = a
    elif o == '-o':
        OUTPUT_PATH = a
    elif o == '-m':
        MIN_AREA = float(a)
    else:
        print("Wrong option "+o)
        print("Usage: %s -i inputvideo -f inputframes -o output -m minarea" % sys.argv[0])
        sys.exit()
if (VIDEO_PATH is None) and (FRAMES_PATH is None):
    print("Missing arguments -i OR -f")
    sys.exit()
if (OUTPUT_PATH is None):
    print("Missing arguments -o")
    sys.exit()
if (MIN_AREA is None):
    print("Missing arguments -m")
    sys.exit()

print "***********************************"
print "Video path:\t"+str(VIDEO_PATH)
print "Frames path:\t"+str(FRAMES_PATH)
print "Output path:\t"+str(OUTPUT_PATH)
print "Minim area:\t"+str(MIN_AREA)
print "***********************************"

if FRAMES_PATH is not None:
    FRAME_PATH_LIST = []
    for filename in os.listdir(FRAMES_PATH):
        if filename.endswith('.jpg'):
            FRAME_PATH_LIST.append(os.path.join(FRAMES_PATH,filename))
    FRAME_PATH_LIST.sort()

index = 0
#cap = cv2.VideoCapture('/home/junchenj/videos/Bellevue_116th_NE12th__2017-04-07_10-54-50.mp4')
cap = cv2.VideoCapture(VIDEO_PATH)
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

while True: 
    if VIDEO_PATH is not None:
        ret, frame = cap.read()
        if not ret: break
    if FRAMES_PATH is not None:
        if index >= len(FRAME_PATH_LIST): break
        frame_id = ntpath.basename(FRAME_PATH_LIST[index])
        frame_name, ext = os.path.splitext(frame_id)
        frame = cv2.imread(FRAME_PATH_LIST[index])

    [total_height,total_width,color] = frame.shape
    total_height = float(total_height)
    total_width = float(total_width)
    MinBoxArea = int(total_height*total_width*MIN_AREA)
    fgmask = fgbg.apply(frame)
    ret,th1 = cv2.threshold(fgmask,25,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    close_operated_image = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    _, thresholded = cv2.threshold(close_operated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median = cv2.medianBlur(thresholded, 5)
    th1 = median
    _,contours,hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    
    #output.write(VIDEO_PATH+"_"+str(index)+"\n")
    rectangle_list = []
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
        contour_name = VIDEO_PATH+"_"+str(index)+"_c_"+str(count)
        contour_id = contour_name+".jpg"
        contour_path = os.path.join(IMAGES_PATH, contour_id)
        rectangle_list.append((x,y,x+w,y+h))
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 5)
        '''
        cv2.imshow('obj',frame[y:y+h, x:x+w])
        print '\t\t\t\t\t'+str(x)+', '+str(y)+', '+str(w)+', '+str(h)
        cv2.imwrite(contour_path, frame[y:y+h, x:x+w])
        output.write(contour_id+\
                    "\t"+"{:.6f}".format(x/total_width)+\
                    "\t"+"{:.6f}".format(y/total_height)+\
                    "\t"+"{:.6f}".format(w/total_width)+\
                    "\t"+"{:.6f}".format(h/total_height)+"\n")
        '''
        count += 1
    #cv2.imshow('frame', frame)
    total_diff = union_rectangles(rectangle_list)
    print str(total_diff)
    mm = int(float(index/25.0/60.0))
    ss = int(float(index)/25.0-mm*60.0)
    ts = int((mm*25*60+ss*25)*1000)
    ff = index-mm*25*60-ss*25
    ms = int(float(ff)*1000.0/25.0)
    output.write(str(index)+"\t"+str(total_diff)+"\t"+str(count)+"\t|\t"+str(mm)+":"+str(ss)+":"+str(ms)+"\t|\t"+str(ts)+"#"+str(ff)+"\n")
    if index % 25 == 0:
        print "index="+str(index)+" "+VIDEO_PATH+"_"+str(index)+" count="+str(count)+" minArea="+str(MinBoxArea)
    
    k = cv2.waitKey(25) & 0xff
    if k == 27:
        break
#    cv2.imshow('Frame',frame)
#    cv2.imshow('Diff',fgmask)
#    cv2.imshow('Final',th1)

    index += 1

output.close()
cap.release()
cv2.destroyAllWindows()







