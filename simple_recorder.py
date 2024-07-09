import cv2
import sys

#set framerate frome command line
#v4l2-ctl -d 1 --set-parm=60

#cascPath = sys.argv[1]
facePath='CascadeModels/haarcascade_frontalface_default.xml'
mouthPath='CascadeModels/Mouth.xml'

'CascadeModels/haarcascade_frontalface_default.xml'
facePath='/home/christopher/Documents/Classes/EE371R/ClassProject/CascadeModels/haarcascade_frontalface_default.xml'

mouthPath='/home/christopher/Documents/Classes/EE371R/ClassProject/CascadeModels/Mouth.xml'

faceCascade = cv2.CascadeClassifier(facePath)
mouthCascade= cv2.CascadeClassifier(mouthPath)

video_capture = cv2.VideoCapture(0)

import time
#cv2.startWindowThread()

#t0=time.time()

num_frames = 0
start = time.time()

while True:

#def grab_frame():
#    time.sleep(5)
    print 'grab_frame()'
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if frame is None:
        print 'failed to capture frame'
#        break


    if num_frames < 50:
        num_frames = num_frames + 1;
        print 'num frames=',num_frames
    else:
        break



#     Display the resulting frame
#    cv2.imshow('Video', frame)
    
    
#    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):#affects the fps
        break
#        return

total_time = (time.time() - start)
fps = (num_frames / total_time)
print str(num_frames) + ' frames in ' + str(total_time) + ' seconds = ' + str(fps) + ' fps'

#grab_frame()

#from scheduler import my_scheduler
#import time
#scheduler = my_scheduler(time.time, time.sleep)
#
#scheduler.new_timed_call(2,grab_frame)
#
#scheduler.run()
#
#
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

cv2.waitKey(0)#needed to make "q" close the window
cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)#rediculous