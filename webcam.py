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


import numpy as np
avg_shape=np.zeros(3)

num_frames = 0
start = time.time()
while True:

#def grab_frame():
#    time.sleep(5)
#    print 'grab_frame()'
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if frame is None:
        print 'failed to capture frame'
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
#        scaleFactor=1.1,#1.2 orriginally#how much image is scalled down each time to check for match
        scaleFactor=1.3,#1.2 orriginally#how much image is scalled down each time to check for match
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    if len(faces) is not 0:
        biggestface=sorted(faces,key=lambda x:-x[2]*x[3])[0]
        xf,yf,wf,hf=biggestface
        face_img=gray[yf:yf+hf,xf:xf+wf]
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)    

    mouths = mouthCascade.detectMultiScale(
        face_img,
        scaleFactor=1.8,
        minNeighbors=4,
        minSize=(10, 3),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    ) 
    mouths_filtered=[m for m in mouths if m[1]>hf/2]
    if len(mouths_filtered) is not 0:
        biggest_mouth=sorted(mouths_filtered,key=lambda x:x[-1]*x[-2])[-1]
        xm,ym,wm,hm=biggest_mouth
        cv2.rectangle(frame,(xm+xf,ym+yf),(xm+xf+wm,ym+yf+hm),(0,0,255), 2)
        color_mouth=frame[ym+yf:ym+yf+hm,xm+xf:xm+xf+wm,:]
        
        
#        save_frame=cv2.resize(color_mouth,frame_shape[::-1])
#    video_writer.write(color_mouth)


    if num_frames < 50:
        avg_shape+=color_mouth.shape
        num_frames = num_frames + 1;
        print 'num frames=',num_frames
    else:
        break

    

#     Display the resulting frame
    cv2.imshow('Video', frame)
#    cv2.waitKey(0)
#    if cv2.waitKey(1) & 0xFF == ord('q'):#affects the fps
#        break
    k=cv2.waitKey(1) & 0xFF
    if k== ord('q'):#affects the fps
        break
    elif k==ord('n'):
        print 'n is pressed'
    
#        return

total_time = (time.time() - start)
fps = (num_frames / total_time)
print str(num_frames) + ' frames in ' + str(total_time) + ' seconds = ' + str(fps) + ' fps'

#avg_shape/=num_frames
print 'avg_shape',avg_shape#### I guess we're using shape=(38,62,3)
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

#video_writer.release()
video_capture.release()
cv2.destroyAllWindows()

cv2.waitKey(0)#needed to make "q" close the window
cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)#rediculous