import cv2
import sys

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

t0=time.time()

while True:

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
    
    

    mouths = mouthCascade.detectMultiScale(
        face_img,
        scaleFactor=1.8,
        minNeighbors=4,
        minSize=(3, 3),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    ) 

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    mouths_filtered=[m for m in mouths if m[1]>hf/2]
#    mouths_filtered=[m for m in mouths if m[1]>hf/2 and wf/3.0>m[0]+m[2]/2>2.0*wf/3.0]
#    for (x,y,w,h) in mouths_filtered:
#        cv2.rectangle(frame,(x+xf,y+yf),(x+xf+w,y+yf+h),(0,0,255), 2)

    if len(mouths_filtered) is not 0:
        biggest_mouth=sorted(mouths_filtered,key=lambda x:x[-1]*x[-2])[-1]
        x,y,w,h=biggest_mouth
        cv2.rectangle(frame,(x+xf,y+yf),(x+xf+w,y+yf+h),(0,0,255), 2)
    

#     Display the resulting frame
    cv2.imshow('Video', frame)
#    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

cv2.waitKey(0)#needed to make "q" close the window
cv2.waitKey(0)
cv2.waitKey(0)
cv2.waitKey(0)#rediculous


