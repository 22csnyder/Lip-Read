# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 21:12:01 2015

@author: christopher
"""
import cv2
import time
import numpy as np

def recording_loop(Writer):
    
    facePath='CascadeModels/haarcascade_frontalface_default.xml'
    mouthPath='CascadeModels/Mouth.xml'
    
    'CascadeModels/haarcascade_frontalface_default.xml'
    facePath='/home/christopher/Documents/Classes/EE371R/ClassProject/CascadeModels/haarcascade_frontalface_default.xml'
    
    mouthPath='/home/christopher/Documents/Classes/EE371R/ClassProject/CascadeModels/Mouth.xml'
    
    faceCascade = cv2.CascadeClassifier(facePath)
    mouthCascade= cv2.CascadeClassifier(mouthPath)
    
    video_capture = cv2.VideoCapture(0)
    


    frame_shape=(38,62)
#    num_frames = -1
    start = time.time()
    while True:

#        if num_frames < 50:
#            print 'num frames=',num_frames
#            num_frames+=1
#        else:
#            break
    
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
        else:
            print 'no face'
            continue#try again
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)    
    
        mouths = mouthCascade.detectMultiScale(
            face_img,
#            scaleFactor=2.0,
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
        else:
            print 'face but no mouth'
            continue
            
        save_frame=cv2.resize(color_mouth,frame_shape[::-1])
        Writer.write(save_frame)

[3:35,15:47]
        
        
#        num_frames = num_frames + 1;
    
    #     Display the resulting frame
        cv2.imshow('Video', frame)
    #    cv2.waitKey(0)
        
        k=cv2.waitKey(1) & 0xFF
        
        if Writer.quit_loop_condition(k):
            break
        
        Writer.check_conditions(k)        
        
#        if cv2.waitKey(1) & 0xFF == ord('q'):#affects the fps
#            break
        
        
    #        return
    
#    total_time = (time.time() - start)
#    fps = (num_frames / total_time)
#    print str(num_frames) + ' frames in ' + str(total_time) + ' seconds = ' + str(fps) + ' fps'
    
    #
    # When everything is done, release the capture
    
    #video_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()
    
    cv2.waitKey(0)#needed to make "q" close the window
    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.waitKey(0)#rediculous




import os
import errno
def create_directory_safely(path):
    try:
        os.makedirs(path)
        return True
    except OSError as exception:
        print 'Directory already exists!'
        if exception.errno != errno.EEXIST:
            raise
        return False


default_data_dir='/home/christopher/Documents/Classes/EE371R/ClassProject/Data/'
class Writer:
    
    def __init__(self,data_directory=default_data_dir):
        self.data_directory=data_directory
        self.current_data=None
        
        self.hotkey_dict=dict()
        
    def new_data_set(self,name):
        self.current_data=name
        self.current_dir=self.data_directory+self.current_data
        
#        self.hotkey_dict[ord(self.current_data[0])]=self.current_dir
        
        return create_directory_safely(self.current_dir)

    
    def start_recording(self):
        self.video_number=0
        self.frame_number=0
        self.current_video_folder=self.current_dir+'/'+str(self.video_number)
        if create_directory_safely(self.current_video_folder):
            recording_loop(self)

    def write(self,frame):
        
        cv2.imwrite(self.current_video_folder+'/mouth'+str(self.frame_number)+'.tiff',frame)      
        self.frame_number+=1
        
    def quit_loop_condition(self,key):
        if key == ord('q'):#affects the fps
            return True
    
    def next_data_condition(self,key):
        if key == ord('n'):
            self.video_number+=1
            print 'frames in last video',self.frame_number,'..starting video number ',self.video_number
            self.current_video_folder=self.current_dir+'/'+str(self.video_number)
            create_directory_safely(self.current_video_folder)
            self.frame_number=0
    def check_conditions(self,key):
        self.next_data_condition(key)
        
        
        
        
if __name__=='__main__':
    

    writer=Writer()
    if writer.new_data_set('just_yellow'):
        writer.start_recording()
        
        
        
        
        
        