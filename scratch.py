
import theano
import theano.tensor as T
import numpy as np

def floatX(arr):
    return np.array(arr,dtype=np.float32)
theano.config.compute_test_value='warn'#instead of 'off'
v=T.vector('v')

a=floatX(np.array([3.0,2.0,1.0]))
v.tag.test_value=a

th=T.tanh(v)#==np.tanh



from theano.tensor.signal.downsample import max_pool_2d


A=floatX(np.random.randint(0,10,(12,12)))

im=T.matrix('im')
im.tag.test_value=A


mp=max_pool_2d(im,(2,2))



A=A.astype(np.int)
import skimage.measure as m
m.block_reduce(A,(2,2),np.max)


A32=floatX(np.random.randint(0,10,(32,32)))
Im=T.tensor4('Im')
Im.tag.test_value=floatX(A32[np.newaxis,np.newaxis,:])
#F=floatX(np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,1,0],[0,0,0,0,0]])[np.newaxis,np.newaxis,:])
F=floatX(np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])[np.newaxis,np.newaxis,:])
w=T.tensor4('w')
w.tag.test_value=F

'''
def conv(X, w, b, activation):
    # z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return activation(z)'''
from theano.tensor.nnet import conv2d

cnv=conv2d(Im,w)


Ai=A.astype(np.int)
cnvi=cnv.tag.test_value.astype(np.int)

#cv2.startWindowThread()

#t0=time.time()

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 18.0, (640,480))

#codec=cv2.cv.CV_FOURCC(*'H264')
#codec=cv2.cv.CV_FOURCC('F', 'M', 'P', '4')#no
#codec = cv2.cv.CV_FOURCC('I','4','2','0')
#codec = cv2.cv.CV_FOURCC('A','V','C','1')
#codec = cv2.cv.CV_FOURCC('Y','U','V','1')
#codec = cv2.cv.CV_FOURCC('P','I','M','1')
#codec = cv2.cv.CV_FOURCC('M','J','P','G')
#codec = cv2.cv.CV_FOURCC('M','P','4','2')
#codec = cv2.cv.CV_FOURCC('D','I','V','3')
#codec = cv2.cv.CV_FOURCC('D','I','V','X')
#codec = cv2.cv.CV_FOURCC('U','2','6','3')
#codec = cv2.cv.CV_FOURCC('I','2','6','3')
#codec = cv2.cv.CV_FOURCC('F','L','V','1')
#codec = cv2.cv.CV_FOURCC('H','2','6','4')
#codec = cv2.cv.CV_FOURCC('A','Y','U','V')
#codec = cv2.cv.CV_FOURCC('I','U','Y','V')
#codec=-1

#frame_shape=(38,62)
#video_writer = cv2.VideoWriter()
#fps=18
#fps=20
#video_writer.open("out.avi", codec, fps, frame_shape, True)
#video_writer.open("out.avi", codec, fps, (62,38), True)
#print v.isOpened()
