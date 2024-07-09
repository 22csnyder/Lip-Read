# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:41:51 2015

@author: csnyder
"""

from conv_deconv_vae import ConvVAE,floatX,imshow
import theano.tensor as T
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

data_dir='/work/03176/csnyder/EE371/Data/'
results_dir='/work/03176/csnyder/EE371/Results/'

tf = ConvVAE(image_save_root=results_dir,snapshot_file=results_dir+'/generic_speaking_5_7_8_1-22000_no-white_square_resized_snapshot.pkl')



maxtrX=254.0
#X=floatX(trX[22])[np.newaxis,:]/maxtrX



orig1=io.imread('/work/03176/csnyder/EE371/exp/image_correction/mouth1firstred53.tiff')[3:35,15:47]
shop1=io.imread('/work/03176/csnyder/EE371/exp/image_correction/mouth1firstred53_missing_quarter.tiff')[3:35,15:47]

#orig1=io.imread('/work/03176/csnyder/EE371/exp/image_correction/mouth15658.tiff')[3:35,15:47]
#shop1=io.imread('/work/03176/csnyder/EE371/exp/image_correction/onetooth.tif')[3:35,15:47]


X=floatX(orig1.transpose(2,0,1)[np.newaxis,:])/maxtrX
#X2=floatX(control.transpose(2,0,1)[np.newaxis,:])/maxtrX


#z=tf.encode(S[0])
#z=tf._z_given_x(S)

e =  floatX(np.ones((X.shape[0], tf.n_code)))
code_mu,code_log_sigma=tf._z_given_x(X)
Z = floatX(code_mu + np.exp(code_log_sigma) * e)
r=tf._x_given_z(Z)##This works
#r=tf._reconstruct(X, floatX(np.ones((X.shape[0], tf.n_code))))##And this works! yay!
im=r[0].transpose(1,2,0)



#io.imshow(im)
#r2=tf._reconstruct(X2, floatX(np.ones((X2.shape[0], tf.n_code))))
#im2=r2[0].transpose(1,2,0)

#title='corrupted image'
#fig,axes=plt.subplots(1,3)
#ax0,ax1,ax2=axes
#ax0.set_title('original image')
#ax0.set_axis_off()
#ax0.imshow(orig1)
#ax1.imshow(shop1)
#ax1.set_axis_off()
#ax1.set_title(title)
#ax2.imshow(im)
#ax2.set_axis_off()
#ax2.set_title('reconstructed')
#plt.show()




#title='corrupted image'
#fig,axes=plt.subplots(1,2)
#ax0,ax1,ax2=axes
#
#ax0.set_title('original image')
#ax0.set_axis_off()
#ax0.imshow(orig1)
#
#ax1.imshow(shop1)
#ax1.set_axis_off()
#ax1.set_title(title)
#
#ax2.imshow(im)
#ax2.set_axis_off()
#ax2.set_title('reconstructed')
#
#plt.show()


#e =  floatX(np.ones((X.shape[0], tf.n_code)))
#code_mu,code_log_sigma=tf._z_given_x(X)
##Z = code_mu + T.exp(code_log_sigma) * e
#Z = floatX(code_mu + np.exp(code_log_sigma) * e)
#
#x=tf._x_given_z(code_mu)
#x=tf._x_given_z(Z)
#im=x[0].transpose(1,2,0)
#
#r=tf._reconstruct(X,e)
#
#im=r[0].transpose(1,2,0)
#
#
#
##z=tf._z_given_x([orig1.transpose(2,0,1)])
###z=tf._z_given_x([shop1.transpose(2,0,1)])
##x=tf._x_given_z(z[1])
#
#xs=floatX(np.random.randn(100, tf.n_code))
#s=xs
#IM=tf._x_given_z(s)
#im=IM[22].transpose(1,2,0)###Works!





