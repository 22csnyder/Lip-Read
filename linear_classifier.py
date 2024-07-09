# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:18:17 2015

@author: csnyder
"""
import numpy as np
import matplotlib.pyplot as plt


data_dir='/work/03176/csnyder/EE371/Data/'
results_dir='/work/03176/csnyder/EE371/Results/'


#trX=np.load(data_dir+'generic_speaking_data_5_7_8.npy')
#trX=trX[:22000,:,3:35,15:47]
#trX=trX.astype(np.float)#maybe should have cast as floatX next time
#trX=trX/trX.max()

#from conv_deconv_vae import ConvVAE,floatX,color_grid_vis

#tf = ConvVAE(image_save_root=results_dir,
#             snapshot_file=results_dir+'/generic_speaking_5_7_8_1-22000_no-white_square_resized_snapshot.pkl')


#trX=trX[:22000,:,3:35,15:47]


####Draw 4 sample Images####
#xs = floatX(np.random.randn(4, tf.n_code))
#X=tf._x_given_z(xs)
#imgs=X.transpose(0,2,3,1)
#color_grid_vis(imgs,show=True,save=False,transform=False)


from Loader import Loader,Video

vid=Video(data_dir+'first_red/2/')

####LOAD FIRST DATA#####
#mydata=Loader(data_dir)
#mydata.load_data('first_red')
#mydata.load_data('first_purple')
#import dill
#with open(data_dir+'first_red first_purple.pkl','wb') as handle:
#    dill.dump(mydata,handle)
import dill
with open(data_dir+'first_red first_purple.pkl','rb') as handle:
    mydata=dill.load(handle)




import matplotlib.animation as animation
#from pylab import *


dpi = 100

def crop_frame(X):
    return X[3:35,15:47]

#def ani_frame(video):

#video=vid
video=mydata.Datasets['first_red'][24]
    
    
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
im = ax.imshow(crop_frame(video[0]),cmap='gray',interpolation='nearest')
im.set_clim([0,1])
fig.set_size_inches([5,5])
plt.tight_layout()


def update_img(n):
    tmp = crop_frame(video[n])
    im.set_data(tmp)
    return im
#legend(loc=0)
ani = animation.FuncAnimation(fig,update_img,len(video),interval=100,repeat_delay=500)
#ani = animation.FuncAnimation(fig,update_img,len(video),repeat_delay=500)
writer = animation.writers['ffmpeg'](fps=10)
#ani.save(results_dir+'demo_movie.mp4',writer=writer,dpi=dpi)


plt.show()
#return ani




#for i,v in enumerate(mydata.Datasets['first_red']):
#    print i,len(v)






