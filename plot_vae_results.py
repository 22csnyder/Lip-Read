# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 20:28:58 2015

@author: csnyder
"""

data_dir='/work/03176/csnyder/EE371/Data/'
results_dir='/work/03176/csnyder/EE371/Results/'


#trX=np.load(data_dir+'generic_speaking_data_5_7_8.npy')
#trX=trX[:22000,:,3:35,15:47]
#trX=trX.astype(np.float)#maybe should have cast as floatX next time
#trX=trX/trX.max()

from conv_deconv_vae import ConvVAE

tf = ConvVAE(image_save_root=results_dir,
             snapshot_file=results_dir+'/generic_speaking_5_7_8_1-22000_no-white_square_resized_snapshot.pkl')


costs=tf.costs_

import matplotlib.pyplot as plt

plt.plot(costs[:1000])
plt.yscale('log')
plt.title('Training Cost')
plt.xlabel('MiniBatch Iteration')
plt.ylabel('Cost')
plt.show()





