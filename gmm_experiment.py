# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:42:16 2015

@author: christopher
"""

import dill

with open('first_data.pkl','rb') as handle:
    mydata=dill.load(handle)
    


from sklearn import mixture


datavecs=[]

for dataset in mydata.Datasets.values():
    for video in dataset:
        for i in range(len(video)):
            frame=video[i]
            datavecs.append(frame.reshape(-1,3))
            
import numpy as np
train_data=np.vstack(datavecs)


from sklearn import mixture

g=mixture.GMM(n_components=5)
g.fit(train_data)


#with open('first_gmm.pkl','wb') as handle:
#    dill.dump(g,handle)





