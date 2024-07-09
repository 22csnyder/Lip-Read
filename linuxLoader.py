# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:41:47 2015

@author: christopher
"""

import re
import os
from skimage import io

def ls_full_path(folder):
    files=os.listdir(folder)
    path=os.path.abspath(folder)+'/'
    return [path+f for f in files]
    

def file2index(f):
    return int(re.findall(r'\d+',f)[0])


class Video:
    def __init__(self,folder):
        self.files=ls_full_path(folder)
#        self.files=os.listdir(folder)
        self.files.sort(key=file2index)
        self.frames=[io.imread(f) for f in self.files]
    def __len__(self):
        return len(self.files)
    def __getitem__(self,i):#not especially well implemented
        return self.frames[i]
        


class Loader:
    def __init__(self,data_directory):
        if data_directory[-1] is not '/':
            data_directory+='/'
        self.data_directory=data_directory
        
        self.Datasets=dict()
    def load_data(self,folder_name):
        samples=ls_full_path(self.data_directory+folder_name)
        samples.sort(key=file2index)
        self.Datasets[folder_name]=[Video(s) for s in samples]
        
        
default_data_dir='/home/christopher/Documents/Classes/EE371R/ClassProject/Data/'






#vid=Video(default_data_dir+'first_red/2/')

datasets=['generic_speaking5','generic_speaking7','generic_speaking8']

####LOAD FIRST DATA#####
mydata=Loader(default_data_dir)
#mydata.load_data('first_red')
mydata.load_data('generic_speaking5')



mydata.load_data('generic_speaking7')
mydata.load_data('generic_speaking8')

trX=[]

import numpy as np

for dataset in datasets:
    for video in mydata.Datasets[dataset]:
        for frame in video.frames:
            ft=frame.transpose(2,0,1)#put color axis first
            trX.append(ft[np.newaxis,:])
            
trX=np.vstack(trX)


    



#mydata.load_data('first_purple')
#import dill
#with open('first_data.pkl','wb') as handle:
#    dill.dump(mydata,handle)




#if __name__=='__main__':
    
    

