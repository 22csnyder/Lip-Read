# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:54:14 2015

@author: christopher
"""
from conv_deconv_vae import ConvVAE,floatX,imshow
import cPickle

if __name__=="__main__":
    results_dir='/work/03176/csnyder/EE371/Results/'
#    results_dir='/home/christopher/Documents/Classes/EE371R/ClassProject/Results/'
    #vae_model=results_dir+'generic_speaking_5_7_8_1-22000_no-white_square_resized_snapshot.pkl'
    vae_model_file=results_dir+'vae_trained_model.pkl'#same as above but shorter file name
    new_model_file=results_dir+'cpu_safe_vae.pkl'
#    from conv_deconv_vae import ConvVAE,floatX,imshow
#    tf = ConvVAE(image_save_root=results_dir,snapshot_file=vae_model_file)


    from theano import config
    
    import dill
    with open(vae_model_file,'rb') as handle:
        classifier=dill.load(handle)
    
    
    parm=[w.get_value() for w in classifier.params]
    
    with open(new_model_file,'wb') as handle:
        dill.dump(parm,handle)
        
        
    
    
#            if os.path.exists(self.snapshot_file):
#            print "Loading from saved snapshot " , self.snapshot_file
#            f = open(self.snapshot_file, 'rb')
#            classifier = cPickle.load(f)
#            self.__setstate__(classifier.__dict__)
#            f.close()


#    import os
#    from pylearn2.utils import serial
#    import pylearn2.config.yaml_parse as yaml_parse
#    
#    
##    _, in_path, out_path = sys.argv
#    in_path=vae_model_file
#    out_path=new_model_file    
#    
#    os.environ['THEANO_FLAGS']="device=cpu"
#    
#    model = serial.load(in_path)
#    
#    model2 = yaml_parse.load(model.yaml_src)
#    model2.set_param_values(model.get_param_values())
#    
#    serial.save(out_path, model2)
