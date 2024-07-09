import cPickle
import gzip
import os

import numpy
import theano


def prepare_data(seqs, labels, maxlen=None):

    
    lengths = [len(s) for s in seqs]
    fdim=len(seqs[0][0])
    

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples,fdim)).astype(theano.config.floatX)
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx,:] = s
        x_mask[:lengths[idx], idx] = 1.
    return x, x_mask, labels


def load_data(valid_portion=0.1):
    path='/work/03176/csnyder/EE371/Results/just_colors_seq.pkl'

    with open(path,'rb') as handle:
        train_set = cPickle.load(handle)
#    test_set = cPickle.load(f)


    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)


#    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
#    test = (test_set_x, test_set_y)

#    return train, valid, test
    return train, valid, valid


if __name__=='__main__':
    #def load_isolated_words():
    data_dir='/work/03176/csnyder/EE371/Data/'
    results_dir='/work/03176/csnyder/EE371/Results/'
    vae_model=results_dir+'/generic_speaking_5_7_8_1-22000_no-white_square_resized_snapshot.pkl'

    from conv_deconv_vae import ConvVAE,floatX,imshow
    tf = ConvVAE(image_save_root=results_dir,snapshot_file=vae_model)
    maxtrX=254.0#test time images need to be rescaled by the max within the training set
    
    
    import dill
#    from Loader import Loader
#    loader_save_name=results_dir+'just_colors_loader.pkl'
#    try:
#        colors_=dill.load(loader_save_name,'rb')
#        print 'loading from file ',loader_save_name
#    except:
#        colors_=Loader(data_dir)
#        colors_.load_data('just_red')
#        colors_.load_data('just_purple')
#        colors_.load_data('just_yellow')
#        dill.dump(colors_,open(loader_save_name,'wb'))
#    sets_n_labels=list(zip(colors_.Datasets.keys(),range(len(colors_.Datasets.keys()))))



#vid=color[22]
    def process(frame_in):
        return ( floatX(frame_in[3:35,15:47]) )/maxtrX
    import numpy as np

    seq_save_name=results_dir+'just_colors_seq.pkl'

    try:
        x,y=dill.load(open(seq_save_name,'rb'))
        print 'loading from file',seq_save_name
        
    except:
        x=[]
        y=[]
    
        for dat,label in sets_n_labels:
            color=colors_.Datasets[dat]
            for vid in color:
                V= np.concatenate([ process(frame)[None,:] for frame in vid.frames],axis=0)
                X=V.transpose(0,3,1,2)
                e =  floatX(np.ones((X.shape[0], tf.n_code)))
                code_mu,code_log_sigma=tf._z_given_x(X)
                Z = floatX(code_mu + np.exp(code_log_sigma) * e)
                x.append(Z)
                y.append(label)
        
        
            dill.dump((x,y),open(seq_save_name,'wb'))



    X,M,L=prepare_data(x,y)








