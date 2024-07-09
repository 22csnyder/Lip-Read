# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:39:10 2015

@author: csnyder
"""

#def _model(self, X, e):
import numpy as np
import __builtin__


def rectify(x):
    return (x + abs(x)) / 2.0    

def conv(X, w, b, activation):
    # z = dnn_conv(X, w, border_mode=int(np.floor(w.get_value().shape[-1]/2.)))
    s = int(np.floor(w.get_value().shape[-1]/2.))
    z = conv2d(X, w, border_mode='full')[:, :, s:-s, s:-s]
    if b is not None:
        z += b.dimshuffle('x', 0, 'x', 'x')
    return activation(z)

import skimage.measure as m
def max_pool_2d(input,ds):    ###I wrote this one myself from scratch#(warning)    
    return m.block_reduce(input,ds,np.max)#ds needs be tuple

def conv_and_pool(X, w, b=None, activation=rectify):
    return max_pool_2d(conv(X, w, b, activation=activation), (2, 2))

class CpuVAE:              
    def _conv_gaussian_enc(self, X, w, w2, w3, b3, wmu, bmu, wsigma, bsigma):
        h = conv_and_pool(X, w)
        h2 = conv_and_pool(h, w2)
        h2 = h2.reshape((h2.shape[0], -1))
        h3 = np.tanh(np.dot(h2, w3) + b3)
        mu = np.dot(h3, wmu) + bmu
        log_sigma = 0.5 * (np.dot(h3, wsigma) + bsigma)
        return mu, log_sigma
        
    def _z_given_x(self,X,e):
        code_mu, code_log_sigma = self._conv_gaussian_enc(X, *self.enc_params)
        Z = code_mu + np.exp(code_log_sigma) * e
        return Z



        
#    from theano.tensor.signal.downsample import max_pool_2d
    
    



#def max_pool_2d(input, ds, ignore_border=False, st=None, padding=(0, 0)):
#    """
#    Takes as input a N-D tensor, where N >= 2. It downscales the input image by
#    the specified factor, by keeping only the maximum value of non-overlapping
#    patches of size (ds[0],ds[1])
#
#    :type input: N-D theano tensor of input images.
#    :param input: input images. Max pooling will be done over the 2 last
#        dimensions.
#    :type ds: tuple of length 2
#    :param ds: factor by which to downscale (vertical ds, horizontal ds).
#        (2,2) will halve the image in each dimension.
#    :type ignore_border: bool
#    :param ignore_border: When True, (5,5) input with ds=(2,2)
#        will generate a (2,2) output. (3,3) otherwise.
#    :type st: tuple of lenght 2
#    :param st: stride size, which is the number of shifts
#        over rows/cols to get the the next pool region.
#        if st is None, it is considered equal to ds
#        (no overlap on pooling regions)
#    :param padding: (pad_h, pad_w), pad zeros to extend beyond four borders
#            of the images, pad_h is the size of the top and bottom margins,
#            and pad_w is the size of the left and right margins.
#    :type padding: tuple of two ints
#
#    """
#    if input.ndim < 2:
#        raise NotImplementedError('max_pool_2d requires a dimension >= 2')
#    if input.ndim == 4:
#        op = DownsampleFactorMax(ds, ignore_border, st=st, padding=padding)
#        output = op(input)
#        return output
#
#    # extract image dimensions
#    img_shape = input.shape[-2:]
#
#    # count the number of "leading" dimensions, store as dmatrix
##    batch_size = tensor.prod(input.shape[:-2])
#    batch_size = np.prod(input.shape[:-2])
##    batch_size = tensor.shape_padright(batch_size, 1)
#    batch_size=batch_size[:,np.newaxis]
#
#    # store as 4D tensor with shape: (batch_size,1,height,width)
##    new_shape = tensor.cast(tensor.join(0, batch_size,tensor.as_tensor([1]),img_shape), 'int64')
#    new_shape=np.array( list(batch_size) + [1] + list(img_shape) ).astype(np.int64)
#    
#    
##    input_4D = tensor.reshape(input, new_shape, ndim=4)
#    input_4D = np.reshape(input, new_shape)
#
#    # downsample mini-batch of images
#    op = DownsampleFactorMax(ds, ignore_border, st=st, padding=padding)
#    output = op(input_4D)
#
#    # restore to original shape
#    outshp=[0];outshp.extend(input.shape[:-2]);outshp.extend(output.shape[-2:])
##    outshp = tensor.join(0, input.shape[:-2], output.shape[-2:])
#    return np.reshape(output, outshp)
##    return tensor.reshape(output, outshp, ndim=input.ndim)
#
#
#
#class DownsampleFactorMax(Op):
#    """For N-dimensional tensors, consider that the last two
#    dimensions span images.  This Op downsamples these images by a
#    factor ds, by taking the max over non- overlapping rectangular
#    regions.
#
#    """
#    __props__ = ('ds', 'ignore_border', 'st', 'padding')
#
#    @staticmethod
#    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
#        """Return the shape of the output from this op, for input of given
#        shape and flags.
#
#        :param imgshape: the shape of a tensor of images. The last two elements
#            are interpreted as the number of rows, and the number of cols.
#        :type imgshape: tuple, list, or similar of integer or
#            scalar Theano variable.
#
#        :param ds: downsample factor over rows and columns
#                   this parameter indicates the size of the pooling region
#        :type ds: list or tuple of two ints
#
#        :param st: the stride size. This is the distance between the pooling
#                   regions. If it's set to None, in which case it equlas ds.
#        :type st: list or tuple of two ints
#
#        :param ignore_border: if ds doesn't divide imgshape, do we include an
#            extra row/col of partial downsampling (False) or ignore it (True).
#        :type ignore_border: bool
#
#        :param padding: (pad_h, pad_w), pad zeros to extend beyond four borders
#            of the images, pad_h is the size of the top and bottom margins,
#            and pad_w is the size of the left and right margins.
#        :type padding: tuple of two ints
#
#        :rtype: list
#        :returns: the shape of the output from this op, for input of given
#            shape.  This will have the same length as imgshape, but with last
#            two elements reduced as per the downsampling & ignore_border flags.
#        """
#        if len(imgshape) < 2:
#            raise TypeError('imgshape must have at least two elements '
#                            '(rows, cols)')
#
#        if st is None:
#            st = ds
#        r, c = imgshape[-2:]
#        r += padding[0] * 2
#        c += padding[1] * 2
#
#        if ignore_border:
#            out_r = (r - ds[0]) // st[0] + 1
#            out_c = (c - ds[1]) // st[1] + 1
#            if isinstance(r, theano.Variable):
#                nr = tensor.maximum(out_r, 0)
#            else:
#                nr = numpy.maximum(out_r, 0)
#            if isinstance(c, theano.Variable):
#                nc = tensor.maximum(out_c, 0)
#            else:
#                nc = numpy.maximum(out_c, 0)
#        else:
#            if isinstance(r, theano.Variable):
#                nr = tensor.switch(tensor.ge(st[0], ds[0]),
#                                   (r - 1) // st[0] + 1,
#                                   tensor.maximum(0, (r - 1 - ds[0])
#                                                  // st[0] + 1) + 1)
#            elif st[0] >= ds[0]:
#                nr = (r - 1) // st[0] + 1
#            else:
#                nr = max(0, (r - 1 - ds[0]) // st[0] + 1) + 1
#
#            if isinstance(c, theano.Variable):
#                nc = tensor.switch(tensor.ge(st[1], ds[1]),
#                                   (c - 1) // st[1] + 1,
#                                   tensor.maximum(0, (c - 1 - ds[1])
#                                                  // st[1] + 1) + 1)
#            elif st[1] >= ds[1]:
#                nc = (c - 1) // st[1] + 1
#            else:
#                nc = max(0, (c - 1 - ds[1]) // st[1] + 1) + 1
#
#        rval = list(imgshape[:-2]) + [nr, nc]
#        return rval
#
#    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0)):
#        """
#        :param ds: downsample factor over rows and column.
#                   ds indicates the pool region size.
#        :type ds: list or tuple of two ints
#
#        :param ignore_border: if ds doesn't divide imgshape, do we include
#            an extra row/col of partial downsampling (False) or
#            ignore it (True).
#        :type ignore_border: bool
#
#        : param st: stride size, which is the number of shifts
#            over rows/cols to get the the next pool region.
#            if st is None, it is considered equal to ds
#            (no overlap on pooling regions)
#        : type st: list or tuple of two ints
#
#        :param padding: (pad_h, pad_w), pad zeros to extend beyond four borders
#            of the images, pad_h is the size of the top and bottom margins,
#            and pad_w is the size of the left and right margins.
#        :type padding: tuple of two ints
#
#        """
#        self.ds = tuple(ds)
#        if not all([isinstance(d, int) for d in ds]):
#            raise ValueError(
#                "DownsampleFactorMax downsample parameters must be ints."
#                " Got %s" % str(ds))
#        if st is None:
#            st = ds
#        self.st = tuple(st)
#        self.ignore_border = ignore_border
#        self.padding = tuple(padding)
#        if self.padding != (0, 0) and not ignore_border:
#            raise NotImplementedError(
#                'padding works only with ignore_border=True')
#        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
#            raise NotImplementedError(
#                'padding_h and padding_w must be smaller than strides')
#
#    def __str__(self):
#        return '%s{%s, %s, %s, %s}' % (
#            self.__class__.__name__,
#            self.ds, self.st, self.ignore_border, self.padding)
#
#    def make_node(self, x):
#        if x.type.ndim != 4:
#            raise TypeError()
#        # TODO: consider restricting the dtype?
#        x = tensor.as_tensor_variable(x)
#        return gof.Apply(self, [x], [x.type()])
#
#    def perform(self, node, inp, out):
#        x, = inp
#        z, = out
#        if len(x.shape) != 4:
#            raise NotImplementedError(
#                'DownsampleFactorMax requires 4D input for now')
#        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,self.padding)
#        if (z[0] is None) or (z[0].shape != z_shape):
#            z[0] = np.empty(z_shape, dtype=x.dtype)
#        zz = z[0]
#        # number of pooling output rows
#        pr = zz.shape[-2]
#        # number of pooling output cols
#        pc = zz.shape[-1]
#        ds0, ds1 = self.ds
#        st0, st1 = self.st
#        pad_h = self.padding[0]
#        pad_w = self.padding[1]
#        img_rows = x.shape[-2] + 2 * pad_h
#        img_cols = x.shape[-1] + 2 * pad_w
#
#        # pad the image
#        if self.padding != (0, 0):
#            fill = x.min()-1.
#            y = numpy.zeros(
#                (x.shape[0], x.shape[1], img_rows, img_cols),
#                dtype=x.dtype) + fill
#            y[:, :, pad_h:(img_rows-pad_h), pad_w:(img_cols-pad_w)] = x
#        else:
#            y = x
#        # max pooling
#        for n in xrange(x.shape[0]):
#            for k in xrange(x.shape[1]):
#                for r in xrange(pr):
#                    row_st = r * st0
#                    row_end = __builtin__.min(row_st + ds0, img_rows)
#                    for c in xrange(pc):
#                        col_st = c * st1
#                        col_end = __builtin__.min(col_st + ds1, img_cols)
#                        zz[n, k, r, c] = y[
#                            n, k, row_st:row_end, col_st:col_end].max()
#
#    def infer_shape(self, node, in_shapes):
#        shp = self.out_shape(in_shapes[0], self.ds,
#                             self.ignore_border, self.st, self.padding)
#        return [shp]
#
#    def grad(self, inp, grads):
#        x, = inp
#        gz, = grads
#        maxout = self(x)
#        return [DownsampleFactorMaxGrad(self.ds,
#                                        ignore_border=self.ignore_border,
#                                        st=self.st, padding=self.padding)(
#                                            x, maxout, gz)]
#
#    def c_code(self, node, name, inp, out, sub):
#        # No implementation is currently for the case where
#        # the stride size and the pooling size are different.
#        # An exception is raised for such a case.
#        if self.ds != self.st or self.padding != (0, 0):
#            raise theano.gof.utils.MethodNotDefined()
#        x, = inp
#        z, = out
#        fail = sub['fail']
#        ignore_border = int(self.ignore_border)
#        ds0, ds1 = self.ds
#        return """
#        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
#        int x_shp0_usable;
#        int x_shp1_usable;
#        int z_shp0, z_shp1;
#        if(PyArray_NDIM(%(x)s)!=4)
#        {
#            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
#            %(fail)s;
#        }
#        z_shp0 = PyArray_DIMS(%(x)s)[2] / %(ds0)s;
#        z_shp1 = PyArray_DIMS(%(x)s)[3] / %(ds1)s;
#        if (%(ignore_border)s)
#        {
#            x_shp0_usable = z_shp0 * %(ds0)s;
#            x_shp1_usable = z_shp1 * %(ds1)s;
#        }
#        else
#        {
#            z_shp0 += (PyArray_DIMS(%(x)s)[2] %% %(ds0)s) ? 1 : 0;
#            z_shp1 += (PyArray_DIMS(%(x)s)[3] %% %(ds1)s) ? 1 : 0;
#            x_shp0_usable = PyArray_DIMS(%(x)s)[2];
#            x_shp1_usable = PyArray_DIMS(%(x)s)[3];
#        }
#        if ((!%(z)s)
#          || *PyArray_DIMS(%(z)s)!=4
#          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
#          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
#          ||(PyArray_DIMS(%(z)s)[2] != z_shp0)
#          ||(PyArray_DIMS(%(z)s)[3] != z_shp1)
#          )
#        {
#          if (%(z)s) Py_XDECREF(%(z)s);
#          npy_intp dims[4] = {0,0,0,0};
#          dims[0]=PyArray_DIMS(%(x)s)[0];
#          dims[1]=PyArray_DIMS(%(x)s)[1];
#          dims[2]=z_shp0;
#          dims[3]=z_shp1;
#          //TODO: zeros not necessary
#          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
#        }
#
#        if (z_shp0 && z_shp1)
#        {
#            for(int b=0;b<PyArray_DIMS(%(x)s)[0];b++){
#              for(int k=0;k<PyArray_DIMS(%(x)s)[1];k++){
#                int mini_i = 0;
#                int zi = 0;
#                for(int i=0;i< x_shp0_usable; i++){
#                  int mini_j = 0;
#                  int zj = 0;
#                  for(int j=0; j<x_shp1_usable; j++){
#                    dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,i,j)))[0];
#                    dtype_%(z)s * __restrict__ z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,zi,zj)));
#                    z[0] = (((mini_j|mini_i) == 0) || z[0] < a) ? a : z[0];
#                    mini_j = ((mini_j + 1) == %(ds1)s) ? 0 : mini_j+1;
#                    zj += (mini_j == 0);
#                  }
#                  mini_i = ((mini_i + 1) == %(ds0)s) ? 0 : mini_i+1;
#                  zi += (mini_i == 0);
#                }
#              }
#            }
#        }
#        """ % locals()
#
#    def c_code_cache_version(self):
#        return (0, 2)
#
#
#
#
#def conv2d(input, filters, image_shape=None, filter_shape=None,
#           border_mode='valid', subsample=(1, 1), **kargs):
#    """This function will build the symbolic graph for convolving a stack of
#    input images with a set of filters. The implementation is modelled after
#    Convolutional Neural Networks (CNN). It is simply a wrapper to the ConvOp
#    but provides a much cleaner interface.
#
#    :type input: symbolic 4D tensor
#    :param input: mini-batch of feature map stacks, of shape
#                  (batch size, stack size, nb row, nb col)
#                  see the optional parameter image_shape
#
#    :type filters: symbolic 4D tensor
#    :param filters: set of filters used in CNN layer of shape
#                    (nb filters, stack size, nb row, nb col)
#                    see the optional parameter filter_shape
#
#    :param border_mode:
#       'valid'-- only apply filter to complete patches of the image. Generates
#                 output of shape: image_shape - filter_shape + 1
#       'full' -- zero-pads image to multiple of filter shape to generate output
#                 of shape: image_shape + filter_shape - 1
#
#    :type subsample: tuple of len 2
#    :param subsample: factor by which to subsample the output.
#                      Also called strides elsewhere.
#
#    :type image_shape: None, tuple/list of len 4 of int, None or
#                       Constant variable
#    :param image_shape: The shape of the input parameter.
#                        Optional, used for optimization like loop unrolling
#                        You can put None for any element of the list
#                        to tell that this element is not constant.
#    :type filter_shape: None, tuple/list of len 4 of int, None or
#                        Constant variable
#    :param filter_shape: Optional, used for optimization like loop unrolling
#                         You can put None for any element of the list
#                         to tell that this element is not constant.
#    :param kwargs: kwargs are passed onto ConvOp.
#                   Can be used to set the following:
#                   unroll_batch, unroll_kern, unroll_patch,
#                   openmp (see ConvOp doc)
#
#                   openmp: By default have the same value as
#                           config.openmp. For small image, filter,
#                           batch size, nkern and stack size, it can be
#                           faster to disable manually openmp. A fast and
#                           incomplete test show that with image size
#                           6x6, filter size 4x4, batch size==1,
#                           n kern==1 and stack size==1, it is faster
#                           to disable it in valid mode. But if we
#                           grow the batch size to 10, it is faster
#                           with openmp on a core 2 duo.
#
#    :rtype: symbolic 4D tensor
#    :return: set of feature maps generated by convolutional layer. Tensor is
#        of shape (batch size, nb filters, output row, output col)
#
#    """
#
#    #accept Constant value for image_shape and filter_shape.
#    if image_shape is not None:
#        image_shape = list(image_shape)
#        for i in xrange(len(image_shape)):
#            if image_shape[i] is not None:
#                try:
#                    image_shape[i] = get_scalar_constant_value(
#                        as_tensor_variable(image_shape[i]))
#                except NotScalarConstantError, e:
#                    raise NotScalarConstantError(
#                        "The convolution need that the shape"
#                        " information are constant values. We got"
#                        " %s for the image_shape parameter" %
#                        image_shape[i])
#                assert str(image_shape[i].dtype).startswith('int')
#                image_shape[i] = int(image_shape[i])
#    if filter_shape is not None:
#        filter_shape = list(filter_shape)
#        for i in xrange(len(filter_shape)):
#            if filter_shape[i] is not None:
#                try:
#                    filter_shape[i] = get_scalar_constant_value(
#                        as_tensor_variable(filter_shape[i]))
#                except NotScalarConstantError, e:
#                    raise NotScalarConstantError(
#                        "The convolution need that the shape"
#                        " information are constant values. We got"
#                        " %s for the filter_shape "
#                        "parameter" % filter_shape[i])
#                assert str(filter_shape[i].dtype).startswith('int')
#                filter_shape[i] = int(filter_shape[i])
#
#    if image_shape and filter_shape:
#        try:
#            assert image_shape[1] == filter_shape[1]
#        except Exception:
#            print 'image ', image_shape, ' filters ', filter_shape
#            raise
#
#    if filter_shape is not None:
#        nkern = filter_shape[0]
#        kshp = filter_shape[2:]
#    else:
#        nkern, kshp = None, None
#
#    if image_shape is not None:
#        bsize = image_shape[0]
#        imshp = image_shape[1:]
#    else:
#        bsize, imshp = None, None
#
#    op = ConvOp(output_mode=border_mode, dx=subsample[0], dy=subsample[1],
#                imshp=imshp, kshp=kshp, nkern=nkern, bsize=bsize, **kargs)
#
#    return op(input, filters)
#    
    

