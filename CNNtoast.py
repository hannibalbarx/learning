"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import os.path
import sys
import time
import glob
from collections import deque

import numpy

import theano
import theano.tensor as T

from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.utils import serial

from mlp import HiddenLayer
from logistic_sgd_test import LogisticRegression

import numpy.linalg

from random import random
from ConfigParser import SafeConfigParser

parser = SafeConfigParser()
parser.read('config.ini')
if parser.getboolean('config', 'exit'):
	print 'why you no love me? bye.'
	exit()
cuda=parser.getboolean('config', 'cuda')

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W=None, b=None, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if W is None:
		self.W = theano.shared(numpy.asarray(
			rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
			dtype=theano.config.floatX),
                               borrow=True)
	else:
		self.W = W

        if b is None:
		# the bias is a 1D tensor -- one bias per output feature map
		b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)
	else:
		self.b = b

        # convolve input feature maps with filters
	if not cuda:
		conv_out = conv.conv2d(input=input, filters=self.W,
			filter_shape=filter_shape, image_shape=image_shape)
	else:
		input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		conv_op = FilterActs(stride=1, partial_sum=1)
		contiguous_input = gpu_contiguous(input_shuffled)
		contiguous_filters = gpu_contiguous(filters_shuffled)
		conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)
	
        # downsample each feature map individually, using maxpooling
	if not cuda:
		pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
	else:
		pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
		pooled_out_shuffled = pool_op(conv_out_shuffled)
		pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
	
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
#        self.output = T.maximum(0,pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class LeNetConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if W is None:
		self.W = theano.shared(numpy.asarray(
			rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
			dtype=theano.config.floatX),
                               borrow=True)
	else:
		self.W = W

        # convolve input feature maps with filters
	if not cuda:
		self.output = conv.conv2d(input=input, filters=self.W,
			filter_shape=filter_shape, image_shape=image_shape)
	else:
		input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
		conv_op = FilterActs(stride=1, partial_sum=1)
		contiguous_input = gpu_contiguous(input_shuffled)
		contiguous_filters = gpu_contiguous(filters_shuffled)
		self.output = conv_op(contiguous_input, contiguous_filters).dimshuffle(3, 0, 1, 2)
	
        # store parameters of this layer
        self.params = [self.W]

def evaluate_lenet5(lambada, nkerns, hnn,
	epoch_data = None):

    rng = numpy.random.RandomState(23455)
    datasets = load_data()

    test_set_x, test_set_y = datasets[0]
    batch_size=len(test_set_x.get_value())

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    W_0 = None ; b_0 = None;
    W_1 = None; b_1 = None;
    W_1_1 = None; b_1_1 = None;
    W_2 = None; b_2 = None;
    W_3 = None; b_3 = None;
    
    if epoch_data is not None:
	    W_0 = epoch_data[0][8] ; b_0 = epoch_data[0][9];
	    W_1 = epoch_data[0][6]; b_1 = epoch_data[0][7];
	    W_1_1 = epoch_data[0][4]; b_1_1 = epoch_data[0][5];
	    W_2 = epoch_data[0][2]; b_2 = epoch_data[0][3];
	    W_3 = epoch_data[0][0]; b_3 = epoch_data[0][1];
		  
    layer0_input = x.reshape((batch_size, 3, 32, 32))[:,:,1:31,1:31]

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 3, 30, 30),
            filter_shape=(nkerns[0], 3, 3, 3), poolsize=(2, 2), W=W_0, b=b_0)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 14, 14),
            filter_shape=(nkerns[1], nkerns[0], 3, 3), poolsize=(2, 2), W=W_1, b=b_1)
    layer1_1 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 6, 6),
            filter_shape=(nkerns[2], nkerns[1], 3, 3), poolsize=(2, 2), W=W_1_1, b=b_1_1)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1_1.output.flatten(2)


    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[2] * 2 * 2,
                         n_out=hnn, activation=T.tanh, W=W_2, b=b_2)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=hnn, n_out=10, W=W_3, b=b_3)

    L2 = 	(layer0.W**2).sum() + (layer1.W**2).sum() + (layer1_1.W**2).sum()\
		+(layer2.W**2).sum()+(layer3.W**2).sum()
		
    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y) + lambada*L2

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([], layer3.errors(y),
             givens={
                x: test_set_x,
                y: test_set_y})
    
    test_losses = test_model()
    test_out_file = open("test_out_file.txt","wb")
    for j in range(batch_size):
	test_out_file.write("%s"%test_losses[1][j])
	for i in range(10):
		test_out_file.write(",%s"%test_losses[2][j][i])
	test_out_file.write("\n")
    test_out_file.close()

def load_data():
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''

	#############
	# LOAD DATA #
	#############

	# Load the datasets

	print '... loading data'
	
	f=open(parser.get('config', 'test_labels'))
	l = cPickle.load(f)
	f.close();
	d=serial.load(parser.get('config', 'test_file'))
	test_set = (d, numpy.asarray(l))

	def shared_dataset(data_xy, borrow=True):
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
						       dtype=theano.config.floatX),
					 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
						       dtype=theano.config.floatX),
					 borrow=borrow)
		return shared_x, T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	#valid_set_x, valid_set_y = shared_dataset(valid_set)
	#train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(test_set_x, test_set_y)]

	return rval

if __name__ == '__main__':

	lambada=parser.getfloat('config', 'lambada')
	nkerns_shape=parser.getint('config', 'nkerns_shape')
	nkerns=[]
	for i in range(0,nkerns_shape):
		nkerns+=[parser.getint('config', 'nkerns_'+str(i))]
        hnn = parser.getint('config', 'hnn_1')
	
	epoch_data = None
	best_data = None

	if os.path.isfile("best.pickle"):
		f=open("best.pickle")
		epoch_data = cPickle.load(f)
		f.close()
	else:
		print 'not best config found'
		exit()

	evaluate_lenet5(lambada, nkerns, hnn,
		epoch_data)
