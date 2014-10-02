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

import numpy.linalg

from boto.sqs.message import RawMessage
import boto.sqs

from random import random
from ConfigParser import SafeConfigParser

parser = SafeConfigParser()
parser.read('config.ini')
messages=parser.getboolean('config', 'messages')
cuda=parser.getboolean('config', 'cuda')
queue = None

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

def evaluate_lenet5(initial_learning_rate, learning_decay, learning_rate_min, lambada, nkerns,
	epoch_data = None, best_data = None,
	n_epochs=250, batch_size=100):

    rng = numpy.random.RandomState(23455)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    W_0 = None; b_0 = None;
    W_1 = None; b_1 = None;
    W_1_1 = None; b_1_1 = None;
    W_2 = None; b_2 = None;
    W_3 = None; b_3 = None;
    epoch = 0
    learning_rate = theano.shared(numpy.float32(initial_learning_rate))
    net_run_time = 0
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    
    m=""
    if epoch_data is not None:
	    W_0 = epoch_data[0][8]; b_0 = epoch_data[0][9];
	    W_1 = epoch_data[0][6]; b_1 = epoch_data[0][7];
	    W_1_1 = epoch_data[0][4]; b_1_1 = epoch_data[0][5];
	    W_2 = epoch_data[0][2]; b_2 = epoch_data[0][3];
	    W_3 = epoch_data[0][0]; b_3 = epoch_data[0][1];
	    epoch = epoch_data[1]
	    learning_rate = epoch_data[2]
	    net_run_time = epoch_data[3]
	    print('Loaded epoch %i model obtained after run time of %.2fm'\
		  %(epoch, (net_run_time) / 60.))
	    m+='Loaded epoch %i model obtained after run time of %.2fm\n'\
		  %(epoch, (net_run_time) / 60.)
		  
    if best_data is not None:
	    best_params = best_data[0]
	    best_validation_loss = best_data[1]
	    best_iter = best_data[2]
	    test_score = best_data[3]
	    print('Loaded previous epoch best validation score of %f%%\n\tobtained at iteration %i, '\
		  'with test performance %f %%' % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	    m+='Loaded previous epoch best validation score of %f%%\n\tobtained at iteration %i, '\
		  'with test performance %f %%' % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    if messages:
	    if m!="":
		raw_message = RawMessage()
		raw_message.set_body(m)
		queue.write(raw_message)
	
    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, 3, 3), poolsize=(2, 2), W=W_0, b=b_0)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 15, 15),
            filter_shape=(nkerns[1], nkerns[0], 2, 2), poolsize=(2, 2), W=W_1, b=b_1)
    layer1_1 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 7, 7),
            filter_shape=(nkerns[2], nkerns[1], 2, 2), poolsize=(2, 2), W=W_1_1, b=b_1_1)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1_1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[2] * 3 * 3,
                         n_out=500, activation=T.tanh, W=W_2, b=b_2)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10, W=W_3, b=b_3)

    params = layer3.params + layer2.params + layer1_1.params +layer1.params + layer0.params
    L2 = 	(layer0.W**2).sum() + (layer1.W**2).sum() \
		+(layer2.W**2).sum()+(layer3.W**2).sum()
		
    t_learning_rate_min = theano.shared(numpy.float32(learning_rate_min))
    t_learning_decay=T.fscalar()
    update_learning_rate = theano.function([t_learning_decay], updates=[(learning_rate, T.maximum(t_learning_rate_min, learning_rate*t_learning_decay))])

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y) + lambada*L2

    train_errors = layer3.errors(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], (cost, train_errors), updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    start_time = time.clock()

    done_looping = False

#    while (epoch < n_epochs) and (not done_looping):
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            (cost_ij, train_losses) = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

		m=""
                # compute zero-one loss on train set
                print('epoch %i, minibatch %i/%i\ntrain error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       train_losses * 100.))
		m+='%i, %.4f' %(epoch, train_losses)

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('validation error %f %%' % \
                      (this_validation_loss * 100.))
		m+=', %.4f' % (this_validation_loss)
		       
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
		    best_params = params

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('test error of best model %f %%') % 
                          (test_score * 100.))
		    m+=(', %.4f') % (test_score)
		    f=open("best.pickle", "wb")
		    cPickle.dump((best_params, best_validation_loss, best_iter, test_score), f)
		    f.close()
		if messages:
			raw_message = RawMessage()
			raw_message.set_body(m)
			queue.write(raw_message)

            if  (train_losses<0.01): #(patience <= iter)
                done_looping = True
                break
	
	update_learning_rate(numpy.float32(learning_decay))
	#post-epoch save state
	#params, epoch, net_run_time, (best_params, best_validation_loss, best_iter, test_score)
	f=open("epoch.pickle", "wb")
	run_time=time.clock()-start_time
	cPickle.dump((params, epoch, learning_rate, net_run_time+run_time), f)
	f.close()

    end_time = time.clock()
    m='Optimization complete.\n'
    m+='Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%\n' %(best_validation_loss * 100., best_iter + 1, test_score * 100.)
    m+=('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((net_run_time+end_time - start_time) / 60.))
    print(m)
    if messages:
	    raw_message = RawMessage()
	    raw_message.set_body(m)
	    queue.write(raw_message)    

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
	if W is None:
		self.W = theano.shared(value=numpy.zeros((n_in, n_out),
							 dtype=theano.config.floatX),
					name='W', borrow=True)
		# initialize the baises b as a vector of n_out 0s
	else:
		self.W = W
	
	if b is None:
		self.b = theano.shared(value=numpy.zeros((n_out,),
							 dtype=theano.config.floatX),
				       name='b', borrow=True)
	else:
		self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data():
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''

	#############
	# LOAD DATA #
	#############
	print 'loading training data'

	# Load the datasets

	validation_divide=parser.getint('config', 'validation_divide')
	f=open(parser.get('config', 'train_labels'))
	l=cPickle.load(f)
	f.close()
	d=serial.load(parser.get('config', 'train_file_1'))
	dnet=d[:validation_divide][:]
	lnet=l[:validation_divide]

	valid_set = (d[validation_divide:][:], numpy.asarray(l[validation_divide:]))

	train_file_count=parser.getint('config', 'train_file_count')
	for i in range(1,train_file_count):
		d=serial.load(parser.get('config', 'train_file_'+str(i)))
		dnet=numpy.vstack((dnet,d[:validation_divide][:]))
		lnet=lnet+l[:validation_divide]
	
	train_set = (dnet, numpy.asarray(lnet))

	print 'loaded training and validation data'

	print 'loading test data...'
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
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
	    (test_set_x, test_set_y)]
	print 'all data now loaded. let the games begin.'
	return rval

if __name__ == '__main__':

	if messages:
		conn = boto.sqs.connect_to_region(
			"us-east-1")
		queue = conn.get_queue('LearningMessages')
		print 'connected to messaging queue'
	
	initial_learning_rate=parser.getfloat('config', 'initial_learning_rate')
	learning_decay=parser.getfloat('config', 'learning_decay')
	learning_rate_min=parser.getfloat('config', 'learning_rate_min')
	lambada=parser.getfloat('config', 'lambada')
	nkerns_shape=parser.getint('config', 'nkerns_shape')
	nkerns=[]
	for i in range(0,nkerns_shape):
		nkerns+=[parser.getint('config', 'nkerns_'+str(i))]
	n_epochs=parser.getint('config', 'n_epochs')
	batch_size=parser.getint('config', 'batch_size')
	
	m="Greetings. Hydra is now online.\n"
	m+="Config:\n"
	m+="learning: %.4f/%.4f/%.4f\n"%(initial_learning_rate, learning_decay, learning_rate_min)
	m+="lambada: %4f\n"%lambada
	m+="kernels: "
	for k in range(0,len(nkerns)):
		m+=str(nkerns[k])+" "
	m+="\nbatch size: %i\n"%batch_size
	m+="epochs: %i\n"%n_epochs

	epoch_data = None
	best_data = None
	
	if parser.getboolean('config', 'resume'):
		if os.path.isfile("epoch.pickle"):
			f=open("epoch.pickle")
			epoch_data = cPickle.load(f)
			f.close()
			print 'loaded last model configuration from previous epoch'
			m+='\nloaded last model configuration from previous epoch'
		if os.path.isfile("best.pickle"):
			f=open("best.pickle")
			best_data = cPickle.load(f)
			f.close()
			print 'loaded best model configuration for previous epoch'
			m+='\nloaded best model configuration for previous epoch'

	if messages:
		raw_message = RawMessage()
		raw_message.set_body(m)
		queue.write(raw_message)
		
		
	evaluate_lenet5(initial_learning_rate, learning_decay, learning_rate_min, lambada, nkerns,
		epoch_data, best_data,
		n_epochs, batch_size)
