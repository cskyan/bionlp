#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: cfkmeans.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-12-14 11:43:31
###########################################################################
''' Constraint Fuzzy K-means clustering '''

import os

import scipy.stats as stats

from keras import backend as K
from keras import activations
from keras.engine.topology import Layer


class CFKBase(Layer):
	def __init__(self, output_dim, input_dim=None, batch_size=32, session=None, **kwargs):
		self.output_dim = output_dim
		self.input_dim = input_dim
		self.batch_size = batch_size
		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		self.sess = session
		super(CFKBase, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		assert input_shape and len(input_shape) == 2
		return (input_shape[0], self.output_dim)
		
		
class CFKU(CFKBase):
	def __init__(self, output_dim, input_dim=None, **kwargs):
		self.activation = activations.get('softmax')
		super(CFKU, self).__init__(output_dim, input_dim=input_dim, **kwargs)

	def build(self, input_shape):
		mean, std = 0.0, 1.0
		W_init = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(self.input_dim, self.output_dim))
		self.W = K.variable(W_init)
		self.b = K.zeros((self.output_dim,))
		self.trainable_weights = [self.W, self.b]
		self.built = True

	def call(self, X, mask=None):
		batch_size = K.shape(X)[0]
		U = K.dot(X, self.W) + self.b
		output = K.reshape(U, (-1, self.output_dim))
		return self.activation(output)
		
		
class CFKD(CFKBase):
	def __init__(self, output_dim, input_dim=None, metric='euclidean', **kwargs):
		self.metric = metric
		super(CFKD, self).__init__(output_dim, input_dim=input_dim, **kwargs)
		
	def build(self, input_shape):
		mean, std = 0.0, 1.0
		M_init = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(self.output_dim, self.input_dim))
		self.M = K.variable(M_init)
		self.trainable_weights = [self.M]
		self.built = True

	def call(self, inputs, mask=None):
		X, U = inputs
		batch_size = K.cast(K.shape(X)[0], 'int32')
		C = K.dot(K.transpose(U), X) / K.cast(batch_size, 'float32')
		large_X = K.repeat_elements(K.reshape(X, (-1, 1, self.input_dim)), self.output_dim, axis=1)
		if (os.environ['KERAS_BACKEND'] == 'tensorflow'):
			batch_size = K.int_shape(X)[0]
		large_C = K.repeat_elements(K.reshape(C, (1, self.output_dim, self.input_dim)), batch_size, axis=0)
		large_M = K.repeat_elements(K.reshape(K.round(K.relu(self.M, max_value=1)), (1, self.output_dim, self.input_dim)), batch_size, axis=0)
		# L = K.sum(large_M, axis=2)
		if (self.metric == 'manhattan'):
			D = K.sum(K.abs((large_X - large_C) * large_M), axis=2)
		else:
			D = K.sum(K.square((large_X - large_C) * large_M), axis=2)
		output = K.reshape(D, (-1, self.output_dim))
		return output
		
	def get_output_shape_for(self, input_shape):
		assert input_shape and len(input_shape) == 2 and len(input_shape[0]) == 2 and len(input_shape[1]) == 2
		return (input_shape[0][0], self.output_dim)
		
		
class CFKC(CFKBase):
	def __init__(self, output_dim, input_dim=None, **kwargs):
		super(CFKC, self).__init__(output_dim, input_dim=input_dim, **kwargs)

	def call(self, inputs, mask=None):
		C, U, D = inputs
		coverlap = K.dot(C, K.transpose(C))
		loverlap = K.dot(U, K.transpose(U))
		output = K.dot((1 - K.clip(coverlap, 0, 1)) * loverlap, D)
		return output

	def get_output_shape_for(self, input_shape):
		assert input_shape and len(input_shape) == 3 and len(input_shape[0]) == 2 and len(input_shape[1]) == 2 and len(input_shape[2]) == 2
		return (input_shape[0][0], self.output_dim)