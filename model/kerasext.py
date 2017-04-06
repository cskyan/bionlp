#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: kearsext.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-12-15 11:07:43
###########################################################################
''' Keras Deep Learning Model '''

import os
import re

import numpy as np

BEMAP = {'th':'theano', 'tf':'tensorflow'}

NUM_PROCESS, DEVICE_ID, DEV, DEVICE_INIT, DEVICE_VARS = 0, 0, '', False, {}


def init(dev_id=0, num_gpu=0, backend='th', num_process=1, use_omp=False):
	global NUM_PROCESS, DEVICE_ID, DEV, DEVICE_INIT
	if (DEVICE_INIT): return
	NUM_PROCESS, DEVICE_ID = num_process, dev_id
	os.environ['KERAS_BACKEND'] = BEMAP[backend]
	if (backend == 'th'):
		DEV = 'gpu%i' % dev_id if num_gpu > 0 else 'cpu'
		os.environ['THEANO_FLAGS'] = re.sub('device=.+,', 'device=%s,' % DEV, os.environ['THEANO_FLAGS'])
		# Multiple CPUs
		if (DEV == 'cpu' and use_omp):
			os.environ['THEANO_FLAGS'] = os.environ['THEANO_FLAGS'] + ',openmp=True'
		import keras.backend.theano_backend as K
	elif (backend == 'tf'):
		DEV = '/gpu:%i' % dev_id if num_gpu > 0 else '/cpu:0'
	DEVICE_INIT = True
	
	
def get_activations(model, layer_name, X):
	import keras.backend as K
	get_activations = K.function([model.layers[0].input, K.learning_phase()], model.get_layer(layer_name).output)
	activations = get_activations([X[0],0])
	return activations
	
	
def get_dummy(**kwargs):
	from keras.engine.topology import InputSpec, Layer
	import keras.backend as K
	class Dummy(Layer):
		def __init__(self, output_dim, input_dim=None, **kwargs):
			self.output_dim = output_dim
			self.input_dim = input_dim
			if self.input_dim:
				kwargs['input_shape'] = (self.input_dim,)
			super(Dummy, self).__init__(**kwargs)
		def build(self, input_shape):
			assert len(input_shape) == 2
			input_dim = input_shape[1]
			self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, input_dim))]
			super(Dummy, self).build(input_shape)
	return Dummy(**kwargs)


def gen_mdl(input_dim, output_dim, model, mdl_type='clf', backend='th', verbose=False, udargs=[], **kwargs):
	global DEVICE_ID, DEV, DEVICE_INIT, DEVICE_VARS
	device = DEV
	if (not DEVICE_INIT):
		if (backend == 'th'):
			os.environ['THEANO_FLAGS'] = re.sub('device=.+,', 'device=%s,' % DEV, os.environ['THEANO_FLAGS'])
			# Multiple CPUs
			if (DEV == 'cpu' and NUM_PROCESS > 1):
				os.environ['THEANO_FLAGS'] = os.environ['THEANO_FLAGS'] + ',openmp=True'
			import keras.backend.theano_backend as K
		DEVICE_INIT = True
	session = None
	if (backend == 'tf'):
		import keras.backend as K
		with gen_cntxt(backend, DEV):
			config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=verbose)
			config.gpu_options.per_process_gpu_memory_fraction=0.7
			config.gpu_options.allow_growth=True
			session = DEVICE_VARS.setdefault('sess', K.tf.Session(config=config))
			K.set_session(session)
	if (udargs):
		kwargs.update({k:v for k, v in [(x, locals()[x]) for x in udargs]})
	from keras.wrappers.scikit_learn import KerasClassifier
	if (mdl_type == 'clf'):
		return KerasClassifier(build_fn=model, **kwargs)
	elif (mdl_type == 'clt'):
		import copy
		import types
		from keras.models import Sequential
		from keras.utils.np_utils import to_categorical
		class KerasCluster(KerasClassifier):
			def __init__(self, build_fn=None, batch_size=32, **kwargs):
				self.kwargs = kwargs
				self.batch_size = batch_size
				super(KerasCluster, self).__init__(build_fn=build_fn)
			def fit(self, X, y, constraint=None, **kwargs):
				if self.build_fn is None:
					self.model = self.__call__(**self.filter_sk_params(self.__call__))
				elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
					self.kwargs.update(self.filter_sk_params(self.build_fn.__call__))
					self.model = self.build_fn(batch_size=self.batch_size, **self.kwargs)
				else:
					self.kwargs.update(self.filter_sk_params(self.build_fn))
					self.model = self.build_fn(batch_size=self.batch_size, **self.kwargs)
				loss_name = self.model.loss
				if hasattr(loss_name, '__name__'):
					loss_name = loss_name.__name__
				if loss_name == 'categorical_crossentropy' and len(y.shape) != 2 and y.shape[1] < 2:
					y = to_categorical(y)
				fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
				fit_args.update(kwargs)
				history = self.model.fit([X, constraint], y, batch_size=self.batch_size, **fit_args)
				return history
			def predict(self, X, constraint=None, **kwargs):
				return super(KerasCluster, self).predict([X, constraint], **kwargs)
		return KerasCluster(build_fn=model, **kwargs)
		
		
def gen_cltmdl(proba_thrsh=0.5, context=None, session=None, **kwargs):
	'''
	Factory method for Clustering Model
	'''
	import keras.backend as K
	from keras.models import Model
	class CLTModel(Model):
		def __init__(self, proba_thrsh=0.5, constraint_dim=0, context=None, session=None, **kwargs):
			super(CLTModel, self).__init__(**kwargs)
			self.constraint_dim = constraint_dim
			self.proba_thrsh = proba_thrsh
			self.context = context
			self.session = session
		def fit(self, X, y, **kwargs):
			with gen_cntxt(**self.context):
				return super(CLTModel, self).fit(X, y, **kwargs)
		def predict_classes(self, X, **kwargs):
			with gen_cntxt(**self.context):
				proba = get_activations(self, 'U', X)
				if (kwargs.setdefault('proba', False)):
					return np.array((proba > proba.mean()).astype('int8'), dtype='int8')
				else:
					return proba
		def __del__(self):
			if (self.session is not None):
				self.session.close()
				del self.session
			super(CLTModel, self).__del__()
	return CLTModel(context=context, session=session, **kwargs)
		
		
def gen_mlmdl(proba_thrsh=0.5, context=None, session=None, **kwargs):
	'''
	Factory method for Mutilabel Model
	'''
	from keras.models import Model
	class MLModel(Model):
		def __init__(self, proba_thrsh=0.5, context=None, session=None, **kwargs):
			super(MLModel, self).__init__(**kwargs)
			self.proba_thrsh = proba_thrsh
			self.context = context
			self.session = session
		def fit(self, X, y, **kwargs):
			with gen_cntxt(**self.context):
				return super(MLModel, self).fit(X, y, **kwargs)
		def predict_classes(self, X, **kwargs):
			with gen_cntxt(**self.context):
				proba = self.predict(X, **kwargs)
			return (proba > self.proba_thrsh).astype('int8')
		def __del__(self):
			if (self.session is not None):
				self.session.close()
				del self.session
			super(MLModel, self).__del__()
	return MLModel(context=context, session=session, **kwargs)
	
	
def gen_mlseq(proba_thrsh=0.5, context=None, session=None, **kwargs):
	'''
	Factory method for Mutilabel Sequential Model
	'''
	from keras.models import Sequential
	class MLSequential(Sequential):
		def __init__(self, proba_thrsh=0.5, context=None, session=None, **kwargs):
			super(MLSequential, self).__init__(**kwargs)
			self.proba_thrsh = proba_thrsh
			self.context = context
			self.session = session
		def fit(self, X, y, **kwargs):
			with gen_cntxt(**self.context):
				return super(MLSequential, self).fit(X, y, **kwargs)
		def predict_classes(self, X, **kwargs):
			with gen_cntxt(**self.context):
				proba = self.predict(X, **kwargs)
			return (proba > self.proba_thrsh).astype('int8')
		def __del__(self):
			if (self.session is not None):
				self.session.close()
				del self.session
			super(MLSequential, self).__del__()
	return MLSequential(context=context, session=session, **kwargs)
	
	
def gen_cntxt(backend, device):
	'''
	Factory method for Keras context object
	'''
	class KerasContext():
		def __init__(self, be, dev):
			self.be = be
			self.dev = dev
			if (self.be == 'tf'):
				import keras.backend.tensorflow_backend as tfbe
				self.cntxt = tfbe.tf.device(self.dev)
			else:
				self.cntxt = None
		def __enter__(self):
			return self.cntxt
		def __exit__(self, type, value, traceback):
			pass
	if (backend == 'tf'):
		return KerasContext(backend, device).cntxt
	else:
		return KerasContext(backend, device)