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

import os, re, sys, traceback

import numpy as np
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model, model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

from ..util import fs, io, math, func, njobs

BEMAP = {'th':'theano', 'tf':'tensorflow'}

NUM_PROCESS, PHSCL_DEV_ID, DEVICE_ID, DEV, DEV_IDQ, DEVICE_INIT, DEVICE_VARS = 0, 0, 0, '', '', False, {'gpu_mem':0.95}


class BaseWrapper(KerasClassifier):
	def __init__(self, context={}, proba_thrsh=0.5, n_jobs=1, **kw_args):
		super(BaseWrapper, self).__init__(**kw_args)
		self.context = context
		self.proba_thrsh = proba_thrsh
		self.n_jobs = n_jobs
		self.histories = []
	# def __del__(self):
		# if (hasattr(self, 'context') and self.context.setdefault('session', None) is not None):
			# self.context['session'].close()
			# del self.context['session']
			# if (self.context.has_key('backend') and self.context['backend'] == 'tf'):
				# import keras.backend.tensorflow_backend as K
				# K.clear_session()
				# K.set_session(None)
	def fit(self, X, y, **kw_args):
		try:
			with gen_cntxt(**self.context):
				self.histories.append(super(BaseWrapper, self).fit(X, y, verbose=DEVICE_VARS['verbose'], **kw_args))
				return self
		except Exception as e:
			print '-' * 60
			traceback.print_exc(file=sys.stderr)
			print '-' * 60
			# print e
	def predict(self, X, **kw_args):
		with gen_cntxt(**self.context):
			return super(BaseWrapper, self).predict(X, **kw_args)
	def save(self, fname='kerasmdl', sep_arch=False, save_history=True):
		fname = os.path.splitext(fname)[0]
		model_params = self.filter_sk_params(self.build_fn)
		build_fn, sk_params, model, histories, context = self.build_fn, self.sk_params, self.model, self.histories, self.context
		self.build_fn, self.sk_params, self.model, self.histories, self.context = build_fn.__name__, {k:v for k, v in sk_params.iteritems() if k not in model_params.keys()}, None, [], {}
		io.write_obj(self, fname)
		if (sep_arch):
			fs.write_file(model.to_json(), '%s.json' % fname)
			model.save_weights('%s.h5' % fname)
		else:
			model.save('%s.h5' % fname)
		if (save_history):
			io.write_obj([his.history for his in histories], '%s_histories' % fname)
		self.build_fn, self.sk_params, self.model, self.context = build_fn, sk_params, model, context
	def load(self, fname='kerasmdl', custom_objects={}, sep_arch=False):
		fname = os.path.splitext(fname)[0]
		if (sep_arch):
			self.model = model_from_json('\n'.join(fs.read_file('%s.json' % os.path.splitext(fname)[0])), custom_objects=custom_objects)
			self.model.load_weights('%s.h5' % os.path.splitext(fname)[0])
		else:
			self.model = load_model('%s.h5' % fname, custom_objects=custom_objects)
	# def clear(self):
		# if DEVICE_VARS.has_key('sess'):
			# DEVICE_VARS['sess'].close()
			# del DEVICE_VARS['sess']
		# if (self.context.has_key('backend') and self.context['backend'] == 'tf'):
			# import keras.backend.tensorflow_backend as K
			# K.clear_session()
			# K.set_session(None)

'Multiple Label Classifier'
class MLClassifier(BaseWrapper):
	def __init__(self, mlmt=False, **kw_args):
		super(MLClassifier, self).__init__(**kw_args)
		self.mlmt = mlmt
		# self.attributes.extend(['mlmt'])
	def fit(self, X, y, **kw_args):
		y = np.array(y)
		if (self.mlmt and len(y.shape)>1 and y.shape[1]>1):
			'Multi-label Multi-trainning Model'
			try:
				with gen_cntxt(**self.context):
					# Copy from keras -- START
					import types, copy
					if len(y.shape) == 2 and y.shape[1] > 1:
						self.classes_ = np.arange(y.shape[1])
					elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
						self.classes_ = np.unique(y)
						y = np.searchsorted(self.classes_, y)
					else:
						raise ValueError('Invalid shape for y: ' + str(y.shape))
					self.n_classes_ = len(self.classes_)
					# Class method separator
					if self.build_fn is None:
						self.train_models, self.predict_model = self.__call__(**self.filter_sk_params(self.__call__))
					elif (not isinstance(self.build_fn, types.FunctionType) and
						  not isinstance(self.build_fn, types.MethodType)):
						self.train_models, self.predict_model = self.build_fn(
							**self.filter_sk_params(self.build_fn.__call__))
					else:
						self.train_models, self.predict_model = self.build_fn(**self.filter_sk_params(self.build_fn))
					loss_name = self.train_models[0].loss
					if hasattr(loss_name, '__name__'):
						loss_name = loss_name.__name__
					if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
						y = to_categorical(y)
					fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
					fit_args.update(kw_args)
					# history = njobs.run_pool(lambda x: self.train_models[x].fit(X, y[:,x], verbose=DEVICE_VARS['verbose'], **fit_args), n_jobs=self.n_jobs, dist_param=['x'], x=range(y.shape[1])) if self.n_jobs > 1 else [self.train_models[i].fit(X, y[:,i], **fit_args) for i in range(y.shape[1])]
					history, all_ids = [], set(range(y.shape[1]))
					for i in range(y.shape[1]):
						for j in (all_ids - set([i])):
							self.train_models[j].trainable = False
						self.train_models[i].trainable = True
						history.append(self.train_models[i].fit(X, y[:,i], verbose=DEVICE_VARS['verbose'], **fit_args))
					self.histories.append(history)
					return self
					# Copy from keras -- END
				# self.attributes.extend(['train_models', 'predict_model'])
			except Exception as e:
				print e
		else:
			self.mlmt = False
			return super(MLClassifier, self).fit(X, y, **kw_args)
	def predict(self, X, **kw_args):
		kw_args = self.filter_sk_params(Sequential.predict_proba, kw_args)
		with gen_cntxt(**self.context):
			probs = self.model.predict(X, **kw_args) if hasattr(self, 'model') else self.predict_model.predict(X, **kw_args)
			probs = probs.reshape(probs.shape[:-1]) if (probs.shape[-1] == 1) else probs
		probs[probs <= self.proba_thrsh] = 0
		return np.sign(probs).astype('int8')
	def predict_classes(self, X, **kw_args):
		with gen_cntxt(**self.context):
			proba = self.model.predict(X, **kw_args) if hasattr(self, 'model') else self.predict_model.predict(X, **kw_args)
			probs = probs.reshape(probs.shape[:-1]) if (probs.shape[-1] == 1) else probs
		return (proba > self.proba_thrsh).astype('int8')
	def predict_proba(self, X, **kw_args):
		kw_args = self.filter_sk_params(Sequential.predict_proba, kw_args)
		with gen_cntxt(**self.context):
			probs = self.model.predict(X, **kw_args) if hasattr(self, 'model') else self.predict_model.predict(X, **kw_args)
		if probs.shape[-1] == 1:
			probs = np.hstack([1 - probs, probs])
		return probs
	def save(self, fname='kerasmdl', sep_arch=False, save_history=True):
		if (not self.mlmt):
			super(MLClassifier, self).save(fname=fname, sep_arch=sep_arch, save_history=save_history)
		else:
			# For MLMT Model
			model_params = self.filter_sk_params(self.build_fn)
			build_fn, sk_params, train_models, predict_model, histories, context = self.build_fn, self.sk_params, self.train_models, self.predict_model, self.histories, self.context
			self.build_fn, self.sk_params, self.train_models, self.predict_model, self.context = None, {k:v for k, v in sk_params.iteritems() if k not in model_params.keys()}, None, None, {}
			io.write_obj(self, fname)
			if (sep_arch):
				for i, train_mdl in enumerate(train_models):
					fs.write_file(train_mdl.to_json(), 'train_%s_%i.json' % (fname, i))
					train_mdl.save_weights('train_%s_%i.h5' % (fname, i))
				fs.write_file(predict_model.to_json(), 'predict_%s_%i.json' % (fname, i))
				predict_model.save_weights('predict_%s_%i.h5' % (fname, i))
			else:
				for i, train_mdl in enumerate(train_models):
					train_mdl.save('train_%s_%i.h5' % (fname, i))
				predict_model.save('predict_%s.h5' % fname)
			if (save_history):
				for i, history in enumerate(histories):
					io.write_obj([his.history for his in history], '%s_histories_%i' % (fname, i))
			self.build_fn, self.sk_params, self.train_models, self.predict_model, self.context = build_fn, sk_params, train_models, predict_model, context
	def load(self, fname='kerasmdl', custom_objects={}, sep_arch=False):
		if (not self.mlmt):
			super(MLClassifier, self).load(fname=fname, custom_objects=custom_objects, sep_arch=sep_arch)
		else:
			# For MLMT Model
			basename = os.path.splitext(os.path.basename(fname))[0]
			if (sep_arch):
				self.train_models = []
				for fpath in fs.listf(os.path.dirname(os.path.abspath(fname)), pattern='train_%s_.*' % basename, full_path=True):
					train_mdl = model_from_json('\n'.join(fs.read_file('%s.json' % os.path.splitext(fpath)[0])), custom_objects=custom_objects)
					train_mdl.load_weights('%s.h5' % os.path.splitext(fpath)[0])
					self.train_models.append(train_mdl)
				self.predict_model = model_from_json('\n'.join(fs.read_file(os.path.join(os.path.dirname(fname), 'predict_%s.json' % basename))), custom_objects=custom_objects)
				self.predict_model.load_weights('%s.h5' % os.path.join(os.path.dirname(fname), 'predict_%s.h5' % basename))
			else:
				self.train_models = [load_model(fpath, custom_objects=custom_objects) for fpath in fs.listf(os.path.dirname(os.path.abspath(fname)), pattern='train_%s_.*' % basename, full_path=True)]
				self.predict_model = load_model(os.path.join(os.path.dirname(fname), 'predict_%s.h5' % basename), custom_objects=custom_objects)

		
'Signed Label Classifier'
class SignedClassifier(BaseWrapper):
	def predict(self, X, **kw_args):
		kw_args = self.filter_sk_params(Sequential.predict_proba, kw_args)
		with gen_cntxt(**self.context):
			probs = self.model.predict(X, **kw_args)
		probs[np.abs(probs) <= self.proba_thrsh] = 0
		return np.nan_to_num(np.sign(probs)).astype('int8')
	def predict_classes(self, X, **kw_args):
		kw_args = self.filter_sk_params(Sequential.predict_classes, kw_args)
		with gen_cntxt(**self.context):
			proba = self.model.predict(X, **kw_args)
		if proba.shape[-1] > 1:
			classes = np.abs(proba).argmax(axis=-1)
			return np.nan_to_num(np.sign(math.slice_last_axis(proba, classes)) * self.classes_[classes]).astype('int8')
		else:
			classes = (np.abs(proba) > 0.5).astype('int32')
			return np.nan_to_num(np.sign(proba) * self.classes_[classes]).astype('int8')
	def predict_proba(self, X, **kw_args):
		kw_args = self.filter_sk_params(Sequential.predict_proba, kw_args)
		with gen_cntxt(**self.context):
			probs = self.model.predict(X, **kw_args)
		if probs.shape[-1] == 1:
			probs = np.nan_to_num(np.hstack([np.sign(probs) * (1 - np.abs(probs)), probs]))
		else:
			probs = np.nan_to_num(probs)
		return probs

		
class KerasCluster(KerasClassifier):
	def __init__(self, batch_size=32, **kw_args):
		self.kw_args = kw_args
		self.batch_size = batch_size
		# self.attributes.extend(['batch_size'])
		super(KerasCluster, self).__init__(build_fn=build_fn)
	def fit(self, X, y, constraint=None, **kw_args):
		if self.build_fn is None:
			self.model = self.__call__(**self.filter_sk_params(self.__call__))
		elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
			self.kw_args.update(self.filter_sk_params(self.build_fn.__call__))
			self.model = self.build_fn(batch_size=self.batch_size, **self.kw_args)
		else:
			self.kw_args.update(self.filter_sk_params(self.build_fn))
			self.model = self.build_fn(batch_size=self.batch_size, **self.kw_args)
		loss_name = self.model.loss
		if hasattr(loss_name, '__name__'):
			loss_name = loss_name.__name__
		if loss_name == 'categorical_crossentropy' and len(y.shape) != 2 and y.shape[1] < 2:
			y = to_categorical(y)
		fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
		fit_args.update(kw_args)
		if (constraint is None):
			history = self.model.fit(X, y, batch_size=self.batch_size, verbose=DEVICE_VARS['verbose'], **fit_args)
		else:
			history = self.model.fit([X, constraint], y, batch_size=self.batch_size, verbose=DEVICE_VARS['verbose'], **fit_args)
		return history
	def predict(self, X, constraint=None, **kw_args):
		if (constraint is None):
			return super(KerasCluster, self).predict(X, **kw_args)
		else:
			return super(KerasCluster, self).predict([X, constraint], **kw_args)
		

def init(dev_id=0, backend='th', num_gpu=0, gpuq=[0], gpu_mem=None, num_process=1, use_omp=False, verbose=False):
	global NUM_PROCESS, PHSCL_DEV_ID, DEVICE_ID, DEV, DEV_IDQ, DEVICE_INIT
	if (DEVICE_INIT): return
	NUM_PROCESS, PHSCL_DEV_ID, DEVICE_ID = num_process, gpuq[dev_id], dev_id
	os.environ['KERAS_BACKEND'] = BEMAP[backend]
	DEV_IDQ = ','.join(map(str, gpuq))
	if (type(gpu_mem) is float and gpu_mem > 0 and gpu_mem <= 1): DEVICE_VARS['gpu_mem'] = gpu_mem
	DEVICE_VARS['verbose'] = verbose
	if (backend == 'th'):
		DEV = 'gpu%i' % dev_id if num_gpu > 0 else 'cpu'
		os.environ['THEANO_FLAGS'] = re.sub('device=.+,', 'device=%s,' % DEV, os.environ['THEANO_FLAGS'])
		# Multiple cpus
		if (DEV == 'cpu' and use_omp):
			os.environ['THEANO_FLAGS'] = os.environ['THEANO_FLAGS'] + ',openmp=True'
		if (verbose == True):
			os.environ['THEANO_FLAGS'] = os.environ['THEANO_FLAGS'] + ',optimizer=None,exception_verbosity=high'
		# import keras.backend.theano_backend as K
	elif (backend == 'tf'):
		DEV = '/gpu:%i' % dev_id if num_gpu > 0 else '/cpu:0'
		if (DEV.startswith('/cpu')): os.environ['CUDA_VISIBLE_DEVICES'] = ''
		# import keras.backend.tensorflow_backend as K
	from keras import backend as K
	reload(K)
	DEVICE_INIT = True
	
	
def get_activations(model, X, layer_name=None):
	import keras.backend as K
	output_layer = model.get_layer(layer_name) if layer_name else model.layers[-1]
	get_activations_func = K.function([model.layers[0].input, K.learning_phase()], output_layer.output)
	activations = get_activations_func([X[0],0])
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


# Factory method for generating Keras Model
def gen_mdl(input_dim, output_dim, model, mdl_type='clf', backend='th', session=None, verbose=False, udargs=['input_dim', 'output_dim', 'backend', 'device', 'session'], **kwargs):
	global PHSCL_DEV_ID, DEVICE_ID, DEV, DEV_IDQ, DEVICE_INIT, DEVICE_VARS
	if (not DEVICE_INIT):
		init(backend=backend)
	device = DEV
	# Each model keeps one session, create a new one or reuse the previous one
	if (session is None and backend == 'tf'):
		import keras.backend.tensorflow_backend as K
		with gen_cntxt(backend, device):
			if (device.lower().startswith('/gpu')):
				config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=verbose)
				config.gpu_options.visible_device_list = DEV_IDQ
				config.gpu_options.per_process_gpu_memory_fraction=DEVICE_VARS['gpu_mem']
				config.gpu_options.allow_growth=True
			else:
				config = K.tf.ConfigProto(device_count={'gpu':0, 'cpu': 1}, allow_soft_placement=True, inter_op_parallelism_threads=NUM_PROCESS, intra_op_parallelism_threads=NUM_PROCESS, log_device_placement=verbose)
			# if DEVICE_VARS.has_key('sess'):
				# DEVICE_VARS['sess'].close()
				# del DEVICE_VARS['sess']
			from numba import cuda
			try:
				cuda.get_current_device().reset()
			except Exception as e:
				print e
			K.clear_session()
			session = K.tf.Session(config=config)
			# session = DEVICE_VARS.setdefault('sess', K.tf.Session(config=config))
	K.set_session(session)
	if (udargs):
		kwargs.update({k:v for k, v in [(x, locals()[x]) for x in udargs]})
	if (mdl_type == 'clf'):
		return BaseWrapper(build_fn=model, verbose=verbose, **kwargs)
	elif (mdl_type == 'mlclf'):
		return MLClassifier(build_fn=model, context=dict(backend=backend, device=device, session=session), verbose=verbose, **kwargs)
	elif (mdl_type == 'signedclf'):
		return SignedClassifier(build_fn=model, context=dict(backend=backend, device=device, session=session), verbose=verbose, **kwargs) 
	elif (mdl_type == 'clt'):
		return KerasCluster(build_fn=model, verbose=verbose, **kwargs)
		
		
def gen_cltmdl(proba_thrsh=0.5, context=None, session=None, **kwargs):
	'''
	Factory method for Clustering Model
	'''
	import keras.backend as K
	from keras.models import Model
	class CLTModel(Model):
		def __init__(self, proba_thrsh=0.5, constraint_dim=0, context=None, session=None, **kw_args):
			super(CLTModel, self).__init__(**kw_args)
			self.constraint_dim = constraint_dim
			self.proba_thrsh = proba_thrsh
			self.context = context
			self.session = session
		def fit(self, X, y, **kw_args):
			with gen_cntxt(**self.context):
				return super(CLTModel, self).fit(X, y, **kw_args)
		def predict_classes(self, X, **kw_args):
			with gen_cntxt(**self.context):
				proba = get_activations(self, X)
				if (kw_args.setdefault('proba', False)):
					return (proba > self.proba_thrsh).astype('int8')
				else:
					return proba
		def __del__(self):
			if (self.session is not None):
				self.session.close()
				del self.session
			super(CLTModel, self).__del__()
	return CLTModel(context=context, session=session, **kw_args)
	
	
def gen_cntxt(backend='th', device=DEV, session=None):
	'''
	Factory method for Keras context object
	'''
	class KerasContext():
		def __init__(self, be, dev, sess):
			self.be = be
			self.dev = dev
			self.sess = sess
			if (self.be == 'tf'):
				import keras.backend.tensorflow_backend as tfbe
				self.cntxt = tfbe.tf.device(self.dev)
			else:
				self.cntxt = None
		def __enter__(self):
			return self.cntxt
		def __exit__(self, type, value, traceback):
			pass
	if (not DEVICE_INIT):
		init(backend=backend)
		device = DEV
	if (backend == 'tf'):
		return KerasContext(backend, device, session).cntxt
	else:
		return KerasContext(backend, device, session)
		

## Custom Metrics		
def precision(y_true, y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true, y_pred):
	"""Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def f1(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    return 2 * ((precision_score * recall_score) / (precision_score + recall_score + K.epsilon()))


CUSTOM_METRIC = {'f1':f1}