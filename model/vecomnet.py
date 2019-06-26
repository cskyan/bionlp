#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2018 by Caspar. All rights reserved.
# File Name: vecomnet.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2018-03-28 15:42:51
###########################################################################
''' Vector Composition Network '''

import os, copy, itertools, functools

import numpy as np
import scipy.stats as stats

from keras.engine.topology import Layer
from keras.layers import Input, StackedRNNCells, RNN, LSTMCell, LSTM, Dense, Dropout, Lambda, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Subtract, Multiply, Concatenate
from keras.models import Model, Sequential, clone_model
from keras.optimizers import SGD, TFOptimizer
from keras.utils import plot_model
import keras.backend as K

from ..spider import w2v
from ..util import func
import kerasext


def get_cbow_context(words, indices, window_size=0, include_target=True):
	if (not words): return []
	indices = indices if type(indices) is list else list(indices)
	if (include_target):
		return [[words[:idx+1], words[-1:idx-len(words)-1:-1]] if type(idx) is not list else [words[:idx[-1]+1], words[-1:idx[0]-len(words)-1:-1]] for idx in indices] if window_size == 0 else [[words[max(idx-window_size+1, 0):idx+1], words[min(-1, idx-len(words)+window_size-1):idx-len(words)-1:-1]] if type(idx) is not list else [words[max(idx[-1]-window_size+1, 0):idx[-1]+1], words[min(-1, idx[0]-len(words)+window_size-1):idx[0]-len(words)-1:-1]] for idx in indices]
	else:
		return [[words[:idx], words[-1:idx:-1]] if type(idx) is not list else [words[:idx[0]], words[-1:idx[-1]:-1]] for idx in indices] if window_size == 0 else [[words[max(idx-window_size, 0):idx], words[min(idx+window_size, len(words)-1):idx:-1]] if type(idx) is not list else [words[max(idx[0]-window_size, 0):idx[0]], words[min(idx[-1]+window_size, len(words)-1):idx[-1]:-1]] for idx in indices]


def w_binary_crossentropy(Y_true, Y_pred, weights=None):
	if (weights is None): return K.binary_crossentropy(Y_true, Y_pred)
	weights = K.constant(np.clip(weights, 0.0, 1.0))
	Y_pred = K.clip(Y_pred, K.epsilon(), 1.0 - K.epsilon())
	out = - (weights * Y_true * K.log(Y_pred) + (1.0 - weights) * (1.0 - Y_true) * K.log(1.0 - Y_pred))
	# n_classes = len(weights)
	# out = - (1.0 / ((1.0 - weights) * n_classes) * Y_true * K.log(Y_pred) + 1.0 / (weights * n_classes) * (1.0 - Y_true) * K.log(1.0 - Y_pred))
	return K.mean(out, axis=-1)


def _vecomnet_loss(Y_true, Y_pred, weights=None):
	Y_pred = K.clip(Y_pred, K.epsilon(), 1.0 - K.epsilon())
	event = K.abs(Y_pred)
	direction = K.clip(K.sign(Y_pred), K.epsilon(), 1.0 - K.epsilon())
	event_true = K.abs(Y_true)
	direction_true = K.sign(Y_true)
	if (weights is None):
		out = - (event_true * K.log(event) + (1.0 - event_true) * K.log(1.0 - event)) - (direction_true * K.log(direction) + (1.0 - direction_true) * K.log(1.0 - direction))
		# return event_true * K.maximum(0.0, 0.9 - event)**2 + (1.0 - event_true) * K.maximum(0.0, event - 0.1)**2 + direction_true * (1.0 - direction)**2 + (1.0 - direction_true) * direction**2
	else:
		weights = K.constant(np.clip(weights, 0.0, 1.0))
		out = - (weights * event_true * K.log(event) + (1 - weights) * (1.0 - event_true) * K.log(1.0 - event)) - (direction_true * K.log(direction) + (1.0 - direction_true) * K.log(1.0 - direction))
	return K.mean(out, axis=-1)


CUSTOM_LOSS = {'_vecomnet_loss':_vecomnet_loss, 'w_binary_crossentropy':w_binary_crossentropy}


def vecomnet_mdl(input_dim=1, output_dim=1, w2v_path='wordvec.bin', cw2v_path=None, backend='tf', device='', session=None, cross_device=True, mltl_accc=False, lstm_dim=128, lstm_num=2, ent_mlp_dim=128, evnt_mlp_dim=64, drop_ratio=0.2, class_weight=None, pretrain_vecmdl=None, precomp_vec=None, avgembd_idx=None):
	main_cntxt = dict(device='/cpu:0') if cross_device else dict(device=device, session=session)
	with kerasext.gen_cntxt(backend, use_sess=True, **main_cntxt):
		X_inputs = [Input(shape=(input_dim,), dtype='int64', name='X%i'%i) for i in range(4)]
		if (pretrain_vecmdl is None and precomp_vec is None):
			w2v_wrapper = w2v.GensimW2VWrapper(w2v_path)
			embd_layer = w2v_wrapper.get_embedding_layer(type='keras', trainable=False, name='WordEmbedding')
			if (cw2v_path is not None and os.path.exists(cw2v_path)): # Append the concept embeddings to the original word embeddings layer
				print 'Using alternative concept embedding model!'
				cw2v_wrapper = w2v.GensimW2VWrapper(cw2v_path)
				cncpt_embd_layer = cw2v_wrapper.get_embedding_layer(type='keras', trainable=False, name='ConceptEmbedding')
				word_embeddings = [embd_layer(crop(1, 0, input_dim)(x)) for i, x in enumerate(X_inputs)]
				cncpt_embeddings = [cncpt_embd_layer(crop(1, input_dim - 1, input_dim)(x)) for i, x in enumerate(X_inputs)]
				w2v_dim, cw2v_dim = w2v_wrapper.get_dim(), cw2v_wrapper.get_dim()
				if (w2v_dim > cw2v_dim):
					dummy_tensor = [fill(2, 0, w2v_dim - cw2v_dim)(crop(2, 0, 1)(x)) for x in cncpt_embeddings]
					cncpt_embeddings = [Concatenate(axis=2)([cmbd, dummy]) for cmbd, dummy in zip(cncpt_embeddings, dummy_tensor)]
				elif (cw2v_dim > w2v_dim):
					dummy_tensor = [fill(2, 0, cw2v_dim - w2v_dim)(crop(2, 0, 1)(x)) for x in word_embeddings]
					word_embeddings = [Concatenate(axis=2)([wmbd, dummy]) for wmbd, dummy in zip(word_embeddings, dummy_tensor)]
				embeddings = [Concatenate(axis=1, name='FusionEmbedding%i'%i)([wmbd, cmbd]) for i, (wmbd, cmbd) in enumerate(zip(word_embeddings, cncpt_embeddings))]
			elif (avgembd_idx is not None):
				embeddings = [embd_layer(x) for i, x in enumerate(X_inputs)]
				wembeds, cembeds = zip(*[(crop(1, 0, avgembd_idx[0])(x), crop(1, avgembd_idx[0], avgembd_idx[1]))(x) for x in embeddings])
				embeddings = [Concatenate(axis=1)([wmbd, cmbd]) for wmbd, cmbd in zip(wembeds, [mask_avg(1, keep_dim=True)(cmbd) for cmbd in cembeds])]
			else:
				embeddings = [embd_layer(x) for i, x in enumerate(X_inputs)]
			with kerasext.gen_cntxt(backend, device, session=session):
				embeddings = [Dropout(drop_ratio, name='WordEmbedding%i-Rgl'%i)(embd) for i, embd in enumerate(embeddings)]
				# lstms = [RNN([LSTMCell(lstm_dim)] * lstm_num, name='BidirectionalLSTM%i-%s'%(i/2,'FW' if i%2==0 else 'BW'))(embd) for i, embd in enumerate(embeddings)]
				lstms = [LSTM(lstm_dim, name='BidirectionalLSTM%i-%s'%(i/2,'FW' if i%2==0 else 'BW'))(embd) for i, embd in enumerate(embeddings)]
				# lstms = [Dropout(drop_ratio, name='BidirectionalLSTM%i-%s-Rgl'%(i/2,'FW' if i%2==0 else 'BW'))(lstm) for i, lstm in enumerate(lstms)]
				cbow_cntxts = [Concatenate(name='CBOW0')(lstms[:2]), Concatenate(name='CBOW1')(lstms[2:])]
				mlps = [Dense(ent_mlp_dim, activation='tanh', input_shape=(lstm_dim*2,), name='MLP%i-L1'%i)(cntxt) for i, cntxt in enumerate(cbow_cntxts)]
				mlps = [Dropout(drop_ratio, name='MLP%i-L1-Rgl'%i)(x) for i, x in enumerate(mlps)]
				mlps = [Dense(ent_mlp_dim, activation='tanh', input_shape=(ent_mlp_dim,), name='MLP%i-L2'%i)(x) for i, x in enumerate(mlps)]
				mlps = [Dropout(drop_ratio, name='MLP%i-L2-Rgl'%i)(x) for i, x in enumerate(mlps)]
				mlps = [Dense(output_dim, activation='sigmoid', name='ENTITY%i'%i)(x) for i, x in enumerate(mlps)]
				vecom = Subtract(name='VecCom')(mlps)
				# mlps = [[Dense(ent_mlp_dim, activation='tanh', input_shape=(lstm_dim*2,), name='MLP%i.%i-L1'%(i,j))(cntxt) for j in range(output_dim)] for i, cntxt in enumerate(cbow_cntxts)]
				# mlps = [[Dropout(drop_ratio, name='MLP%i.%i-L1-Rgl'%(i,j))(y) for j, y in enumerate(x)] for i, x in enumerate(mlps)]
				# mlps = [[Dense(ent_mlp_dim, activation='tanh', input_shape=(ent_mlp_dim,), name='MLP%i.%i-L2'%(i,j))(y) for j, y in enumerate(x)] for i, x in enumerate(mlps)]
				# mlps = [[Dropout(drop_ratio, name='MLP%i.%i-L2-Rgl'%(i,j))(y) for j, y in enumerate(x)] for i, x in enumerate(mlps)]
				# vecom = Subtract(name='VecCom')([Concatenate(name='ENTITY%i'%i)(x) for i, x in enumerate(mlps)])
		elif (precomp_vec is None):
			mdls = [clone_model(pretrain_vecmdl), clone_model(pretrain_vecmdl)]
			# for mdl in mdls:
				# mdl.set_weights(pretrain_vecmdl.get_weights())
			X_inputs = func.flatten_list([[mdl.get_layer(name='X0'), mdl.get_layer(name='X1')] for mdl in mdls])
			for l_layer, r_layer in zip(mdls[0].layers, mdls[1].layers):
				l_layer.name = 'LF-' + l_layer.name
				r_layer.name = 'RT-' + r_layer.name
			vecom = Subtract(name='VecCom')([mdls[0].get_layer(name='LF-ENTITY').output, mdls[1].get_layer(name='RT-ENTITY').output])
		else:
			X_inputs = [Input(shape=(input_dim,), dtype='float32', name=name) for i, name in enumerate(['LF-ENTITY', 'RT-ENTITY'])]
			vecom = Subtract(name='VecCom')(X_inputs)
	with kerasext.gen_cntxt(backend, device, session=session):
		# vecom = Dropout(drop_ratio, name='VecCom-Rgl')(vecom)
		vecom_abs = Lambda(lambda x: K.abs(x), name='AbsVecCom')(vecom)
		event = Dense(evnt_mlp_dim, activation='relu', name='EVENT-L1')(vecom_abs)
		event = Dropout(drop_ratio, name='EVENT-L1-RGL')(event)
		event = Dense(evnt_mlp_dim, activation='relu', name='EVENT-L2')(event)
		event = Dropout(drop_ratio, name='EVENT-L2-RGL')(event)
		event = Dense(output_dim, activation='sigmoid', name='EVENT')(event)
		# event = Dropout(drop_ratio, name='EVENT-RGL')(event)
		direction = Dense(evnt_mlp_dim, activation='relu', name='DIRECTION-L1')(vecom)
		direction = Dropout(drop_ratio, name='DIRECTIONL1-RGL')(direction)
		direction = Dense(evnt_mlp_dim, activation='relu', name='DIRECTION-L2')(direction)
		direction = Dropout(drop_ratio, name='DIRECTIONL2-RGL')(direction)
		direction = Dense(output_dim, activation='sigmoid', name='DIRECTION')(direction)
		# direction = Dense(output_dim, activation='softsign', input_shape=(mlp_dim*2,), name='DIRECTION')(Concatenate(name='VecOrnt')(mlps))
		# direction = Lambda(lambda x: K.sign(2 * x - 1), name='DIRECT')(direction)
		# direction = Dropout(drop_ratio, name='DIRECTION-RGL')(direction)
		output = Concatenate(name='DirEvent')([event, direction])
		# output = Dropout(drop_ratio, name='DirEvent-Rgl')(output)
		model = Model(X_inputs, output)
		loss = func.wrapped_partial(w_binary_crossentropy, weights=class_weight.mean(axis=0) if (output_dim > 1 and class_weight is not None) else class_weight)
		optmzr = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		plot_model(model, show_shapes=True, to_file='model.png')
		kwargs = {}
		if (backend == 'tf'):
			import tensorflow as tf
			# kwargs['options'] = tf.RunOptions(report_tensor_allocations_upon_oom=True) # Will crash if low memory
			optmzr = TFOptimizer(tf.train.GradientDescentOptimizer(learning_rate=0.1))
			with session.as_default():
				tf.global_variables_initializer().run()
		model.compile(optimizer=optmzr, loss='binary_crossentropy', metrics=['acc', 'mse'], **kwargs)
	return model


def vecentnet_mdl(input_dim=1, output_dim=1, w2v_path='wordvec.bin', cw2v_path=None, backend='tf', device='', session=None, cross_device=True, mltl_accc=False, conv_dim=128, conv_ksize=4, maxp_size=2, lstm_dim=128, lstm_num=2, mlp_dim=128, drop_ratio=0.2, class_weight=None, pretrain_vecmdl=None, avgembd_idx=None):
	if (pretrain_vecmdl is not None): return pretrain_vecmdl
	main_cntxt = dict(device='/cpu:0') if cross_device else dict(device=device, session=session)
	# import tensorflow as tf
	# graph = tf.get_default_graph()
	with kerasext.gen_cntxt(backend, use_sess=True, **main_cntxt):
	# with kerasext.gen_cntxt(backend, device, session=session):
	# with tf.device('/cpu:0'):
		X_inputs = [Input(shape=(input_dim,), dtype='int64', name='X%i'%i) for i in range(2)]
		w2v_wrapper = w2v.GensimW2VWrapper(w2v_path)
		embd_layer = w2v_wrapper.get_embedding_layer(type='keras', trainable=False, name='WordEmbedding')
		if (cw2v_path is not None and os.path.exists(cw2v_path)): # Append the concept embeddings to the original word embeddings layer
			print 'Using alternative concept embedding model!'
			cw2v_wrapper = w2v.GensimW2VWrapper(cw2v_path)
			word_embeddings = [embd_layer(crop(1, 0, input_dim)(x)) for i, x in enumerate(X_inputs)]
			cncpt_embeddings = [embd_layer(crop(1, input_dim - 1, input_dim)(x)) for i, x in enumerate(X_inputs)]
			w2v_dim, cw2v_dim = w2v_wrapper.get_dim(), cw2v_wrapper.get_dim()
			if (w2v_dim > cw2v_dim):
				dummy_tensor = [fill(2, 0, w2v_dim - cw2v_dim)(crop(2, 0, 1)(x)) for x in cncpt_embeddings]
				cncpt_embeddings = [Concatenate(axis=2)([cmbd, dummy]) for cmbd, dummy in zip(cncpt_embeddings, dummy_tensor)]
			elif (cw2v_dim > w2v_dim):
				dummy_tensor = [fill(2, 0, cw2v_dim - w2v_dim)(crop(2, 0, 1)(x)) for x in word_embeddings]
				word_embeddings = [Concatenate(axis=2)([wmbd, dummy]) for wmbd, dummy in zip(word_embeddings, dummy_tensor)]
			embeddings = [Concatenate(axis=1, name='FusionEmbedding%i'%i)([wmbd, cmbd]) for i, (wmbd, cmbd) in enumerate(zip(word_embeddings, cncpt_embeddings))]
		elif (avgembd_idx is not None):
			print('Averaging the concept embedding...')
			embeddings = [embd_layer(x) for i, x in enumerate(X_inputs)]
			wembeds, cembeds = zip(*[(crop(1, 0, avgembd_idx[0])(x), crop(1, avgembd_idx[0], avgembd_idx[1])(x)) for x in embeddings])
			embeddings = [Concatenate(axis=1)([wmbd, cmbd]) for wmbd, cmbd in zip(wembeds, [mask_avg(1, keep_dim=True)(cmbd) for cmbd in cembeds])]
		else:
			embeddings = [embd_layer(x) for i, x in enumerate(X_inputs)]
	with kerasext.gen_cntxt(backend, device, session=session):
		embeddings = [Dropout(drop_ratio, name='WordEmbedding%i-Rgl'%i)(embd) for i, embd in enumerate(embeddings)]
		# lstms = [RNN([LSTMCell(lstm_dim)] * lstm_num, name='BidirectionalLSTM%i-%s'%(i/2,'FW' if i%2==0 else 'BW'))(embd) for i, embd in enumerate(embeddings)]
		lstms = [LSTM(lstm_dim, name='BidirectionalLSTM-%s'%('FW' if i%2==0 else 'BW'))(embd) for i, embd in enumerate(embeddings)]
		# convs = [Conv1D(filters=conv_dim, kernel_size=conv_ksize, padding='same', activation='tanh', name='Convolution%i'%i)(embd) for i, embd in enumerate(embeddings)]
		# lstms = [LSTM(lstm_dim, name='BidirectionalLSTM-%s'%('FW' if i%2==0 else 'BW'))(conv) for i, conv in enumerate(convs)]
		# maxpools = [MaxPooling1D(pool_size=maxp_size, name='MaxPooling%i'%i)(conv) for i, conv in enumerate(convs)]
		# lstms = [LSTM(lstm_dim, return_sequences=True, name='BidirectionalLSTM-%s'%('FW' if i%2==0 else 'BW'))(maxp) for i, maxp in enumerate(maxpools)]
		# lstms = [GlobalMaxPooling1D(name='GlobalMaxPooling%i'%i)(lstm) for i, lstm in enumerate(lstms)]
		# lstms = [Dropout(drop_ratio, name='BidirectionalLSTM%s-Rgl'%('FW' if i%2==0 else 'BW'))(lstm) for i, lstm in enumerate(lstms)]
		cbow_cntxt = Concatenate(name='CBOW')(lstms)
		mlp = Dense(mlp_dim, activation='tanh', input_shape=(lstm_dim*2,), name='MLP-L1')(cbow_cntxt)
		mlp = Dropout(drop_ratio, name='MLP-L1-Rgl')(mlp)
		mlp = Dense(mlp_dim, activation='tanh', input_shape=(mlp_dim,), name='MLP-L2')(mlp)
		mlp = Dropout(drop_ratio, name='MLP-L2-Rgl')(mlp)
		output = Dense(output_dim, activation='sigmoid', input_shape=(mlp_dim,), name='ENTITY')(mlp)
		# mlps = [Dense(mlp_dim, activation='tanh', input_shape=(lstm_dim*2,), name='MLP%i-L1'%i)(cbow_cntxt) for i in range(output_dim)]
		# mlps = [Dropout(drop_ratio, name='MLP%i-L1-Rgl'%i)(mlp) for i, mlp in enumerate(mlps)]
		# mlps = [Dense(mlp_dim, activation='tanh', input_shape=(mlp_dim,), name='MLP%i-L2'%i)(mlp) for i, mlp in enumerate(mlps)]
		# mlps = [Dropout(drop_ratio, name='MLP%i-L2-Rgl'%i)(mlp) for i, mlp in enumerate(mlps)]
		# mlps = [Dense(1, activation='sigmoid', input_shape=(mlp_dim,), name='ENTITY%i'%i)(mlp) for i, mlp in enumerate(mlps)]
		# output = Concatenate(name='ENTITY')(mlps) if len(mlps) > 1 else mlps[0]
		model = Model(X_inputs, output)
		optmzr = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		loss = func.wrapped_partial(w_binary_crossentropy, weights=class_weight.mean(axis=0) if (output_dim > 1 and class_weight is not None) else class_weight)
		# plot_model(model, show_shapes=True, to_file='model.png')
		kwargs = {}
		if (backend == 'tf'):
			import tensorflow as tf
			# kwargs['options'] = tf.RunOptions(report_tensor_allocations_upon_oom=True) # Will crash if low memory
			optmzr = TFOptimizer(tf.train.GradientDescentOptimizer(learning_rate=0.1))
			with session.as_default():
				tf.global_variables_initializer().run()
		model.compile(optimizer=optmzr, loss=loss, metrics=['acc', 'mse'], **kwargs)
	return model


def mlmt_vecentnet_mdl(input_dim=1, output_dim=1, w2v_path='wordvec.bin', backend='th', device='', session=None, conv_dim=128, conv_ksize=4, maxp_size=2, lstm_dim=128, lstm_num=2, mlp_dim=128, drop_ratio=0.2, class_weight=None):
	with kerasext.gen_cntxt(backend, device):
		X_inputs = [Input(shape=(input_dim,), dtype='int64', name='X%i'%i) for i in range(2)]
		w2v_wrapper = w2v.GensimW2VWrapper(w2v_path)
		embd_layer = w2v_wrapper.get_embedding_layer(type='keras', trainable=False, name='WordEmbedding')
		embeddings = [embd_layer(x) for i, x in enumerate(X_inputs)]
		embeddings = [Dropout(drop_ratio, name='WordEmbedding%i-Rgl'%i)(embd) for i, embd in enumerate(embeddings)]
		lstms = [LSTM(lstm_dim, name='BidirectionalLSTM-%s'%('FW' if i%2==0 else 'BW'))(embd) for i, embd in enumerate(embeddings)]
		cbow_cntxt = concatenate(lstms, name='CBOW')
		mlps = [Dense(mlp_dim, activation='tanh', input_shape=(lstm_dim*2,), name='MLP%i-L1'%i)(cbow_cntxt) for i in range(output_dim)]
		# lstms = [[LSTM(lstm_dim, name='BidirectionalLSTM%i-%s'%(j, 'FW' if i%2==0 else 'BW'))(embd) for i, embd in enumerate(embeddings)] for j in range(output_dim)]
		# cbow_cntxt = [concatenate(lstm, name='CBOW%i' % i) for i, lstm in enumerate(lstms)]
		# mlps = [Dense(mlp_dim, activation='tanh', input_shape=(lstm_dim*2,), name='MLP%i-L1'%i)(cbow_cntxt[i]) for i in range(output_dim)]
		mlps = [Dropout(drop_ratio, name='MLP%i-L1-Rgl'%i)(mlp) for i, mlp in enumerate(mlps)]
		mlps = [Dense(mlp_dim, activation='tanh', input_shape=(mlp_dim,), name='MLP%i-L2'%i)(mlp) for i, mlp in enumerate(mlps)]
		mlps = [Dropout(drop_ratio, name='MLP%i-L2-Rgl'%i)(mlp) for i, mlp in enumerate(mlps)]
		mlps = [Dense(1, activation='sigmoid', input_shape=(mlp_dim,), name='ENTITY%i'%i)(mlp) for i, mlp in enumerate(mlps)]
		train_models = [Model(X_inputs, output) for output in mlps]
		for i, model in enumerate(train_models):
			loss = loss = func.wrapped_partial(w_binary_crossentropy, weights=class_weight[i] if class_weight is not None else None)
			optmzr = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
			model.compile(optimizer=optmzr, loss=loss, metrics=[kerasext.f1])
		output = concatenate(mlps, name='ENTITY') if len(mlps) > 1 else mlps[0]
		predict_model = Model(X_inputs, output)
		plot_model(predict_model, show_shapes=True, to_file='model.png')
	return train_models, predict_model


def crop(axis, start, end):
	def func(x):
		if axis == 0:
			return x[start:end]
		if axis == 1:
			return x[:,start:end]
		if axis == 2:
			return x[:,:,start:end]
		if axis == 3:
			return x[:,:,:,start:end]
		if axis == 4:
			return x[:,:,:,:,start:end]
	return Lambda(func)


def fill(axis, val=0, rep=1, expand_dims=False):
	def func(x):
		return K.repeat_elements(val * K.ones_like(K.expand_dims(x, axis=axis)), rep=rep, axis=axis) if expand_dims else K.repeat_elements(val * K.ones_like(x), rep=rep, axis=axis)
	return Lambda(func)


def mask_avg(axis, keep_dim=False):
	def func(x):
		reduced_sum = K.sum(x, axis=axis) / K.sum(K.cast(K.not_equal(x, 0), 'float32'), axis=axis)
		return K.expand_dims(reduced_sum, axis=axis) if keep_dim else reduced_sum
	return Lambda(func)
