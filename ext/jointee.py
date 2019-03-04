#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2018 by Caspar. All rights reserved.
# File Name: jointee.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2018-03-28 15:42:51
###########################################################################
''' Vector Composition Network '''

import os, copy, itertools, functools

import numpy as np
import scipy.stats as stats

from keras.engine.topology import Layer
from keras.layers import Input, StackedRNNCells, RNN, GRU, LSTMCell, LSTM, Dense, Dropout, Lambda, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Subtract, Multiply, Concatenate, Reshape
from keras.models import Model, Sequential, clone_model
from keras.optimizers import SGD
from keras.utils import plot_model
import keras.backend as K

from ..spider import w2v
from ..util import func
from ..util import math as imath
import kerasext

	
def jointee_mdl(input_dim=1, ent_dim=1, depend_dim=1, output_dim=1, w2v_path='wordvec.bin', backend='tf', device='', session=None, half_window_size=2, gru_dim=128, mlp_dim=128, drop_ratio=0.2):
	with kerasext.gen_cntxt(backend, device):
		word_vec_ids, ent_type_ids = [Input(shape=(input_dim,), dtype='int64', name='%s' % in_name) for in_name in ['Words', 'EntityType']]
		ent_ids = Input(shape=(ent_dim,), dtype='int64', name='Entities')
		ent_indicators = [Input(shape=(input_dim, ent_dim,), dtype='int8', name='EntityMention-%s'%('Head' if i%2==0 else 'Tail')) for x in range(2)]
		depend_vecs = Input(shape=(input_dim, depend_dim), dtype='int8', name='Dependency')
		X_inputs = [word_vec_ids, ent_type_ids, ent_ids, depend_vecs]
		
		w2v_wrapper = w2v.GensimW2VWrapper(w2v_path)
		last_widx, wordvec_dim = w2v_wrapper.get_vocab_size() - 1, w2v_wrapper.get_dim()
		word_embeddings = w2v_wrapper.get_embedding_layer(type='keras', name='WordEmbedding')(word_vec_ids)
		ent_embeddings = w2v_wrapper.get_embedding_layer(type='keras', name='EntityEmbedding')(ent_ids)
		entype_embeddings = w2v_wrapper.get_embedding_layer(type='keras', name='EntTypeEmbedding')(ent_type_ids)
		
		sent_vecs = Concatenate(name='SentEncoding-%s'%('FW' if i%2==0 else 'BW'))([word_embeddings, entype_embeddings, depend_vecs])
		sent_vecs = Dropout(drop_ratio, name='SentEncoding')(sent_vecs)
		grus = Bidirectional(GRU(gru_dim, name='BidirectionalGRU', return_sequences=True))(sent_vecs)
		window_size = 2 * half_window_size + 1
		word_vec_sw_orig = w2v_wrapper.get_embedding_layer(type='keras', name='LocalContext')(imath.slide_window(word_vec_ids, half_window_size=half_window_size, padding_val=last_widx))
		word_vec_sws = Reshape(target_shape=(input_dim, window_size * wordvec_dim))(word_vec_sw_orig)
		
		trigger_vecs = Concatenate(name='TriggerEncoding'))([grus, word_vec_sws])
		trigger_mlp = Dense(mlp_dim, activation='tanh', input_shape=(2 * gru_dim,), name='Trigger-MLP-L1')(trigger_vecs)
		trigger_mlp = Dropout(drop_ratio, name='Trigger-MLP-L1-Rgl')(trigger_mlp)
		trigger_mlp = Dense(mlp_dim, activation='tanh', input_shape=(mlp_dim,), name='Trigger-MLP-L2')(trigger_mlp)
		trigger_mlp = Dropout(drop_ratio, name='Trigger-MLP-L2-Rgl')(trigger_mlp)
		trigger_output = Dense(output_dim, activation='softmax', input_shape=(mlp_dim,), name='TRIGGER')(trigger_mlp)
		
		is_trigger = K.cast(K.greater(K.sum(K.round(trigger_output), axis=-1), 0), 'int8')
		word_ent_indicators = [K.repeat_elements(Reshape(target_shape=(input_dim, ent_dim, 1))(x), 2 * gru_dim, axis=-1) for i, x in enumerate(ent_indicators)]
		dup_grus = K.repeat_elements(Reshape(target_shape=(input_dim, 1, 2 * gru_dim))(grus), ent_dim, axis=-2)
		ent_grus = Concatenate(name='EntityGRU')([K.sum(Multiply(names='EntityGRU-%s'%('Head' if i%2==0 else 'Tail'))([dup_grus, x]), axis=-3) for i, x in enumerate(word_ent_indicators)])
		wordsw_ent_indicators = [K.repeat_elements(Reshape(target_shape=(input_dim, ent_dim, 1))(x), half_window_size * wordvec_dim, axis=-1) for i, x in enumerate(ent_indicators)]
		a = K.sum(Multiply()([wordsw_ent_indicators, K.repeat_elements(Reshape(target_shape=(input_dim, 1, half_window_size * wordvec_dim))(word_vec_sw_orig[:,:,:half_window_size,:]), ent_dim, axis=-2)]), axis=-3)
		b = K.sum(Multiply()([wordsw_ent_indicators, K.repeat_elements(Reshape(target_shape=(input_dim, 1, half_window_size * wordvec_dim))(word_vec_sw_orig[:,:,half_window_size+1:,:]), ent_dim, axis=-2)]), axis=-3)
		local_context = K.repeat_elements(Reshape(target_shape=(1, ent_dim, 2 * half_window_size * wordvec_dim))(Concatenate()([a, b])), input_dim, axis=-3)
		# ent_vec_sws = Reshape(target_shape=(input_dim, window_size * wordvec_dim))(w2v_wrapper.get_embedding_layer(type='keras', name='LocalContext')(imath.slide_window(ent_ids, half_window_size=half_window_size, padding_val=last_widx)))
		argrole_vecs = Concatenate(name='ArgumentRoleEncoding')([dup_grus, K.repeat_elements(Reshape(target_shape=(1, ent_dim, 4 * gru_dim))(ent_grus), input_dim, axis=-3), K.repeat_elements(Reshape(target_shape=(input_dim, 1, window_size * wordvec_dim))(word_vec_sws), ent_dim, axis=-2), local_context])
		
		argrole_mlp = Dense(mlp_dim, activation='tanh', input_shape=(2 * gru_dim,), name='ArgumentRole-MLP-L1')(argrole_vecs)
		argrole_mlp = Dropout(drop_ratio, name='ArgumentRole-MLP-L1-Rgl')(argrole_mlp)
		argrole_mlp = Dense(mlp_dim, activation='tanh', input_shape=(mlp_dim,), name='ArgumentRole-MLP-L2')(argrole_mlp)
		argrole_mlp = Dropout(drop_ratio, name='ArgumentRole-MLP-L2-Rgl')(argrole_mlp)
		argrole_output = Dense(output_dim, activation='softmax', input_shape=(mlp_dim,), name='ArgumentRole')(argrole_mlp)
		
		is_other = K.repeat_elements(Reshape(target_shape=(input_dim, 1))(is_trigger), ent_dim, axis=-1)
		
		argrole_output = Multiply(name='ARGROLE')([is_other, argrole_output])
	
		model = Model(X_inputs, [trigger_output, argrole_output])
		optmzr = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		plot_model(model, show_shapes=True, to_file='model.png')
		model.compile(optimizer=optmzr, loss={'TRIGGER':'categorical_crossentropy', 'ARGROLE':'categorical_crossentropy'}, metrics=['acc', 'mse'])
	return model