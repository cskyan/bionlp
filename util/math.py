#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: math.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-29 21:48:19
###########################################################################
#

import os
from itertools import chain, combinations, zip_longest

import numpy as np
from scipy import sparse

from . import io, func


ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(int(n/10)%10!=1)*(n%10<4)*n%10::4])


def exmp_accuracy(y_true, y_pred):
	y_true, y_pred = y_true.astype('bool').astype('int8'), y_pred.astype('bool').astype('int8')
    # return (((~np.bitwise_xor(y_true, y_pred).astype('bool')).astype('int8') * y_pred).sum(axis=1) / (sys.float_info.epsilon + np.bitwise_or(y_true, y_pred).astype('bool').astype('int8').sum(axis=1))).mean()
	return ((y_true * y_pred).sum(axis=1) / (np.finfo(float).eps + (y_true + y_pred).astype('bool').astype('int8').sum(axis=1))).mean()

def exmp_precision(y_true, y_pred):
	y_true, y_pred = y_true.astype('bool').astype('int8'), y_pred.astype('bool').astype('int8')
    # return (((~np.bitwise_xor(y_true, y_pred).astype('bool')).astype('int8') * y_pred).sum(axis=1) / (sys.float_info.epsilon + y_pred.sum(axis=1))).mean()
	return ((y_true * y_pred).sum(axis=1) / (np.finfo(float).eps + y_pred.sum(axis=1))).mean()

def exmp_recall(y_true, y_pred):
	y_true, y_pred = y_true.astype('bool').astype('int8'), y_pred.astype('bool').astype('int8')
    # return (((~np.bitwise_xor(y_true, y_pred).astype('bool')).astype('int8') * y_true).sum(axis=1) / (sys.float_info.epsilon + y_true.sum(axis=1))).mean()
	return ((y_true * y_pred).sum(axis=1) / (np.finfo(float).eps + y_true.sum(axis=1))).mean()

def exmp_fscore(y_true, y_pred, beta=1):
	precision = exmp_precision(y_true, y_pred)
	recall = exmp_recall(y_true, y_pred)
	beta_pw2 = beta**2
	return (1 + beta_pw2)*(precision * recall)/(beta_pw2 * precision + recall)

def normalize(a, ord=1):
    norm=np.linalg.norm(a, ord=ord)
    if norm==0: norm=np.finfo(a.dtype).eps
    return a/norm

def topk(D, k=10, descend=False):
    I = np.argsort(D, axis=1)[:,-1:D.shape[1]-k-1:-1] if descend else np.argsort(D, axis=1)[:,:k]
    return D[np.arange(D.shape[0])[:,None],I], I

class VectorDB():
	def __init__(self, metric='euclidean', feature_weights=None):
		self.metric = metric
		if feature_weights is not None and len(feature_weights) > 0:
			if type(feature_weights) is list:
				# self.feature_weights = np.array([])
				# for ftw in feature_weights:
				# 	indices, ws = zip(*ftw.values())
				# 	indices, ws = np.hstack(indices), np.hstack([np.array([w]*len(idx)) for idx, w in zip(indices, ws)])
				# 	weights = np.zeros(indices.max()+1)
				# 	weights[indices] = ws
				# 	self.feature_weights = np.hstack(self.feature_weights, weights)
				self.feature_weights = {}
				offset = 0
				for ftw in feature_weights:
					partial_max = 0
					for k, v in ftw.items():
						self.feature_weights[k] = (np.hstack([self.feature_weights[k], v[0] + offset]) if k in self.feature_weights else v[0] + offset, v[1])
						partial_max = max(partial_max, v[0].max())
					offset += partial_max
			elif type(feature_weights) is dict:
				# indices, ws = zip(*feature_weights.values())
				# indices, ws = np.hstack(indices), np.hstack([np.array([w]*len(idx)) for idx, w in zip(indices, ws)])
				# self.feature_weights = np.zeros(indices.max()+1)
				# self.feature_weights[indices] = ws
				self.feature_weights = feature_weights
			else:
				self.feature_weights = None
		else:
			self.feature_weights = None

	def add(self, Y):
		self.Y = Y

	def search(self, X, k, n_jobs=None):
		from sklearn.metrics import pairwise_distances as pdist
		if self.feature_weights is not None:
			D = np.zeros((X.shape[0], self.Y.shape[0]))
			for k, v in self.feature_weights.items():
				D = np.add(D, v[1] * np.nan_to_num(pdist(X[:,v[0]], self.Y[:,v[0]], metric=self.metric, n_jobs=n_jobs)))
		else:
			D = pdist(X, self.Y, metric=self.metric, n_jobs=n_jobs)
		return topk(D, k)

def sort_coo_row(coo_row):
	tuples = zip(coo_row.col, coo_row.data)
	return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def sort_csr_row(csr_row):
	tuples = zip(csr_row.indices, csr_row.data)
	return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def sort_coo_rows(coo_mt):
	return [sort_csr_row(coo_mt.getrow(i)) for i in range(coo_mt.shape[0])]


def subset(iterable, min_crdnl=0):
	s = list(iterable) if type(iterable) != list else iterable
	return chain.from_iterable(combinations(s, x) for x in range(min_crdnl, len(s)+1))


def grouper(n, iterable, fillvalue=None):
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)


def array2index(array, ret_base=False):
	dim_array = list(range(len(array.shape)))
	dim_array_set = set(dim_array)
	e = np.ones(len(array.shape), dtype='int')
	index_base = []
	for i in dim_array:
		dim_idx = np.copy(e)
		dim_idx[i] = array.shape[i]
		idx_mt = np.arange(array.shape[i]).reshape(dim_idx)
		for j in dim_array_set - set([i]):
			idx_mt = np.repeat(idx_mt, array.shape[i], j)
		index_base.append(idx_mt)
	return index_base if ret_base else np.array(np.core.records.fromarrays(index_base, names=','.join(['dim_'+str(x) for x in range(len(index_base))])))


def slice_last_axis(array, slice_array):
	index_base = array2index(slice_array, ret_base=True)
	idx = [x.flatten() for x in index_base] + [slice_array.flatten()]
	return array[idx].reshape(array.shape[:-1])


def slide_window(X, half_window_size=1, padding_val=0):
    if (type(X) == np.ndarray):
        X = X.tolist()
    if (all([type(x) is not list or len(x) == 1 for x in X])):
        return [[padding_val] * max(0, half_window_size - i) + X[max(0, i - half_window_size):min(len(X), i + half_window_size + 1)] + [padding_val] * (half_window_size - min(half_window_size, len(X) - 1 - i)) for i in range(len(X))]
    else:
        return [[[padding_val] * max(0, half_window_size - j) + X[i][max(0, j - half_window_size):min(len(X[i]), j + half_window_size + 1)] + [padding_val] * (half_window_size - min(half_window_size, len(X[i]) - 1 - j)) for j in range(len(X[i]))] for i in range(len(X))]

def softmax(w, t=1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


def mlb_clsw(Y, norm=False):
    Y = np.array(Y).reshape((len(Y),-1))
    clsw = (1 - Y).mean(axis=0) if norm else 1.0 / (Y.mean(axis=0) * Y.shape[1])
    return clsw


def mlb_class_weight(class_weight, Y):
	from sklearn.utils.class_weight import compute_class_weight
	Y = np.array(Y)
	if not (len(Y.shape) > 1 and Y.shape[1] > 1):
		return compute_class_weight(class_weight, np.unique(Y), func.flatten_list(Y.tolist()))
	return np.array([compute_class_weight(class_weight, np.array([0,1]), Y[:,i]) for i in range(Y.shape[1])])


class PearsonR(object):
	def __init__(self, X, dtype='float32', zero_value=0, sparse_output=False, cache_path=None):
		if (type(X) is not int):
			X = X.astype(dtype)
			self.m = X.shape[0]
			self.scale = self.m - 1
			self.X = (X - np.mean(X, axis=1, dtype=dtype)[:, None])
			self.var = ((self.X**2.).sum(axis=1, dtype=dtype) / self.scale).astype(dtype)
		else:
			self.m = X
			self.scale = self.m - 1
			self.X = None
			self.var = None
		self._corrcoef = None
		self.dtype = dtype
		self.zero_value = zero_value
		self.sparse_output = sparse_output
		self.cache_path = cache_path

	def corrcoef_k(self, k):
		c = (np.dot(self.X, self.X[k]) / self.scale).astype(self.dtype)
		return c / (self.var[k] * self.var)**.5

	def gen_corrcoef(self):
		use_cache = self.cache_path is not None and os.path.exists(self.cache_path)
		for k in range(self.m):
			if (use_cache):
				cache_f = os.path.join(self.cache_path, 'corrcoef_%i.npz' % k)
				if (os.path.exists(cache_f)):
					try:
						coef = io.read_spmt(cache_f, sparse_fmt='csr')
					except Exception as e:
						print('Cannot read cache %s, recalculating it again...' % cache_f)
					else:
						if (self.sparse_output):
							yield coef
						else:
							yield coef.todense()
						continue
			coef = self.corrcoef_k(k)
			coef[(coef <= self.zero_value) & (coef >= -self.zero_value)] = 0
			if (self.sparse_output):
				sp_coef = sparse.csr_matrix(coef)
				del coef
				coef = sp_coef
			if (use_cache):
				io.write_spmt(coef, cache_f, sparse_fmt='csr', compress=True)
			yield coef

	@property
	def corrcoef(self):
		if (self._corrcoef is None):
			if (self.sparse_output):
				self._corrcoef = sparse.vstack([c for c in self.gen_corrcoef()], format='csr')
			else:
				self._corrcoef = np.array([c for c in self.gen_corrcoef()])
		return self._corrcoef


class SymNDArray(np.ndarray):
	def __setitem__(self, indices, value):
		(i, j) = indices
		super(SymNDArray, self).__setitem__((i, j), value)
		super(SymNDArray, self).__setitem__((j, i), value)


def symmetrize(a):
	return np.array(a + a.T - np.diag(a.diagonal()), dtype=a.dtype)


def symarray(input_array, combine_data=True):
	"""
	Returns a symmetrized version of the array-like input_array.
	Further assignments to the array are automatically symmetrized.
	"""
	if (combine_data):
		return symmetrize(np.asarray(input_array)).view(SymNDArray)
	else:
		return np.asarray(input_array).view(SymNDArray)
