#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: dstclc.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-12-06 15:09:32
###########################################################################
#

import os
import itertools

import numpy as np
import scipy as sp
import pandas as pd
# from scipy.spatial.distance import pdist
# from scipy.sparse.csgraph import shortest_path

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import pairwise_distances as pdist

from util import njobs


def parallel_pairwise(X, Y, func, n_jobs=1, symmetric=False, min_chunksize=1, **kwargs):
	from sklearn.metrics import pairwise
	from sklearn.utils import gen_even_slices
	# from sklearn.externals.joblib import Parallel
	# from sklearn.externals.joblib import delayed
	from sklearn.externals.joblib import cpu_count
	import parmap
	if n_jobs < 0:
		n_jobs = max(cpu_count() + 1 + n_jobs, 1)
	if Y is None:
		Y = X
	if n_jobs == 1:
		return func(X, Y, **kwargs)
	# if (hasattr(Y, 'shape')):
		# pairwise._parallel_pairwise(X, Y, func, n_jobs, **kwargs)
	slices = [slice(min_chunksize*s.start,min_chunksize*s.stop) for s in gen_even_slices(len(Y)/min_chunksize, n_jobs)]
	# ret = Parallel(n_jobs=n_jobs, verbose=5)(delayed(func)(X, Y[s], **kwargs) for s in slices)
	# ret = njobs.run_pool(func, n_jobs=n_jobs, dist_param=['Y'], X=X, Y=[Y[s] for s in slices], **kwargs)
	# ret = njobs.run_ipp(func, n_jobs=n_jobs, dist_param=['Y'], X=X, Y=[Y[s] for s in slices], **kwargs)
	ret = [r.T for r in parmap.map(func, [Y[s] for s in slices], X, processes=n_jobs)] # have not supported passing kwargs to func
	if (len(ret) == 0): return []
	return np.hstack(ret)


def normdist(D, range=None):
	flatten_D = D.ravel()
	minimum, maximum = flatten_D.min(), flatten_D.max()
	if (minimum == maximum):
		return np.ones_like(D)
	if (range is None):
		return (D - minimum) / (maximum - minimum)
	else:
		return (range[1] - range[0]) * (D - minimum) / (maximum - minimum) + range[0]


def sji_func(x, y):
	return np.sum((x-y)**2)
	
	
def sji(X, scaled=False):
	if (scaled):
		mms = MinMaxScaler()
		X = mms.fit_transform(np.nan_to_num(X))
	D = np.dot(D, D.T)
	return D
	
	
def z_dist(X):
	f = lambda x: np.exp(-x) if x < 0 else 1
	func = np.frompyfunc(f, 1, 1)
	cov = np.cov(X.T)
	D = np.ones((X.shape[0], X.shape[0]))
	for i, j in itertools.combinations(range(X.shape[0]), 2):
		delta = X[i] - X[j]
		D[i, j] = D[j, i] = np.abs(np.dot(np.dot(delta, func(10 * cov)), delta.T))
	return D
	
	
def cns_dist(X, C=None, metric='euclidean', a=0.5, n_jobs=1, **kwargs):
	D1 = pdist(X, metric=metric, n_jobs=n_jobs, **kwargs)
	if (C is None): return D1
	D2 = z_dist(C)
	return (1 - a) * normdist(D1) + a * normdist(D2)
	
	
def infer_pdist(D=None, metric='manhattan', transpose=True, n_jobs=1, **kwargs):
	def func(X, Y):
		PD = pdist(X, Y, metric=metric, n_jobs=1, **kwargs)
		if D is None: return PD
		if (metric == 'euclidean'):
			pass
		elif (metric == 'manhattan'):
			for i in xrange(PD.shape[0]):
				for j in [x for x in range(PD.shape[0]) if x != i]:
					PD[j] = D[i,j] * np.ones_like(PD[i]) - PD[i]
		return PD.T if transpose else PD
	return func
	
def gen_dstclc(func, **kwargs):
	return FunctionTransformer(func=func, **kwargs)