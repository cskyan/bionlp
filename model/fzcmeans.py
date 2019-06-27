#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: fzcmeans.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-01-16 15:56:35
###########################################################################
''' Fuzzy C-means Clustering Wrapper '''

import os

from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted
from skfuzzy.cluster import cmeans, cmeans_predict
from skfuzzy.cluster import _cmeans
from .. import dstclc


class FZCMeans(KMeans):
	def __init__(self, n_clusters=8, m=2, max_iter=300, error=0.005, init=None, random_state=None):
		self.n_clusters = n_clusters
		self.m = m
		self.max_iter = max_iter
		self.error = error
		self.init = init
		self.random_state = random_state

	def fit(self, X, y=None):
		'''Compute Fuzzy c-means clustering.
		Parameters
		----------
		X : array-like or sparse matrix, shape = [n_samples, n_features]
			Training instances to cluster.
		'''
		X = self._check_fit_data(X)
		self.cluster_centers_, self.u_, self.u0_, self.distmt_, self.jm_, self.n_iter_, self.fpc_ = \
			cmeans(data=X.T, c=self.n_clusters, m=self.m, maxiter=self.max_iter, error=self.error, init=self.init, seed=self.random_state)
		print('fuzzy partition coefficient: %0.3f' % self.fpc_)
		self.fit_u_, self.fit_u0_, self.fit_distmt_, self.fit_jm_, self.fit_n_iter_, self.fit_fpc_ = self.u_.T, self.u0_.T, self.distmt_.T, self.jm_, self.n_iter_, self.fpc_
		self.fit_labels_ = self.labels_ = self.fit_u_.argmax(axis=1)
		return self

	def predict(self, X, fuzzy=False):
		'''Predict the closest cluster each sample in X belongs to.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to predict.
		Returns
		-------
		u : array, shape [n_samples, n_clusters] or [n_samples,]
			Predicted fuzzy c-partitioned matrix or most likely cluster labels.
		'''
		check_is_fitted(self, 'cluster_centers_')
		X = self._check_test_data(X)
		self.u_, self.u0_, self.distmt_, self.jm_, self.n_iter_, self.fpc_ = \
			cmeans_predict(X.T, self.cluster_centers_, m=self.m, maxiter=self.max_iter, error=self.error, init=self.init, seed=self.random_state)
		self.pred_u_, self.pred_u0_, self.pred_distmt_, self.pred_jm_, self.pred_n_iter_, self.pred_fpc_ = self.u_.T, self.u0_.T, self.distmt_.T, self.jm_, self.n_iter_, self.fpc_
		self.pred_labels_ = self.labels_ = self.pred_u_.argmax(axis=1)
		if (fuzzy): return self.pred_u_
		return self.pred_labels_

	def fit_predict(self, X, y=None, fuzzy=False):
		'''Compute cluster centers and predict cluster index for each sample.
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to predict.
		Returns
		-------
		u : array, shape [n_samples, n_clusters] or [n_samples,]
			Predicted fuzzy c-partitioned matrix or most likely cluster labels.
		'''
		self.fit(X)
		if (fuzzy): return self.fit_u_
		return self.fit_labels_

	def transform(self, X, y=None, fuzzy=False):
		'''Transform X to a cluster-distance space.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to transform.
		Returns
		-------
		X_new : array, shape [n_samples, k]
			X transformed in the new space.
		'''
		check_is_fitted(self, 'cluster_centers_')
		X = self._check_test_data(X)
		if (fuzzy): return super(FZCMeans, self)._transform(X)
		return super(FZCMeans, self)._transform(X) * self.predict(X, fuzzy=True)**self.m

	def _transform(self, X):
		'''guts of transform method; no input validation'''
		return super(FZCMeans, self)._transform(X) * self.fit_u_**self.m


class CNSFZCMeans(FZCMeans):
	def __init__(self, n_clusters=8, metric='manhattan', m=2, a=0.5, max_iter=300, error=0.005, init=None, random_state=None, n_jobs=1):
		self.n_clusters = n_clusters
		self.metric = metric
		self.m = m
		self.a = a
		self.max_iter = max_iter
		self.error = error
		self.init = init
		self.random_state = random_state
		self.n_jobs = n_jobs


	def fit(self, X, y=None, constraint=None):
		cns_distance = dstclc.cns_dist(X, C=constraint, metric=self.metric, a=self.a, n_jobs=self.n_jobs)
		_cmeans._distance = dstclc.infer_pdist(D=cns_distance, metric=self.metric, transpose=True, n_jobs=self.n_jobs)
		return super(CNSFZCMeans, self).fit(X, y)


	def fit_predict(self, X, y=None, fuzzy=False, constraint=None):
		self.fit(X, y=y, constraint=constraint)
		if (fuzzy): return self.fit_u_
		return self.fit_labels_
