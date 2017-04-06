#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: fzcmeans.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-01-23 12:52:12
###########################################################################
''' Latent Dirichlet Allocation Clustering Wrapper '''

import os

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils.validation import check_is_fitted


class LDACluster(LatentDirichletAllocation):
	def __init__(self, n_clusters=10, **kwargs):
		self.n_clusters = n_clusters
		super(LDACluster, self).__init__(n_topics=n_clusters, **kwargs)

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
		X[X < 0] = 0
		self.fit_u_ = self.u_ = super(LDACluster, self).fit_transform(X)
		self.fit_labels_ = self.labels_ = self.fit_u_.argmax(axis=1)
		if (fuzzy):
			return self.fit_u_
		return self.fit_labels_