#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: sampling.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-04-20 15:17:48
###########################################################################
#

from sklearn.cross_validation import StratifiedShuffleSplit


def samp_df_iter(X_iter, iter_size, y, size=0.5):
	'''
	Stratified Sampling
	'''
	if (size >= 1.0 or size <= 0 or y[y==0].shape[0] < 2 or y[y==1].shape[0] < 2):
		return X, y
	sss = StratifiedShuffleSplit(y, 1, train_size=size)
	samp_idx = [idx[0] for idx in sss][0]
	spidx_set = set(samp_idx)
	iter_offset = 0
	slct_X = []
	for sub_X in X_iter():
		idx_set = (sub_X.index + iter_offset)
		slct_idx = idx_set & spidx_set
		slct_X.append(sub_X.iloc[list(slct_idx - iter_offset),:])
		iter_offset += iter_size
		del sub_X
	new_X = pd.concat(slct_X, axis=0, copy=True)
	for sx in slct_X:
		del sx
	new_y = y.iloc[samp_idx]
	return new_X, new_y