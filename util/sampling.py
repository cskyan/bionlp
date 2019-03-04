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

from sklearn.model_selection import StratifiedShuffleSplit


def samp_df(df, n=None, frac=None, key=None, filt_func=lambda x: x, reset_idx=False):
	if (n is None == frac is None): return df
	if (frac is None):
		if (key is None):
			new_df = filt_func(df)
			new_df = new_df.sample(n=min(n, new_df.shape[0]))
		else:
			new_df = filt_func(df).groupby(key).apply(lambda x: x.sample(n=min(n, x.shape[0])))
	else:
		if (key is None):
			new_df = filt_func(df).sample(frac=frac)
		else:
			new_df = filt_func(df).groupby(key).apply(lambda x: x.sample(frac=frac))
	return new_df.reset_index(drop=True) if reset_idx else new_df


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