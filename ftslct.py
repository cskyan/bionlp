#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: ftslct.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-06-28 12:32:49
###########################################################################
#

import os

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

from util import io
	
	
def cooc_mt(X, Y):
	nX, nY = 1 - X, 1 - Y
	coocrnc, nxny_coocrnc = np.dot(Y.T, X), np.dot(nY.T, nX)
	nx_coocrnc, ny_coocrnc = np.dot(Y.T, nX), np.dot(nY.T, X)
	return coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc
	
	
def cooc_stat(X, Y, coocrnc):
	coocrnc_avg = coocrnc / Y.sum(axis=0).reshape((-1,1)).repeat(X.shape[1], axis=1)
	xy = X.reshape((X.shape[0], 1, X.shape[1])).repeat(Y.shape[1], axis=1) * Y.reshape((Y.shape[0], Y.shape[1], 1)).repeat(X.shape[1], axis=2)
	coocrnc_std = np.sqrt(np.square(xy - coocrnc_avg.reshape((1, coocrnc_avg.shape[0], coocrnc_avg.shape[1])).repeat(xy.shape[0], axis=0)).mean(axis=0))
	return coocrnc_avg, coocrnc_std
	
	
def gu_metric(X, Y, scaled=True):
	coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc = cooc_mt(X, Y)
	y_sum, ny_sum = coocrnc + nx_coocrnc, ny_coocrnc + nxny_coocrnc
	p = 1.0 * (ny_coocrnc + coocrnc) / (y_sum + ny_sum)
	z = 1.0 * (coocrnc - ny_coocrnc) / np.sqrt(p * (1 - p) * (1.0 / y_sum + 1.0 / ny_sum))
	gu = (np.fabs(z) * coocrnc * ny_sum / (y_sum * ny_coocrnc)).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(gu).reshape((-1,1))).ravel(), np.zeros_like(gu)
	else:
		return np.nan_to_num(gu), np.zeros_like(gu)

	
def fisher_crtrn(X, Y, scaled=True):
	coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc = cooc_mt(X, Y)
	coocrnc_avg, coocrnc_std = cooc_stat(X, Y, coocrnc)
	ny_coocrnc_avg, ny_coocrnc_std = cooc_stat(X, 1-Y, ny_coocrnc)
	fc = (np.square(coocrnc_avg - ny_coocrnc_avg) / (np.square(coocrnc_std) + np.square(ny_coocrnc_std))).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(fc).reshape((-1,1))).ravel(), np.zeros_like(fc)
	else:
		return np.nan_to_num(fc), np.zeros_like(fc)
	
	
def odds_ratio(X, Y, scaled=True):
	# nY = 1 - Y
	# y_freqs, ny_freqs = Y.sum(axis=0), nY.sum(axis=0)
	# cond_freqs, ncond_freqs = np.dot(Y.T, X), np.dot(nY.T, X)
	# cond_prob, ncond_prob = 1.0 * cond_freqs / y_freqs.reshape((-1,1)).repeat(cond_freqs.shape[1], axis=1), 1.0 * ncond_freqs / ny_freqs.reshape((-1,1)).repeat(ncond_freqs.shape[1], axis=1)
	# or_sum = ((cond_prob * (1 - ncond_prob)) / ((1 - cond_prob) * ncond_prob)).sum(axis=0)
	
	coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc = cooc_mt(X, Y)
	or_sum = (1.0 * (coocrnc * nxny_coocrnc) / (nx_coocrnc * ny_coocrnc)).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(or_sum).reshape((-1,1))).ravel(), np.zeros_like(or_sum)
	else:
		return np.nan_to_num(or_sum), np.zeros_like(or_sum)
	
	
def ngl_coef(X, Y, scaled=True):
	# nX, nY = 1 - X, 1 - Y
	# x_freqs, y_freqs = X.sum(axis=0), Y.sum(axis=0)
	# x_prob, y_prob = 1.0 * x_freqs / X.shape[0], 1.0 * y_freqs / Y.shape[0]
	# nx_prob, ny_prob = 1 - x_prob, 1 - y_prob
	# x_prob_mt, y_prob_mt = x_prob.reshape((1,-1)).repeat(y_prob.shape[0], axis=0), y_prob.reshape((-1,1)).repeat(x_prob.shape[0], axis=1)
	# nx_prob_mt, ny_prob_mt = nx_prob.reshape((1,-1)).repeat(ny_prob.shape[0], axis=0), ny_prob.reshape((-1,1)).repeat(nx_prob.shape[0], axis=1)
	# joint_prob, nxny_joint_prob = 1.0 * np.dot(Y.T, X) / X.shape[0], 1.0 * np.dot(nY.T, nX) / X.shape[0]
	# nx_joint_prob, ny_joint_prob = 1.0 * np.dot(Y.T, nX) / X.shape[0], 1.0 * np.dot(nY.T, X) / X.shape[0]
	# ngl = (X.shape[0] ** 0.5 * (joint_prob * nxny_joint_prob - nx_joint_prob * ny_joint_prob) / np.sqrt(x_prob_mt * y_prob_mt * nx_prob_mt * ny_prob_mt)).sum(axis=0)

	coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc = cooc_mt(X, Y)
	ngl = (X.shape[0] ** 0.5 * (coocrnc * nxny_coocrnc - nx_coocrnc * ny_coocrnc) / np.sqrt((coocrnc + nx_coocrnc) * (ny_coocrnc + nxny_coocrnc) * (coocrnc + ny_coocrnc) * (nx_coocrnc + nxny_coocrnc))).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(ngl).reshape((-1,1))).ravel(), np.zeros_like(ngl)
	else:
		return np.nan_to_num(ngl), np.zeros_like(ngl)
	
	
def gss_coef(X, Y, scaled=True):
	# nX, nY = 1 - X, 1 - Y
	# joint_prob, nxny_joint_prob = 1.0 * np.dot(Y.T, X) / X.shape[0], 1.0 * np.dot(nY.T, nX) / X.shape[0]
	# nx_joint_prob, ny_joint_prob = 1.0 * np.dot(Y.T, nX) / X.shape[0], 1.0 * np.dot(nY.T, X) / X.shape[0]
	# gss = (joint_prob * nxny_joint_prob - nx_joint_prob * ny_joint_prob).sum(axis=0)
	
	coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc = cooc_mt(X, Y)
	gss = (coocrnc * nxny_coocrnc - nx_coocrnc * ny_coocrnc).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(gss).reshape((-1,1))).ravel(), np.zeros_like(gss)
	else:
		return np.nan_to_num(gss), np.zeros_like(gss)

	
def info_gain(X, Y, scaled=True):
	# x_freqs, y_freqs = X.sum(axis=0), Y.sum(axis=0)
	# x_prob, y_prob = 1.0 * x_freqs / X.shape[0], 1.0 * y_freqs / Y.shape[0]
	# nx_prob = 1 - x_prob
	# x_prob_mt, y_prob_mt = x_prob.reshape((1,-1)).repeat(y_prob.shape[0], axis=0), y_prob.reshape((-1,1)).repeat(x_prob.shape[0], axis=1)
	# nx_prob_mt = nx_prob.reshape((1,-1)).repeat(y_prob.shape[0], axis=0)
	# joint_prob = 1.0 * np.dot(Y.T, X) / X.shape[0]
	# njoint_prob = 1 - joint_prob
	# ig = (joint_prob * np.log(joint_prob / (y_prob_mt * x_prob_mt)) + njoint_prob * np.log(njoint_prob / (y_prob_mt * nx_prob_mt))).sum(axis=0)
	
	coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc = cooc_mt(X, Y)
	x_sum, nx_sum, y_sum, ny_sum, xor_sum = coocrnc + ny_coocrnc, nx_coocrnc + nxny_coocrnc, coocrnc + nx_coocrnc, ny_coocrnc + nxny_coocrnc, nx_coocrnc + ny_coocrnc
	x_prob, y_prob = 1.0 * x_sum / X.shape[0], 1.0 * y_sum / X.shape[0]
	nx_prob, ny_prob = 1 - x_prob, 1.0 * ny_sum / X.shape[0]
	joint_prob, nxny_joint_prob = 1.0 * coocrnc / x_sum, 1.0 * nxny_coocrnc / xor_sum
	nx_joint_prob, ny_joint_prob = 1.0 * nx_coocrnc / xor_sum, 1.0 * ny_coocrnc / x_sum
	ig = (-1 * y_prob * np.log2(y_prob) + ny_prob * np.log2(ny_prob) - (x_prob * (-1 * joint_prob * np.log2(joint_prob) - ny_prob * np.log2(ny_prob))) + (nx_prob * (-1 * nx_joint_prob * np.log2(nx_joint_prob) - nxny_joint_prob * np.log2(nxny_joint_prob)))).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(ig).reshape((-1,1))).ravel(), np.zeros_like(ig)
	else:
		return np.nan_to_num(ig), np.zeros_like(ig)
	
	
def mutual_info(X, Y, scaled=True):
	# x_freqs, y_freqs = X.sum(axis=0), Y.sum(axis=0)
	# x_prob, y_prob = 1.0 * x_freqs / X.shape[0], 1.0 * y_freqs / Y.shape[0]
	# x_prob_mt, y_prob_mt = x_prob.reshape((1,-1)).repeat(y_prob.shape[0], axis=0), y_prob.reshape((-1,1)).repeat(x_prob.shape[0], axis=1)
	# joint_prob = 1.0 * np.dot(Y.T, X) / X.shape[0] # P(y_j, x_i)
	# mi = np.log(joint_prob / (y_prob_mt * x_prob_mt)).sum(axis=0)
	# return np.nan_to_num(mi), np.zeros_like(mi)
	
	coocrnc, nxny_coocrnc, nx_coocrnc, ny_coocrnc = cooc_mt(X, Y)
	mi = (1.0 * (coocrnc * X.shape[0]) / ((coocrnc + nx_coocrnc) * (coocrnc + ny_coocrnc))).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(mi).reshape((-1,1))).ravel(), np.zeros_like(mi)
	else:
		return np.nan_to_num(mi), np.zeros_like(mi)

	
def freqs(X, Y, min_t=1, max_t=None, scaled=True):
	if (max_t is None): max_t = X.shape[0]
	mt = sp.sparse.coo_matrix(X)
	mask_mt = np.zeros(mt.shape)
	mask_mt[mt.row, mt.col] = 1
	stat = mask_mt.sum(axis=0)
	return np.nan_to_num(stat), np.zeros_like(stat)
	stat[np.any([stat < min_t, stat > min_t], axis=0)] = 0
	median = np.median(stat[stat > 0])
	fi = (np.fabs(stat - median) + 0.5)**(-2)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(fi).reshape((-1,1))).ravel(), np.zeros_like(fi)
	else:
		return np.nan_to_num(fi), np.zeros_like(fi)
	
	
def decision_tree(X, Y, scaled=True):
	from sklearn.tree import DecisionTreeClassifier
	filter = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=0)
	filter.fit(X, Y)
	fi = filter.feature_importances_
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(fi).reshape((-1,1))).ravel(), np.zeros_like(fi)
	else:
		return np.nan_to_num(fi), np.zeros_like(fi)


def filtref(cur_f, ref_f, ori_f=None):
	cur_X, ref_X = io.read_df(cur_f, with_idx=True, sparse_fmt='csr'), io.read_df(ref_f, with_idx=True, sparse_fmt='csr')
	ori_X = io.read_df(ori_f, sparse_fmt='csr') if ori_f is not None else None
	fi_df = pd.DataFrame([[1] * cur_X.shape[1]], columns=cur_X.columns)
	if (ori_f is None):
		filtout = list(set(cur_X.columns) - set(ref_X.columns))
	else:
		filtout = list((set(ori_X.columns) - set(ref_X.columns)) & set(cur_X.columns))
	fi_df.loc[:,filtout] = 0
	fi = fi_df.values.ravel().tolist()
	def func(X, y):
		return fi, np.zeros_like(fi)
	return func
	
	
def gen_fis(X, Y, filtfunc=freqs, scaled=True, **kwargs):
	fi_list, pval_list = [[] for i in range(2)]
	if (len(Y.shape) == 1): Y = Y.reshape((-1, 1))
	for i in xrange(Y.shape[1]):
		y = Y[:,i].reshape((-1,1))
		fi, pval = filtfunc(X, y, scaled=scaled, **kwargs)
		fi_list.append(fi)
		pval_list.append(pval)
	return np.vstack(fi_list), np.vstack(pval_list)
	
	
def utopk(X, Y, filtfunc=freqs, fsw=1, fn=10, scaled=True, **kwargs):
	fi_mt, _ = gen_fis(X, Y, filtfunc, scaled=False, **kwargs)
	#reduce(np.union1d, fi_mt.argsort(axis=1)[:,-fn:][:,::-1])
	slct_idx1 = fi_mt.argsort(axis=1)[:,-fn:][:,::-1]
	slct_idx0 = np.arange(fi_mt.shape[0]).reshape((-1,1)).repeat(slct_idx1.shape[1], axis=1)
	fi_count = np.zeros_like(fi_mt)
	fi_count[slct_idx0.ravel(), slct_idx1.ravel()] = 1
	fi = (fi_mt * fi_count).sum(axis=0)
	if (scaled):
		mms = MinMaxScaler()
		return mms.fit_transform(np.nan_to_num(fi).reshape((-1,1))).ravel(), np.zeros_like(fi)
	else:
		return np.nan_to_num(fi), np.zeros_like(fi)
	
	
def gen_ftslct_func(func, **kwargs):
	def ftslct_func(X, Y):
		return func(X, Y, **kwargs)
	return ftslct_func
	
	
def lasso_path(spdr):
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	from sklearn import linear_model
	
	if (hasattr(opts, 'pid')):
		pid = opts.pid
	else:
		pid = 0
	print 'Process ID: %i' % pid
	Xs, Ys = hoc.get_mltl_npz([pid], mltlx=False)
	if (Ys == None):
		print 'Can not find the data file!'
		exit(1)
	X = Xs[0]
	y = Ys[0]
	
	print("Computing regularization path using the LARS ...")
	alphas, _, coefs = linear_model.lars_path(X.as_matrix().astype('float64'), y.as_matrix().astype('float64'), method='lasso', verbose=True)

	xx = np.sum(np.abs(coefs.T), axis=1)
	xx /= xx[-1]

	plt.plot(xx, coefs.T)
	ymin, ymax = plt.ylim()
	plt.vlines(xx, ymin, ymax, linestyle='dashed')
	plt.xlabel('|coef| / max|coef|')
	plt.ylabel('Coefficients')
	plt.title('LASSO Path')
	plt.axis('tight')
	plt.savefig('lasso_path')


class MSelectKBest(SelectKBest):
#	ff_kwargs = {}
	def __init__(self, score_func=freqs, k=10, **kwargs):
#		if len(kwargs)>0: MSelectKBest.ff_kwargs = kwargs
		super(MSelectKBest, self).__init__(score_func)
		self.k = k
		self.sf_kwargs = kwargs

	def fit(self, X, Y):
#		self.scores_, self.pvalues_ = self.score_func(X, Y, **MSelectKBest.ff_kwargs)
		self.scores_, self.pvalues_ = self.score_func(X, Y, **self.sf_kwargs)
		self.scores_, self.pvalues_ = np.asarray(self.scores_), np.asarray(self.pvalues_)
		return self
		
		
class MSelectOverValue(SelectKBest):
	def __init__(self, score_func=freqs, threshold=0, **kwargs):
		super(MSelectOverValue, self).__init__(score_func)
		self.threshold = threshold
		self.sf_kwargs = kwargs

	def fit(self, X, Y):
		self.scores_, self.pvalues_ = self.score_func(X, Y, **self.sf_kwargs)
		self.scores_, self.pvalues_ = np.asarray(self.scores_), np.asarray(self.pvalues_)
		self.k = len(self.scores_[self.scores_ > self.threshold])
		return self


def main():
	pass


if __name__ == '__main__':
	main()