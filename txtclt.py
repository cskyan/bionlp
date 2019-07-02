#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: txtclt.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-11-30 16:24:13
###########################################################################
#

import difflib, itertools
from time import time

import numpy as np
import scipy as sp
from scipy import stats, linalg
from scipy.misc import comb
import pandas as pd

from sklearn.base import clone, BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from keras.utils.io_utils import HDF5Matrix

from .util import io, func, plot
from .util import math as imath

common_cfg = {}
METRIC_NAME = {'homogeneity':'Homogeneity', 'completeness':'Completeness', 'v_measure':'V-measure', 'f_measure':'F-measure', 'purity':'Purity', 'ri':'Rand Index', 'ari':'Adjusted Rand Index', 'mi':'Mutual Information', 'ami':'Adjusted Mutual Information', 'nmi':'Normalized Mutual Information', 'fmi':'Fowlkes-Mallows Index', 'slcoef':'Silhouette Coefficient', 'chscore':'Calinski and Harabaz Score', 'time':'Elapsed Time'}


def init(plot_cfg={}, plot_common={}):
	if (len(plot_cfg) > 0 and plot_cfg['MON'] is not None):
		plot.MON = plot_cfg['MON']
	global common_cfg
	if (len(plot_common) > 0):
		common_cfg = plot_common


# Fuzzy Clustering Metrics
def fuzzy_metrics(labels_true, labels_pred, X, r=0, beta=1, simple=True, use_gpu=False):
	if (simple):
		H = np.dot(labels_true.T, labels_pred)
	else:
		r = max(0.0, 1.0 / labels_pred.shape[1])
		H = np.zeros((labels_true.shape[1], labels_pred.shape[1]), dtype='float16')
		for c, k in itertools.product(range(labels_true.shape[1]), range(labels_pred.shape[1])):
			a_c_v, a_k_v = labels_true[:,c], labels_pred[:,k]
			a_c, a_k = a_c_v[a_c_v > 0], a_k_v[a_k_v > r]
			X_c, X_k = X[a_c_v > 0].T, X[a_k_v > r].T
			PX_c = np.dot(np.dot(X_c, linalg.pinv(np.dot(X_c.T, X_c))), X_c.T)
			PX_k = np.dot(np.dot(X_k, linalg.pinv(np.dot(X_k.T, X_k))), X_k.T)

			# eig_vals, eig_vecs = linalg.eig(PX_c + PX_k)
			## Using Scipy Sparse Begin ##
			from scipy import sparse
			from scipy.sparse.linalg import eigs
			sp_PX = sparse.csr_matrix(PX_c + PX_k)
			eig_vals, eig_vecs = eigs(sp_PX)
			## Using Scipy Sparse End ##

			f = eig_vecs[:,eig_vals.argmax()]
			H[c,k] = np.square(f - np.dot(X_c, a_c)).sum() + np.square(f - np.dot(X_k, a_k)).sum()
	H_CK = -1 / X.shape[0] * (H * np.log(H / (H.sum(axis=0).reshape((1,-1)).repeat(labels_true.shape[1], axis=0)))).sum()
	H_sumk = H.sum(axis=1)
	homogeneity = 1 if (H_CK == 0) else 1 - H_CK / -(H_sumk / labels_true.shape[1] * np.log(H_sumk / labels_true.shape[1])).sum()
	H_KC = -1 / X.shape[0] * (H * np.log(H / (H.sum(axis=1).reshape((-1,1)).repeat(labels_pred.shape[1], axis=1)))).sum()
	H_sumc = H.sum(axis=0)
	completeness = 1 if (H_KC == 0) else 1 - H_KC / -(H_sumc / labels_true.shape[1] * np.log(H_sumc / labels_true.shape[1])).sum()
	vmeasure = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
	purity = H.max(axis=0).mean()

	comb_func = np.frompyfunc(lambda x: comb(x, 2), 1, 1)
	b_combsum, a_combsum = comb_func(H.sum(axis=0)).sum(), comb_func(H.sum(axis=1)).sum()
	expected_index = a_combsum * b_combsum / comb(labels_true.shape[0], 2)
	adjusted_rand_index = (comb_func(H).sum() - expected_index) / ((a_combsum + b_combsum) / 2 - expected_index)

	marginals = np.outer(labels_true.sum(axis=0), labels_pred.sum(axis=0))
	mi = stats.entropy(H.reshape((-1,)), marginals.reshape((-1,))).sum()

	S, A, B, C = [np.zeros((labels_true.shape[0], labels_true.shape[0]), dtype='float16') for x in range(4)]
	for i, j in itertools.combinations(range(labels_true.shape[0]), 2):
		kld_true = stats.entropy(labels_true[i], labels_true[j])
		kld_pred = stats.entropy(labels_pred[i], labels_pred[j])
		S[i, j] = 4 * abs((kld_true - 0.5) * (kld_pred - 0.5))
		A[i, j] = kld_true * kld_pred
		B[i, j] = (1 - kld_true) * kld_pred
		C[i, j] = kld_true * (1 - kld_pred)
	rand_index = S.sum() / (labels_true.shape[0]**2 - labels_true.shape[0])
	fmeasure = (2 * A / (2 * A + B + C)).sum()

	return homogeneity, completeness, vmeasure, fmeasure, purity, rand_index, adjusted_rand_index, mi


# Benchmark
def benchmark(pipeline, X, Y, metric='euclidean', is_fuzzy=False, is_nn=False, constraint=None, use_gpu=False):
	print('+' * 80)
	print('Fitting Model: ')
	print(pipeline)
	t0 = time()
	if (is_nn):
		pipeline.fit(X, Y, constraint=constraint)
		pred = pipeline.predict(X, constraint=constraint, proba=True)
	else:
		# Extract extra parameters
		model_param = dict(is_fuzzy=is_fuzzy, constraint=constraint)
		kwargs = dict([(kp[0], model_param[kp[1]]) for kp in [('clt__fuzzy', 'is_fuzzy'), ('clt__constraint', 'constraint')] if kp[1] in model_param and (isinstance(model_param[kp[1]], bool) and model_param[kp[1]] or not isinstance(model_param[kp[1]], bool) and model_param[kp[1]] is not None)])
		try:
			pred = pipeline.fit_predict(X, **kwargs)
		except TypeError as e:
			print(e)
			pred = pipeline.fit_predict(X)
	elapsed_time = time() - t0
	print('elapsed time: %0.3fs' % elapsed_time)

	# Number of clusters, ignoring noise if present
	if (is_nn):
		Y, pred_lb = Y.argmax(axis=1).reshape((-1,)), pred.argmax(axis=1).reshape((-1,))
		pred_lb[pred.sum(axis=1) == 0] == -1
		pred = pred_lb
	is_fuzzy = True if (len(pred.shape) > 1 and pred.shape[1] > 1) else False
	if (is_fuzzy):
		n_clusters = pred.shape[1]
		# lbz = LabelBinarizer()
		# Y = lbz.fit_transform(Y)
	else:
		n_clusters = len(set(func.flatten_list(pred))) - (1 if -1 in pred else 0)
	print('estimated number of clusters: %d' % n_clusters)
	if (is_fuzzy or (len(pred.shape) > 1 and pred.shape[1] > 1 and any(pred.sum(axis=1) > 1))):
		homogeneity, completeness, v_measure, f_measure, purity, ri, ari, mi = fuzzy_metrics(Y, pred, X, use_gpu=use_gpu)
	else:
		# Convert the binary labels into numerical labels
		if (len(pred.shape) > 1 and pred.shape[1] > 1):
			new_pred = np.zeros(pred.shape[0], dtype='int8')
			idx = np.where(pred==1)
			new_pred[idx[0]] = idx[1] + 1
			pred = new_pred
		if (len(Y.shape) > 1 and Y.shape[1] > 1):
			new_Y = np.zeros(Y.shape[0], dtype='int8')
			idx = np.where(Y==1)
			new_Y[idx[0]] = idx[1] + 1
			Y = new_Y
		homogeneity, completeness, v_measure, ari = metrics.homogeneity_score(Y, pred), metrics.completeness_score(Y, pred), metrics.v_measure_score(Y, pred), metrics.adjusted_rand_score(Y, pred)
	print("homogeneity: %0.3f" % homogeneity)
	print("completeness: %0.3f" % completeness)
	print("v-measure: %0.3f" % v_measure)
	print("adjusted rand index: %0.3f" % ari)
	if (is_fuzzy):
		print("purity: %0.3f" % purity)
		print("f-measure: %0.3f" % f_measure)
		print("rand index: %0.3f" % ri)
		print("mutual information: %0.3f" % mi)
		print('+' * 80 + '\n')
		return {'homogeneity':homogeneity, 'completeness':completeness, 'v_measure':v_measure, 'f_measure':f_measure, 'purity':purity, 'ri':ri, 'ari':ari, 'mi':mi, 'time':elapsed_time, 'pred_lb':pred}
	fmi = metrics.fowlkes_mallows_score(Y, pred)
	print("Fowlkes-Mallows index: %0.3f" % fmi)
	ami = metrics.adjusted_mutual_info_score(Y, pred)
	print("adjusted mutual information: %0.3f" % ami)
	nmi = metrics.normalized_mutual_info_score(Y, pred)
	print("normalized mutual information: %0.3f" % nmi)
	slcoef = metrics.silhouette_score(X, pred, metric=metric) if n_clusters > 1 else 0
	print("silhouette coefficient: %0.3f" % slcoef)
	# chscore = metrics.calinski_harabaz_score(X, pred)
	# print("Calinski-Harabaz score: %0.3f" % chscore)
	print('+' * 80)
	print('\n')

	return {'homogeneity':homogeneity, 'completeness':completeness, 'v_measure':v_measure, 'fmi':fmi, 'ari':ari, 'ami':ami, 'nmi':nmi, 'slcoef':slcoef, 'chscore':0, 'time':elapsed_time, 'pred_lb':pred}


# Calculate the venn digram overlaps
def pred_ovl(preds, pred_true=None, axis=1):
	if (axis == 0):
		preds = preds.T
	if (pred_true is not None):
		pred_true = pred_true.reshape((-1,))
	# Row represents feature, column represents instance
	var_num, dim = preds.shape[0], preds.shape[1]
	orig_idx = np.arange(var_num)

	if (len(preds.shape) < 2 or preds.shape[1] == 1):
		if (pred_true is None):
			return np.ones(shape=(1,), dtype='int')
		else:
			overlap_mt = np.ones(shape=(1,2), dtype='int')
			overlap_mt[0,1] = orig_idx[preds.reshape((-1,)) == pred_true].shape[0]
			return overlap_mt

	# Calculate possible subsets of all the instance indices
	subset_idx = list(imath.subset(list(range(dim)), min_crdnl=1))
	# Initialize result matrix
	if (pred_true is None):
		overlap_mt = np.zeros(shape=(len(subset_idx),), dtype='int')
	else:
		overlap_mt = np.zeros(shape=(len(subset_idx), 2), dtype='int')
	# Calculate overlap for each subset
	for i, idx in enumerate(subset_idx):
		rmn_idx = set(range(dim)) - set(idx)
		# Select the positions of the target instance that without any overlap with other instances
		pred_sum, chsn_sum, rmn_sum = preds.sum(axis=1), preds[:,idx].sum(axis=1), preds[:,list(rmn_idx)].sum(axis=1)
		condition = np.all([np.logical_or(chsn_sum == 0, chsn_sum == len(idx)), np.logical_or(rmn_sum == 0, rmn_sum == len(rmn_idx)), np.logical_or(pred_sum == len(idx), pred_sum == len(rmn_idx))], axis=0)
		if (pred_true is None):
			overlap_mt[i] = orig_idx[condition].shape[0]
		else:
			# And the selected positions should be true
			true_cond = np.logical_and(condition, preds[:,idx[0]] == pred_true)
			overlap_mt[i,0] = orig_idx[condition].shape[0]
			overlap_mt[i,1] = orig_idx[true_cond].shape[0]
	return overlap_mt


# Clustering
def clustering(X, model_iter, model_param={}, cfg_param={}, global_param={}, lbid=''):
	global common_cfg
	FILT_NAMES, CLT_NAMES, PL_NAMES, PL_SET = model_param['glb_filtnames'], model_param['glb_cltnames'], global_param['pl_names'], global_param['pl_set']
	lbidstr = ('_' + (str(lbid) if lbid != -1 else 'all')) if lbid is not None and lbid != '' else lbid

	# Format the data
	if (type(X) != pd.DataFrame):
		X = pd.DataFrame(X) if type(X) != HDF5Matrix else X

	print('Clustering is starting...')
	preds = []
	for vars in model_iter(**model_param):
		if (global_param['comb']):
			mdl_name, mdl = [vars[x] for x in range(2)]
		else:
			clt_name, clt = [vars[x] for x in range(2)]
			# filt_name, filter, clt_name, clt= [vars[x] for x in range(4)]
		print('#' * 80)
		# Assemble a pipeline
		if ('filter' in locals() and filter != None):
			model_name = '%s [Ft Filt] & %s [CLT]' % (filt_name, clt_name)
			pipeline = Pipeline([('featfilt', clone(filter)), ('clt', clt)])
		elif ('clt' in locals() and clt != None):
			model_name = '%s [CLT]' % clt_name
			pipeline = Pipeline([('clt', clt)])
		else:
			model_name = mdl_name
			pipeline = mdl
		if (model_name in PL_SET): continue
		PL_NAMES.append(model_name)
		PL_SET.add(model_name)
		print(model_name)
		# Extract extra parameters
		kwargs = dict([(kp[0], model_param[kp[1]]) for kp in [('clt__fuzzy', 'is_fuzzy'), ('clt__constraint', 'constraint')] if kp[1] in model_param and ((type(model_param[kp[1]]) is not bool and model_param[kp[1]] is not None) or model_param[kp[1]])])
		# Build the model
		print('+' * 80)
		print('Fitting Model: ')
		print(pipeline)
		t0 = time()

		pred = pipeline.fit_predict(X, **kwargs)
		elapsed_time = time() - t0
		print('elapsed time: %0.3fs' % elapsed_time)
		preds.append(pred)
		if (cfg_param.setdefault('save_model', True)):
			io.write_obj(pipeline, 'clt_%s%s.mdl' % (model_name.replace(' ', '_').lower(), lbidstr))
		if (cfg_param.setdefault('save_pred', True)):
			io.write_npz(dict(pred_lb=pred), 'clt_pred_%s%s' % (model_name.replace(' ', '_').lower(), lbidstr))

	# Prediction overlap
	if (model_param.setdefault('is_fuzzy', False) or (len(pred.shape) > 1 and pred.shape[1] > 1)):
		preds = [pred.argmax(axis=1) for pred in preds]
	preds_mt = np.column_stack(preds)
	povl = np.array(pred_ovl(preds_mt))
	# Spearman's rank correlation
	spmnr, spmnr_pval = stats.spearmanr(preds_mt)
	# Kendall rank correlation
#	kendalltau = stats.kendalltau(preds_mt)[0]
	# Pearson correlation
#	pearson = tats.pearsonr(preds_mt)[0]

	## Save performance data
	povl_idx = [' & '.join(x) for x in imath.subset(PL_NAMES, min_crdnl=1)]
	povl_df = pd.DataFrame(povl, index=povl_idx, columns=['pred_ovl'])
	spmnr_df = pd.DataFrame(spmnr, index=PL_NAMES, columns=PL_NAMES)
	spmnr_pval_df = pd.DataFrame(spmnr_pval, index=PL_NAMES, columns=PL_NAMES)
	if (cfg_param.setdefault('save_povl', False)):
		povl_df.to_excel('cpovl_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_povl_npz', False)):
		io.write_df(povl_df, 'povl_clt%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr', False)):
		spmnr_df.to_excel('spmnr_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_npz', False)):
		io.write_df(spmnr_df, 'spmnr_clt%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_pval', False)):
		spmnr_pval_df.to_excel('spmnr_pval_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_pval_npz', False)):
		io.write_df(spmnr_pval_df, 'spmnr_pval_clt%s.npz' % lbidstr, with_idx=True)

	return preds


# Cross validation
def cross_validate(X, Y, model_iter, model_param={}, kfold=5, cfg_param={}, split_param={}, global_param={}, lbid=''):
	global common_cfg
	FILT_NAMES, CLT_NAMES, PL_NAMES, PL_SET = model_param['glb_filtnames'], model_param['glb_cltnames'], global_param['pl_names'], global_param['pl_set']
	lbidstr = ('_' + str(lbid) if lbid != -1 else 'all') if lbid is not None and lbid != '' else lbid

	# Format the data
	if (type(X) != pd.DataFrame):
		X = pd.DataFrame(X)
	if (type(Y) == pd.DataFrame):
		Y = Y.as_matrix()
	if (len(Y.shape) == 1 or Y.shape[1] == 1):
		Y = Y.reshape((Y.shape[0],))
	is_fuzzy = True if (len(Y.shape) > 1 and Y.shape[1] > 1) else False

	print('Benchmark is starting...')
	if (kfold == 1):
		kf = [(np.arange(X.shape[0]), np.array([]))]
	elif (len(split_param) == 0):
		kf = list(KFold(n_splits=kfold, shuffle=True, random_state=0).split(X, Y))
	else:
		if ('train_size' in split_param and 'test_size' in split_param):
			kf = list(StratifiedShuffleSplit(n_splits=kfold, train_size=split_param['train_size'], test_size=split_param['test_size'], random_state=0).split(X, Y))
		else:
			kf = list(StratifiedKFold(n_splits=kfold, shuffle=split_param.setdefault('shuffle', True), random_state=0).split(X, Y))
	crsval_results, crsval_povl, crsval_spearman, crsval_kendalltau, crsval_pearson = [[] for i in range(5)]
	for i, (train_idx, test_idx) in enumerate(kf):
		del PL_NAMES[:]
		PL_SET.clear()
		print('\n' + '-' * 80 + '\n' + '%s time validation' % imath.ordinal(i+1) + '\n' + '-' * 80 + '\n')
		sub_X = X.iloc[train_idx,:].as_matrix()
		sub_Y = Y[train_idx]
		sub_idx_df = pd.DataFrame(np.arange(sub_X.shape[0]), index=X.index[train_idx])
		if (cfg_param.setdefault('save_crsval_idx', False)):
			io.write_df(sub_idx_df, 'clt_sub_idx_crsval_%s%s.npz' % (i, lbidstr), with_idx=True)
		results, preds = [[] for x in range(2)]
		for vars in model_iter(**model_param):
			if (global_param['comb']):
				mdl_name, mdl = [vars[x] for x in range(2)]
			else:
				clt_name, clt = [vars[x] for x in range(2)]
				# filt_name, filter, clt_name, clt= [vars[x] for x in range(4)]
			print('#' * 80)
			# Assemble a pipeline
			if ('filter' in locals() and filter != None):
				model_name = '%s [Ft Filt] & %s [CLT]' % (filt_name, clt_name)
				pipeline = Pipeline([('featfilt', clone(filter)), ('clt', clt)])
			elif ('clt' in locals() and clt != None):
				model_name = '%s [CLT]' % clt_name
				if (model_param.setdefault('is_nn', False)):
					pipeline = clt
				else:
					pipeline = Pipeline([('clt', clt)])
			else:
				model_name = mdl_name
				pipeline = mdl
			if (model_name in PL_SET): continue
			PL_NAMES.append(model_name)
			PL_SET.add(model_name)
			print(model_name)
			# Benchmark results
			bm_results = benchmark(pipeline, sub_X, sub_Y, **{k:model_param[k] for k in model_param.keys() if k in ['metric', 'is_fuzzy', 'is_nn', 'constraint', 'use_gpu']})
			results.append([bm_results[x] for x in ['homogeneity', 'completeness', 'v_measure', 'f_measure', 'purity', 'ri', 'ari', 'mi', 'ami', 'nmi', 'fmi', 'slcoef', 'chscore', 'time'] if x in bm_results])
			preds.append(bm_results['pred_lb'])
			if (cfg_param.setdefault('save_crsval_pred', False)):
				io.write_npz(dict(pred_lb=bm_results['pred_lb'], true_lb=sub_Y), 'clt_pred_crsval_%s_%s%s' % (i, model_name.replace(' ', '_').lower(), lbidstr))
			print('\n')
		# Cross validation results
		crsval_results.append(results)
		# Prediction overlap
		if ((model_param.setdefault('is_fuzzy', False) or is_fuzzy) and len(preds[0].shape) > 1 and preds[0].shape[1] > 1):
			preds = [pred.argmax(axis=1) for pred in preds]
			sub_Y = sub_Y.argmax(axis=1)
		preds_mt = np.column_stack(preds)
		if (model_param.setdefault('is_nn', False)):
			sub_Y_lb = sub_Y.argmax(axis=1).reshape((-1,))
			sub_Y_lb[sub_Y.sum(axis=1) == 0] == -1
			sub_Y = sub_Y_lb
		preds.append(sub_Y)
		tpreds_mt = np.column_stack([x.ravel() for x in preds])
		crsval_povl.append(pred_ovl(preds_mt, sub_Y))
		# Spearman's rank correlation
		crsval_spearman.append(stats.spearmanr(tpreds_mt))
		# Kendall rank correlation
#		crsval_kendalltau.append(stats.kendalltau(preds_mt)[0])
		# Pearson correlation
#		crsval_pearson.append(stats.pearsonr(preds_mt)[0])
		del sub_X, sub_Y
		print('\n')
	perf_avg = np.array(crsval_results).mean(axis=0)
	perf_std = np.array(crsval_results).std(axis=0)
	povl_avg = np.array(crsval_povl).mean(axis=0).round()
	spmnr_avg = np.array([crsp[0] for crsp in crsval_spearman]).mean(axis=0)
	spmnr_pval = np.array([crsp[1] for crsp in crsval_spearman]).mean(axis=0)
#	kndtr_avg = np.array(crsval_kendalltau).mean(axis=0)
#	prsnr_avg = np.array(crsval_pearson).mean(axis=0)

	## Save performance data
	metric_idx = [METRIC_NAME[x] for x in ['homogeneity', 'completeness', 'v_measure', 'f_measure', 'purity', 'ri', 'ari', 'mi', 'ami', 'nmi', 'fmi', 'slcoef', 'chscore', 'time'] if x in bm_results]
	# metric_idx = ['Homogeneity', 'Completeness', 'V-measure', 'Fowlkes-Mallows Index', 'Adjusted Rand Index', 'Adjusted Mutual Information', 'Normalized Mutual Information', 'Silhouette Coefficient', 'Calinski and Harabaz Score', 'Elapsed Time']
	perf_avg_df = pd.DataFrame(perf_avg.T, index=metric_idx, columns=PL_NAMES)
	perf_std_df = pd.DataFrame(perf_std.T, index=metric_idx, columns=PL_NAMES)
	povl_idx = [' & '.join(x) for x in imath.subset(PL_NAMES, min_crdnl=1)]
	povl_avg_df = pd.DataFrame(povl_avg, index=povl_idx, columns=['pred_ovl', 'tpred_ovl'])
	spmnr_avg_df = pd.DataFrame(spmnr_avg, index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	spmnr_pval_df = pd.DataFrame(spmnr_pval, index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	if (cfg_param.setdefault('save_perf_avg', False)):
		perf_avg_df.to_excel('perf_avg_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_perf_avg_npz', False)):
		io.write_df(perf_avg_df, 'perf_avg_clt%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_perf_std', False)):
		perf_std_df.to_excel('perf_std_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_perf_std_npz', False)):
		io.write_df(perf_std_df, 'perf_std_clt%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_povl', False)):
		povl_avg_df.to_excel('cpovl_avg_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_povl_npz', False)):
		io.write_df(povl_avg_df, 'povl_avg_clt%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_avg', False)):
		spmnr_avg_df.to_excel('spmnr_avg_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_avg_npz', False)):
		io.write_df(spmnr_avg_df, 'spmnr_avg_clt%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_pval', False)):
		spmnr_pval_df.to_excel('spmnr_pval_clt%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_pval_npz', False)):
		io.write_df(spmnr_pval_df, 'spmnr_pval_clt%s.npz' % lbidstr, with_idx=True)

	## Plot figures
	group_dict = {}
	for i, pl in enumerate(PL_NAMES):
		group_dict.setdefault(tuple(set(difflib.get_close_matches(pl, PL_NAMES))), []).append(i)
	if (len(group_dict) == len(PL_NAMES)):
		groups = None
	else:
		group_array = np.array(group_dict.values())
		group_array.sort()
		groups = group_array.tolist()
	if (cfg_param.setdefault('plot_prc', False)):
		pass
	if (cfg_param.setdefault('save_auc', False)):
		pass
	filt_num, clt_num = len(FILT_NAMES), len(CLT_NAMES)
	if (cfg_param.setdefault('plot_metric', False)):
		for mtrc in metric_idx:
			mtrc_avg_list, mtrc_std_list = [[] for i in range(2)]
			if (global_param['comb']):
				mtrc_avg = perf_avg_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				mtrc_std = perf_std_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=PL_NAMES, labels=None, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clt_ft' % (mtrc.replace(' ', '_').lower(), lbidstr), plot_cfg=common_cfg)
			else:
				for i in range(filt_num):
					offset = i * clt_num
					mtrc_avg_list.append(perf_avg_df.ix[mtrc,offset:offset+clt_num].as_matrix().reshape((1,-1)))
					mtrc_std_list.append(perf_std_df.ix[mtrc,offset:offset+clt_num].as_matrix().reshape((1,-1)))
				mtrc_avg = np.concatenate(mtrc_avg_list)
				mtrc_std = np.concatenate(mtrc_std_list)
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=CLT_NAMES, labels=FILT_NAMES, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clt_ft' % (mtrc.replace(' ', '_').lower(), lbidstr), plot_cfg=common_cfg)


def tune_param(mdl_name, mdl, X, Y, rdtune, params, n_jobs=-1):
	if (rdtune):
		param_dist, n_iter = [params[k] for k in ['param_dist', 'n_iter']]
		grid = RandomizedSearchCV(estimator=mdl, param_distributions=param_dist, n_iter=n_iter, scoring='f1_%s' % avg if mltl else 'f1', n_jobs=n_jobs, error_score=0)
	else:
		param_grid, cv = [params[k] for k in ['param_grid', 'cv']]
		grid = GridSearchCV(estimator=mdl, param_grid=param_grid, scoring='f1_micro' if mltl else 'f1', cv=cv, n_jobs=n_jobs, error_score=0)
	grid.fit(X, Y)
	print("The best parameters of [%s] are %s, with a score of %0.3f" % (mdl_name, grid.best_params_, grid.best_score_))
	# Store all the parameter candidates into a dictionary of list
	if (rdtune):
		param_grid = {}
		for p_option in grid.cv_results_['params']:
			for p_name, p_val in p_option.items():
				param_grid.setdefault(p_name, []).append(p_val)
	else:
		param_grid = grid.param_grid
	# Index the parameter names and valules
	dim_names = dict([(k, i) for i, k in enumerate(param_grid.keys())])
	dim_vals = {}
	for pn in dim_names.keys():
		dim_vals[pn] = dict([(k, i) for i, k in enumerate(param_grid[pn])])
	# Create data cube
	score_avg_cube = np.ndarray(shape=[len(param_grid[k]) for k in param_grid.keys()], dtype='float')
	score_std_cube = np.ndarray(shape=[len(param_grid[k]) for k in param_grid.keys()], dtype='float')
	# Calculate the score list
	score_avg_list = (np.array(grid.cv_results_['mean_train_score']) + np.array(grid.cv_results_['mean_test_score'])) / 2
	score_std_list = (np.array(grid.cv_results_['std_train_score']) + np.array(grid.cv_results_['std_test_score'])) / 2
	# Fill in the data cube
	for i, p_option in enumerate(grid.cv_results_['params']):
		idx = np.zeros((len(dim_names),), dtype='int')
		for k, v in p_option.items():
			idx[dim_names[k]] = dim_vals[k][v]
		score_avg_cube[tuple(idx)] = score_avg_list[i]
		score_std_cube[tuple(idx)] = score_std_list[i]
	return grid.best_params_, grid.best_score_, score_avg_cube, score_std_cube, dim_names, dim_vals


# Cluster filtering
def filt_clt(X, Y, method='std', **kwargs):
	Y = np.copy(Y)
	if (method == 'std'):
		threshold = kwargs.setdefault('threshold', 0.2)
		return filt_clt_std(X, Y, threshold=threshold)


def filt_clt_std(X, Y, threshold=0.2):
	clts = set(Y)
	clts.discard(-1)
	filtout = []
	for clt_lb in clts:
		clt_idx = np.where(Y==clt_lb)[0]
		if (X[clt_idx,:].std(axis=0).mean() > threshold):
			filtout.append(clt_lb)
	for fo in filtout:
		np.place(Y, Y==fo, -1)
	return Y


class DummyCluster(BaseEstimator, ClusterMixin, TransformerMixin):
	def __init__(self, output=None, **kwargs):
		self.output_ = io.read_npz(output)['pred_lb']

	def fit_predict(self, X, y=None, **kwargs):
		return self.output_

	def fit(self, X, y=None, constraint=None):
		return self

	def predict(self, X, constraint=None):
		return self.output_
