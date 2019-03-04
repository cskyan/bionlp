#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: txtclf.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-07-05 14:39:18
###########################################################################
#

import os, sys, difflib, itertools
from time import time

import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, label_binarize, normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from keras.utils.io_utils import HDF5Matrix

from util import io, func, plot
import util.math as imath

common_cfg = {}


def init(plot_cfg={}, plot_common={}):
	if (len(plot_cfg) > 0 and plot_cfg['MON'] is not None):
		plot.MON = plot_cfg['MON']
	global common_cfg
	if (len(plot_common) > 0):
		common_cfg = plot_common

		
def get_featw(pipeline, feat_num):
	feat_w_dict, sub_feat_w = [{} for i in range(2)]
	filt_feat_idx = feature_idx = np.arange(feat_num)
	for component in ('featfilt', 'clf'):
		if (type(pipeline) != Pipeline):
			if (component == 'featfilt'):
				continue
			else:
				cmpn = pipeline
		elif (pipeline.named_steps.has_key(component)):
			cmpn = pipeline.named_steps[component]
		else:
			continue
		if (hasattr(cmpn, 'estimators_')):
			for i, estm in enumerate(cmpn.estimators_):
				filt_subfeat_idx = feature_idx[:]
				if (hasattr(estm, 'get_support')):
					filt_subfeat_idx = feature_idx[estm.get_support()]
				for measure in ('feature_importances_', 'coef_', 'scores_'):
					if (hasattr(estm, measure)):
						filt_subfeat_w = getattr(estm, measure)
						subfeat_w = (filt_subfeat_w.min() - 1) * np.ones_like(feature_idx)
						# subfeat_w[filt_subfeat_idx] = normalize(estm.feature_importances_, norm='l1')
						subfeat_w[filt_subfeat_idx] = filt_subfeat_w
						# print 'Sub FI shape: (%s)' % ','.join([str(x) for x in filt_subfeat_w.shape])
						# print 'Feature Importance inside %s Ensemble Method: %s' % (component, filt_subfeat_w)
						sub_feat_w[(component, i)] = subfeat_w
		if (hasattr(component, 'get_support')):
			filt_feat_idx = feature_idx[component.get_support()]
		for measure in ('feature_importances_', 'coef_', 'scores_'):
			if (hasattr(cmpn, measure)):
				filt_feat_w = getattr(cmpn, measure)
				# print '*' * 80 + '\n%s\n'%filt_feat_w + '*' * 80
				feat_w = (filt_feat_w.min() - 1) * np.ones_like(feature_idx)
				# feat_w[filt_feat_idx] = normalize(filt_feat_w, norm='l1')
				feat_w[filt_feat_idx] = filt_feat_w
				# print '*' * 80 + '\n%s\n'%feat_w + '*' * 80
				feat_w_dict[(component, measure)] = feat_w
				print 'FI shape: (%s)' % ','.join([str(x) for x in feat_w_dict[(component, measure)].shape])
				print 'Sample 10 Feature from %s.%s: %s' % (component, measure, feat_w[feat_w > 0][:10])
				# print 'Feature Importance from %s.%s: %s' % (component, measure, feat_w)
	return feat_w_dict, sub_feat_w


def get_score(pipeline, X_test, mltl=False):
	if ((not isinstance(pipeline, Pipeline) and hasattr(pipeline, 'predict_proba')) or(isinstance(pipeline.named_steps['clf'], OneVsRestClassifier) and hasattr(pipeline.named_steps['clf'].estimators_[0], 'predict_proba')) or (not isinstance(pipeline.named_steps['clf'], OneVsRestClassifier) and hasattr(pipeline, 'predict_proba'))):
		if (mltl):
			return pipeline.predict_proba(X_test)
		else:
			# return pipeline.predict_proba(X_test)[:, 1]
			return pipeline.predict_proba(X_test)
	elif (hasattr(pipeline, 'decision_function')):
		return pipeline.decision_function(X_test)
	else:
		print 'Neither probability estimate nor decision function is supported in the classification model!'
		return [0] * Y_test.shape[0]


# Benchmark
def benchmark(pipeline, X_train, Y_train, X_test, Y_test, mltl=False, signed=False, average='micro'):
	print '+' * 80
	print 'Training Model: '
	print pipeline
	t0 = time()
	pipeline.fit(X_train, Y_train)
	train_time = time() - t0
	print 'train time: %0.3fs' % train_time

	t0 = time()
	orig_pred = pred = pipeline.predict(X_test)
	test_time = time() - t0
	print '+' * 80
	print 'Testing: '
	print 'test time: %0.3fs' % test_time
	
	is_mltl = mltl
	if (signed):
		Y_test = np.column_stack([np.abs(Y_test).reshape((Y_test.shape[0],-1))] + [label_binarize(lb, classes=[-1,1,0])[:,1] for lb in (np.sign(Y_test).astype('int8').reshape((Y_test.shape[0],-1))).T]) if (len(Y_test.shape) < 2 or Y_test.shape[1] == 1 or np.where(Y_test<0)[0].shape[0]>0) else Y_test
		pred = np.column_stack([np.abs(pred).reshape((pred.shape[0],-1))] + [label_binarize(lb, classes=[-1,1,0])[:,1] for lb in (np.sign(pred).astype('int8').reshape((pred.shape[0],-1))).T]) if (len(pred.shape) < 2 or pred.shape[1] == 1 or np.where(pred<0)[0].shape[0]>0) else pred
		is_mltl = True
	try:
		accuracy = metrics.accuracy_score(Y_test, pred)
	except ValueError as e:
		print e
		Y_test, pred = Y_test.ravel(), pred.ravel()
		accuracy = metrics.accuracy_score(Y_test, pred)
	print 'accuracy: %0.3f' % accuracy
	if (is_mltl and average == 'all'):
		micro_precision = metrics.precision_score(Y_test, pred, average='micro')
		print 'micro-precision: %0.3f' % micro_precision
		micro_recall = metrics.recall_score(Y_test, pred, average='micro')
		print 'micro-recall: %0.3f' % micro_recall
		micro_fscore = metrics.fbeta_score(Y_test, pred, beta=1, average='micro')
		print 'micro-fscore: %0.3f' % micro_fscore
		macro_precision = metrics.precision_score(Y_test, pred, average='macro')
		print 'macro-precision: %0.3f' % macro_precision
		macro_recall = metrics.recall_score(Y_test, pred, average='macro')
		print 'macro-recall: %0.3f' % macro_recall
		macro_fscore = metrics.fbeta_score(Y_test, pred, beta=1, average='macro')
		print 'macro-fscore: %0.3f' % macro_fscore
	else:
		precision = metrics.precision_score(Y_test, pred, average=average if is_mltl else 'binary')
		print 'precision: %0.3f' % precision
		recall = metrics.recall_score(Y_test, pred, average=average if is_mltl else 'binary')
		print 'recall: %0.3f' % recall
		fscore = metrics.fbeta_score(Y_test, pred, beta=1, average=average if is_mltl else 'binary')
		print 'fscore: %0.3f' % fscore

	print 'classification report:'
	print metrics.classification_report(Y_test, pred)

	print 'confusion matrix:'
	if (is_mltl):
		pass
	else:
		print metrics.confusion_matrix(Y_test, pred)
	print '+' * 80

	clf = pipeline.named_steps['clf'] if (type(pipeline) is Pipeline) else pipeline
	if ((isinstance(clf, OneVsRestClassifier) and hasattr(clf.estimators_[0], 'predict_proba')) or (not isinstance(clf, OneVsRestClassifier) and hasattr(pipeline, 'predict_proba'))):
		if (mltl):
			scores = pipeline.predict_proba(X_test)
			if (type(scores) == list):
				scores = np.concatenate([score[:, -1].reshape((-1, 1)) for score in scores], axis=1)
		else:
			scores = pipeline.predict_proba(X_test)[:, -1]
	elif (hasattr(pipeline, 'decision_function')):
		scores = pipeline.decision_function(X_test)
	else:
		print 'Neither probability estimate nor decision function is supported in the classification model! ROC and PRC figures will be invalid.'
		scores = [0] * Y_test.shape[0]

	if (signed and (len(scores.shape) < 2 or scores.shape[1] < pred.shape[1])):
		scores = np.concatenate([np.abs(scores).reshape((scores.shape[0],-1))] + [label_binarize(lb, classes=[-1,1,0])[:,:2] for lb in (np.sign(scores).astype('int8').reshape((scores.shape[0],-1))).T], axis=1)
		
	if (is_mltl):
		if ((len(Y_test.shape) == 1 or Y_test.shape[1] == 1) and len(np.unique(Y_test)) > 2):
			lbz = LabelBinarizer()
			Y_test = lbz.fit_transform(Y_test)
		def micro():
			# Micro-average ROC curve
			y_true = np.array(Y_test)
			s_array = np.array(scores)
			if (len(s_array.shape) == 3):
				s_array = s_array[:,:,1].reshape((s_array.shape[0],s_array.shape[1],))
			if (y_true.shape[0] == s_array.shape[1] and y_true.shape[1] == s_array.shape[0]):
				s_array = s_array.T
			return metrics.roc_curve(y_true.ravel(), s_array.ravel())
		def macro():
			# Macro-average ROC curve
			n_classes = Y_test.shape[1]
			fpr, tpr = [dict() for i in range(2)]
			for i in range(n_classes):
				fpr[i], tpr[i], _ = metrics.roc_curve(Y_test[:, i], scores[:, i])
			# First aggregate all false positive rates
			all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
			# Then interpolate all ROC curves at this points
			mean_tpr = np.zeros_like(all_fpr)
			for i in range(n_classes):
				mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
			# Finally average it and compute AUC
			mean_tpr /= n_classes
			return all_fpr, mean_tpr, _
		if (average == 'micro'):
			roc = micro()
		elif (average == 'macro'):
			roc = macro()
		elif (average == 'all'):
			micro_roc = micro()
			macro_roc = macro()
		if (type(scores) == list):
			scores = np.array(scores)[:,:,0]
		prc = metrics.precision_recall_curve(Y_test.ravel(), scores.ravel()) # Only micro-prc is supported
	else:
		roc = metrics.roc_curve(Y_test, scores)
		prc = metrics.precision_recall_curve(Y_test, scores)
#	print 'ROC:\n%s\n%s' % (roc[0], roc[1])
#	print 'PRC:\n%s\n%s' % (prc[0], prc[1])

	print 'Training and Testing X shape: %s; %s' % (', '.join(['(%s)' % ','.join([str(x) for x in X.shape]) for X in X_train]) if type(X_train) is list else '(%s)' % ','.join([str(x) for x in X_train.shape]), ', '.join(['(%s)' % ','.join([str(x) for x in X.shape]) for X in X_test]) if type(X_test) is list else '(%s)' % ','.join([str(x) for x in X_test.shape]))
	feat_w_dict, sub_feat_w = [{} for i in range(2)]
	filt_feat_idx = feature_idx = np.arange(X_train[0].shape[1] if type(X_train) is list else X_train.shape[1])
	for component in ('featfilt', 'clf'):
		if (type(pipeline) != Pipeline):
			if (component == 'featfilt'):
				continue
			else:
				cmpn = pipeline
		elif (pipeline.named_steps.has_key(component)):
			cmpn = pipeline.named_steps[component]
		else:
			continue
		if (hasattr(cmpn, 'estimators_')):
			for i, estm in enumerate(cmpn.estimators_):
				filt_subfeat_idx = filt_feat_idx[:]
				if (hasattr(estm, 'get_support')):
					filt_subfeat_idx = filt_feat_idx[estm.get_support()]
				for measure in ('feature_importances_', 'coef_', 'scores_'):
					if (hasattr(estm, measure)):
						filt_subfeat_w = getattr(estm, measure)
						subfeat_w = (filt_subfeat_w.min() - 1) * np.ones_like(feature_idx)
	#					subfeat_w[filt_subfeat_idx][:len(estm.feature_importances_)] = normalize(estm.feature_importances_, norm='l1')
						subfeat_w[filt_subfeat_idx][:len(filt_subfeat_w)] = filt_subfeat_w
						# print 'Sub FI shape: (%s)' % ','.join([str(x) for x in filt_subfeat_w.shape])
						# print 'Feature Importance inside %s Ensemble Method: %s' % (component, filt_subfeat_w)
						sub_feat_w[(component, i)] = subfeat_w
			for measure in ('feature_importances_', 'coef_', 'scores_'):
				if (hasattr(cmpn, measure)):
					filt_feat_w = getattr(cmpn, measure)
#					print '*' * 80 + '\n%s\n'%filt_feat_w + '*' * 80
					feat_w = (filt_feat_w.min() - 1) * np.ones_like(feature_idx)
#					feat_w[filt_feat_idx][:filt_feat_w.shape[1] if len(filt_feat_w.shape) > 1 else len(filt_feat_w)] = normalize(filt_feat_w[1,:] if len(filt_feat_w.shape) > 1 else filt_feat_w, norm='l1')
					feat_w[filt_feat_idx][:filt_feat_w.shape[1] if len(filt_feat_w.shape) > 1 else len(filt_feat_w)] = filt_feat_w[1,:] if len(filt_feat_w.shape) > 1 else filt_feat_w
#					print '*' * 80 + '\n%s\n'%feat_w + '*' * 80
					feat_w_dict[(component, measure)] = feat_w
					print 'FI shape: (%s)' % ','.join([str(x) for x in feat_w_dict[(component, measure)].shape])
					print 'Sample 10 Feature from %s.%s: %s' % (component, measure, feat_w[feat_w > 0][:10])
#					print 'Feature Importance from %s.%s: %s' % (component, measure, feat_w)
			if (hasattr(cmpn, 'get_support')):
				filt_feat_idx = filt_feat_idx[cmpn.get_support()]
	print '\n'
	if (is_mltl and average == 'all'):
		return {'accuracy':accuracy, 'micro-precision':micro_precision, 'micro-recall':micro_recall, 'micro-fscore':micro_fscore, 'macro-precision':macro_precision, 'macro-recall':macro_recall, 'macro-fscore':macro_fscore, 'train_time':train_time, 'test_time':test_time, 'micro-roc':micro_roc, 'macro-roc':macro_roc, 'prc':prc, 'feat_w':feat_w_dict, 'sub_feat_w':sub_feat_w, 'pred_lb':orig_pred}
	else:
		return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'fscore':fscore, 'train_time':train_time, 'test_time':test_time, 'roc':roc, 'prc':prc, 'feat_w':feat_w_dict, 'sub_feat_w':sub_feat_w, 'pred_lb':orig_pred}


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
	subset_idx = list(imath.subset(range(dim), min_crdnl=1))
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
	
	
def save_featw(features, crsval_featw, crsval_subfeatw, cfg_param={}, lbid=''):
	lbidstr = ('_' + (str(lbid) if lbid != -1 else 'all')) if lbid is not None and lbid != '' else ''
	for k, v in crsval_featw.iteritems():
		measure_str = k.replace(' ', '_').strip('_').lower()
		feat_w_mt = np.column_stack(v)
		mms = MinMaxScaler()
		feat_w_mt = mms.fit_transform(feat_w_mt)
		feat_w_avg = feat_w_mt.mean(axis=1)
		feat_w_std = feat_w_mt.std(axis=1)
		sorted_idx = np.argsort(feat_w_avg, axis=-1)[::-1]
		# sorted_idx = sorted(range(feat_w_avg.shape[0]), key=lambda k: feat_w_avg[k])[::-1]
		sorted_feat_w = np.column_stack((features[sorted_idx], feat_w_avg[sorted_idx], feat_w_std[sorted_idx]))
		feat_w_df = pd.DataFrame(sorted_feat_w, index=sorted_idx, columns=['Feature Name', 'Importance Mean', 'Importance Std'])
		if (cfg_param.setdefault('save_featw', False)):
			feat_w_df.to_excel('featw%s_%s.xlsx' % (lbidstr, measure_str))
		if (cfg_param.setdefault('save_featw_npz', False)):
			io.write_df(feat_w_df, 'featw%s_%s' % (lbidstr, measure_str), with_idx=True)
		if (cfg_param.setdefault('plot_featw', False)):
			plot.plot_bar(feat_w_avg[sorted_idx[:10]].reshape((1,-1)), feat_w_std[sorted_idx[:10]].reshape((1,-1)), features[sorted_idx[:10]], labels=None, title='Feature importances', fname='fig_featw%s_%s' % (lbidstr, measure_str), plot_cfg=common_cfg)
	for k, v in crsval_subfeatw.iteritems():
		measure_str = k.replace(' ', '_').strip('_').lower()
		subfeat_w_mt = np.column_stack(v)
		mms = MinMaxScaler()
		subfeat_w_mt = mms.fit_transform(subfeat_w_mt)
		subfeat_w_avg = subfeat_w_mt.mean(axis=1)
		subfeat_w_std = subfeat_w_mt.std(axis=1)
		sorted_idx = np.argsort(subfeat_w_avg, axis=-1)[::-1]
		sorted_subfeat_w = np.column_stack((features[sorted_idx], subfeat_w_avg[sorted_idx], subfeat_w_std[sorted_idx]))
		subfeat_w_df = pd.DataFrame(sorted_subfeat_w, index=sorted_idx, columns=['Feature Name', 'Importance Mean', 'Importance Std'])
		if (cfg_param.setdefault('save_subfeatw', False)):
			subfeat_w_df.to_excel('subfeatw%s_%s.xlsx' % (lbidstr, measure_str))
		if (cfg_param.setdefault('save_subfeatw_npz', False)):
			io.write_df(subfeat_w_df, 'subfeatw%s_%s' % (lbidstr, measure_str), with_idx=True)
		if (cfg_param.setdefault('plot_subfeatw', False)):
			plot.plot_bar(subfeat_w_avg[sorted_idx[:10]].reshape((1,-1)), subfeat_w_std[sorted_idx[:10]].reshape((1,-1)), features[sorted_idx[:10]], labels=None, title='Feature importances', fname='fig_subfeatw_%s' % measure_str, plot_cfg=common_cfg)
	
	
# Classification
def classification(X_train, Y_train, X_test, model_iter, model_param={}, cfg_param={}, global_param={}, lbid=''):
	global common_cfg
	FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET = model_param['glb_filtnames'], model_param['glb_clfnames'], global_param['pl_names'], global_param['pl_set']
	lbidstr = ('_' + (str(lbid) if lbid != -1 else 'all')) if lbid is not None and lbid != '' else ''
	to_hdf, hdf5_fpath = cfg_param.setdefault('to_hdf', False), '%s' % 'crsval_dataset.h5' if cfg_param.setdefault('hdf5_fpath', 'crsval_dataset.h5') is None else cfg_param['hdf5_fpath']
	
	# Format the data
	if (type(X_train) == list):
		assert all([len(x) == len(X_train[0]) for x in X_train[1:]])
		X_train = [pd.DataFrame(x) if (type(x) != pd.io.parsers.TextFileReader and type(x) != pd.DataFrame) else x for x in X_train]
		X_train = [pd.concat(x) if (type(x) == pd.io.parsers.TextFileReader and not to_hdf) else x for x in X_train]
	else:
		if (type(X_train) != pd.io.parsers.TextFileReader and type(X_train) != pd.DataFrame):
			X_train = pd.DataFrame(X_train)
		X_train = pd.concat(X_train) if (type(X_train) == pd.io.parsers.TextFileReader and not to_hdf) else X_train
	if (type(X_test) == list):
		assert all([len(x) == len(X_test[0]) for x in X_test[1:]])
		X_test = [pd.DataFrame(x) if (type(x) != pd.io.parsers.TextFileReader and type(x) != pd.DataFrame) else x for x in X_test]
		X_test = [pd.concat(x) if (type(x) == pd.io.parsers.TextFileReader and not to_hdf) else x for x in X_test]
	else:
		if (type(X_test) != pd.io.parsers.TextFileReader and type(X_test) != pd.DataFrame):
			X_test = pd.DataFrame(X_test)
		X_test = pd.concat(X_test) if (type(X_test) == pd.io.parsers.TextFileReader and not to_hdf) else X_test
	if (type(Y_train) != pd.io.parsers.TextFileReader and type(Y_train) != pd.DataFrame):
		Y_train = pd.DataFrame(Y_train)
	Y_train_mt = Y_train.as_matrix().reshape((Y_train.shape[0],)) if (len(Y_train.shape) == 1 or Y_train.shape[1] == 1) else Y_train.as_matrix()
	mltl=True if len(Y_train_mt.shape) > 1 and Y_train_mt.shape[1] > 1 or 2 in Y_train_mt else False

	print 'Classification is starting...'
	preds, probs, scores = [[] for i in range(3)]
	crsval_featw, crsval_subfeatw = [{} for i in range(2)]
	for vars in model_iter(**model_param):
		if (global_param['comb']):
			mdl_name, mdl = [vars[x] for x in range(2)]
		else:
			filt_name, filter, clf_name, clf= [vars[x] for x in range(4)]
		print '#' * 80
		# Assemble a pipeline
		if ('filter' in locals() and filter != None):
			model_name = '%s [Ft Filt] & %s [CLF]' % (filt_name, clf_name)
			pipeline = Pipeline([('featfilt', clone(filter)), ('clf', clf)])
		elif ('clf' in locals() and clf != None):
			model_name = '%s [CLF]' % clf_name
			pipeline = Pipeline([('clf', clf)])
		else:
			model_name = mdl_name
			pipeline = mdl if (type(mdl) is Pipeline) else Pipeline([('clf', mdl)])
		if (model_name in PL_SET): continue
		PL_NAMES.append(model_name)
		PL_SET.add(model_name)
		print model_name
		# Build the model
		print '+' * 80
		print 'Training Model: '
		print pipeline
		t0 = time()
		pipeline.fit(X_train, Y_train_mt)
		train_time = time() - t0
		print 'train time: %0.3fs' % train_time
		t0 = time()
		pred = pipeline.predict(X_test)
		prob = pipeline.predict_proba(X_test)
		test_time = time() - t0
		print '+' * 80
		print 'Testing: '
		print 'test time: %0.3fs' % test_time
		preds.append(pred)
		probs.append(prob)
		scores.append(get_score(pipeline, X_test, mltl))
		
		# Save predictions and model
		if (cfg_param.setdefault('save_pred', True)):
			io.write_npz(dict(pred_lb=pred, pred_prob=prob), 'clf_pred_%s%s' % (model_name.replace(' ', '_').lower(), lbidstr))
		if (cfg_param.setdefault('save_model', True)):
			mdl_name = '%s' % model_name.replace(' ', '_').lower()
			if (all([hasattr(pipeline.steps[i][1], 'save') for i in range(len(pipeline.steps))])):
				for sub_mdl_name, mdl in pipeline.steps:
					mdl.save('%s_%s%s' % (mdl_name, sub_mdl_name.replace(' ', '_').lower(), lbidstr), **global_param.setdefault('mdl_save_kwargs', {}))
			else:
				io.write_obj(pipeline, '%s%s' % (mdl_name, lbidstr))

		# Feature importances
		feat_w, sub_feat_w = get_featw(pipeline, X_train[0].shape[1] if (type(X_train) is list) else X_train.shape[1])
		for k, v in feat_w.iteritems():
			key = '%s_%s_%s' % (model_name, k[0], k[1])
			crsval_featw.setdefault(key, []).append(v)
		for k, v in sub_feat_w.iteritems():
			key = '%s_%s_%s' % (model_name, k[0], k[1])
			crsval_subfeatw.setdefault(key, []).append(v)
		print '\n'

	# Prediction overlap
	preds_mt = np.column_stack([x.ravel() for x in preds])
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
		povl_df.to_excel('cpovl_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_povl_npz', False)):
		io.write_df(povl_df, 'povl_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr', False)):
		spmnr_df.to_excel('spmnr_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_npz', False)):
		io.write_df(spmnr_df, 'spmnr_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_pval', False)):
		spmnr_pval_df.to_excel('spmnr_pval_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_pval_npz', False)):
		io.write_df(spmnr_pval_df, 'spmnr_pval_clf%s.npz' % lbidstr, with_idx=True)
	save_featw(X_train[0].columns.values if (type(X_train) is list) else X_train.columns.values, crsval_featw, crsval_subfeatw, cfg_param=cfg_param, lbid=lbid)
	
	return preds, scores


def kf2data(kf, X, Y, to_hdf=False, hdf5_fpath='crsval_dataset.h5'):
	if (to_hdf):
		import h5py
		from keras.utils.io_utils import HDF5Matrix
		hdf5_fpath = hdf5_fpath if hdf5_fpath else os.path.abspath('crsval_dataset.h5')
	for i, (train_idx, test_idx) in enumerate(kf):
		if (type(X)==list):
			if (type(X[0]) == pd.io.parsers.TextFileReader):
				pass
			assert all([len(x) == len(X[0]) for x in X[1:]])
			X_train, X_test = [x.iloc[train_idx,:] for x in X] if type(X[0]) != HDF5Matrix else [x[train_idx,:] for x in X], [x.iloc[test_idx,:] for x in X] if type(X[0]) != HDF5Matrix else [x[test_idx,:] for x in X]
			train_idx_df, test_idx_df = pd.DataFrame(np.arange(X_train[0].shape[0]), index=X[0].index[train_idx]), pd.DataFrame(np.arange(X_test[0].shape[0]), index=X[0].index[test_idx])
		else:
			if (type(X) == pd.io.parsers.TextFileReader):
				pass
			X_train, X_test = X.iloc[train_idx,:] if type(X) != HDF5Matrix else X[train_idx], X.iloc[test_idx,:] if type(X) != HDF5Matrix else X[test_idx]
			train_idx_df, test_idx_df = pd.DataFrame(np.arange(X_train.shape[0]), index=X.index[train_idx] if type(X) != HDF5Matrix else None), pd.DataFrame(np.arange(X_test.shape[0]), index=X.index[test_idx] if type(X) != HDF5Matrix else None)
		Y_train, Y_test = Y[train_idx], Y[test_idx]
		# Y_train = Y_train.reshape((Y_train.shape[0],)) if (len(Y_train.shape) > 1 and Y_train.shape[1] == 1) else Y_train
		# Y_test = Y_test.reshape((Y_test.shape[0],)) if (len(Y_test.shape) > 1 and Y_test.shape[1] == 1) else Y_test
		if (to_hdf):
			with h5py.File(hdf5_fpath, 'w') as hf:
				if (type(X_train) == list):
					for idx, x_train in enumerate(X_train):
						hf.create_dataset('X_train%i' % idx, data=x_train.as_matrix() if type(X) != HDF5Matrix else x_train[:])
				else:
					hf.create_dataset('X_train', data=X_train.as_matrix() if type(X) != HDF5Matrix else X_train[:])
				if (type(X_test) == list):
					for idx, x_test in enumerate(X_test):
						hf.create_dataset('X_test%i' % idx, data=x_test.as_matrix() if type(X) != HDF5Matrix else x_test[:])
				else:
					hf.create_dataset('X_test', data=X_test.as_matrix() if type(X) != HDF5Matrix else X_test[:])
				hf.create_dataset('Y_train', data=Y_train if type(Y) != HDF5Matrix else Y_train[:])
				hf.create_dataset('Y_test', data=Y_test if type(Y) != HDF5Matrix else Y_test[:])
			yield i, [HDF5Matrix(hdf5_fpath, 'X_train%i' % idx) for idx in range(len(X_train))] if (type(X_train) == list) else HDF5Matrix(hdf5_fpath, 'X_train'), [HDF5Matrix(hdf5_fpath, 'X_test%i' % idx) for idx in range(len(X_test))] if (type(X_test) == list) else HDF5Matrix(hdf5_fpath, 'X_test'), HDF5Matrix(hdf5_fpath, 'Y_train'), HDF5Matrix(hdf5_fpath, 'Y_test'), train_idx_df, test_idx_df
			# The implementation of HDF5Matrix is not good since it keep all the hdf5 file opened, so we need to manually close them.
			remove_hfps = []
			for hfpath, hf in HDF5Matrix.refs.iteritems():
				if (hfpath.startswith(hdf5_fpath)):
					hf.close()
					remove_hfps.append(hfpath)
			for hfpath in remove_hfps:
				HDF5Matrix.refs.pop(hfpath, None)
		else:
			yield i, [x.as_matrix() for x in X_train] if (type(X_train) == list) else X_train.as_matrix(), [x.as_matrix() for x in X_test] if (type(X_test) == list) else X_test.as_matrix(), Y_train, Y_test, train_idx_df, test_idx_df
			
			
# Evaluation
def evaluate(X_train, Y_train, X_test, Y_test, model_iter, model_param={}, avg='micro', kfold=5, cfg_param={}, global_param={}, lbid=''):
	global common_cfg
	FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET = model_param['glb_filtnames'], model_param['glb_clfnames'], global_param['pl_names'], global_param['pl_set']
	lbidstr = ('_' + (str(lbid) if lbid != -1 else 'all')) if lbid is not None and lbid != '' else ''
	
	# Format the data
	if (type(X_train) == list):
		assert all([len(x) == len(X_train[0]) for x in X_train[1:]])
		X_train = [pd.DataFrame(x) if (type(x) != pd.io.parsers.TextFileReader and type(x) != pd.DataFrame) else x for x in X_train]
		X_train = [pd.concat(x) if (type(x) == pd.io.parsers.TextFileReader and not to_hdf) else x for x in X_train]
	else:
		if (type(X_train) != pd.io.parsers.TextFileReader and type(X_train) != pd.DataFrame):
			X_train = pd.DataFrame(X_train) if type(X_train) != HDF5Matrix else X_train
		X_train = pd.concat(X_train) if (type(X_train) == pd.io.parsers.TextFileReader and not to_hdf) else X_train
	if (type(Y_train) != pd.io.parsers.TextFileReader and type(Y_train) != pd.DataFrame):
		Y_train = pd.DataFrame(Y_train) if (type(Y_train) == pd.io.parsers.TextFileReader and not to_hdf) else Y_train
	if (type(Y_train) != HDF5Matrix):
		Y_train = Y_train.as_matrix().reshape((Y_train.shape[0],)) if (len(Y_train.shape) == 1 or Y_train.shape[1] == 1) else Y_train.as_matrix()
	else:
		Y_train = Y_train
		
	if (type(X_test) == list):
		assert all([len(x) == len(X_test[0]) for x in X_test[1:]])
		X_test = [pd.DataFrame(x) if (type(x) != pd.io.parsers.TextFileReader and type(x) != pd.DataFrame) else x for x in X_test]
		X_test = [pd.concat(x) if (type(x) == pd.io.parsers.TextFileReader and not to_hdf) else x for x in X_test]
	else:
		if (type(X_test) != pd.io.parsers.TextFileReader and type(X_test) != pd.DataFrame):
			X_test = pd.DataFrame(X_test) if type(X_test) != HDF5Matrix else X_test
		X_test = pd.concat(X_test) if (type(X_test) == pd.io.parsers.TextFileReader and not to_hdf) else X_test
	if (type(Y_test) != pd.io.parsers.TextFileReader and type(Y_test) != pd.DataFrame):
		Y_test = pd.DataFrame(Y_test) if (type(Y_test) == pd.io.parsers.TextFileReader and not to_hdf) else Y_test
	if (type(Y_test) != HDF5Matrix):
		Y_test = Y_test.as_matrix().reshape((Y_test.shape[0],)) if (len(Y_test.shape) == 1 or Y_test.shape[1] == 1) else Y_test.as_matrix()
	else:
		Y_test = Y_test

	is_mltl = True if len(Y_train.shape) > 1 and Y_train.shape[1] > 1 or 2 in Y_train else False
		
	print 'Benchmark is starting...'
	mean_fpr = np.linspace(0, 1, 100)
	mean_recall = np.linspace(0, 1, 100)
	xdf = X_train[0] if type(X_train)==list else X_train

	roc_dict, prc_dict, featw_data, subfeatw_data = [{} for i in range(4)]
	## Copy from cross_validate function Start ##
	del PL_NAMES[:]
	PL_SET.clear()
	if (cfg_param.setdefault('npg_ratio', None) is not None):
		npg_ratio = cfg_param['npg_ratio']
		Y_train = np.array(Y_train) # HDF5Matrix is not working in matrix slicing and boolean operation
		y = Y_train[:,0] if (len(Y_train.shape) > 1) else Y_train
		if (1.0 * np.abs(y).sum() / Y_train.shape[0] < 1.0 / (npg_ratio + 1)):
			all_true = np.arange(Y_train.shape[0])[y > 0].tolist()
			all_false = np.arange(Y_train.shape[0])[y <= 0].tolist()
			true_id = np.random.choice(len(all_true), size=int(1.0 / npg_ratio * len(all_false)), replace=True)
			true_idx = [all_true[i] for i in true_id]
			all_train_idx = sorted(set(true_idx + all_false))
			X_train = [x.iloc[all_train_idx] if type(x) != HDF5Matrix else x[all_train_idx] for x in X_train] if (type(X_train) is list) else X_train.iloc[all_train_idx] if type(x) != HDF5Matrix else X_train[all_train_idx]
			Y_train = Y_train[all_train_idx,:] if (len(Y_train.shape) > 1) else Y_train[all_train_idx]
	results, preds = [[] for x in range(2)]
	Y_test = np.column_stack([np.abs(Y_test).reshape((Y_test.shape[0],-1))] + [label_binarize(lb, classes=[-1,1,0])[:,1] for lb in (np.sign(Y_test).astype('int8').reshape((Y_test.shape[0],-1))).T]) if (len(Y_test.shape) < 2 or Y_test.shape[1] == 1 or np.where(Y_test<0)[0].shape[0]>0) else Y_test
	for vars in model_iter(**model_param):
		if (global_param['comb']):
			mdl_name, mdl = [vars[x] for x in range(2)]
		else:
			filt_name, filter, clf_name, clf= [vars[x] for x in range(4)]
		print '#' * 80
		# Assemble a pipeline
		if ('filter' in locals() and filter != None):
			model_name = '%s [Ft Filt] & %s [CLF]' % (filt_name, clf_name)
			pipeline = Pipeline([('featfilt', clone(filter)), ('clf', clf)])
		elif ('clf' in locals() and clf != None):
			model_name = '%s [CLF]' % clf_name
			pipeline = Pipeline([('clf', clf)])
		else:
			model_name = mdl_name
			pipeline = mdl
		if (model_name in PL_SET): continue
		PL_NAMES.append(model_name)
		PL_SET.add(model_name)
		print model_name
		# Benchmark results
		bm_results = benchmark(pipeline, X_train, Y_train, X_test, Y_test, mltl=is_mltl, signed=global_param.setdefault('signed', True if np.where(Y_train<0)[0].shape[0]>0 else False), average=avg)
		# Clear the model environment (e.g. GPU resources)
		del pipeline
		# if (type(pipeline) is Pipeline):
			# for cmpn in pipeline.named_steps.values():
				# if (getattr(cmpn, "clear", None)): cmpn.clear()
		# else:
			# if (getattr(pipeline, "clear", None)):
				# pipeline.clear()
		# Obtain the results
		if (is_mltl and avg == 'all'):
			results.append([bm_results[x] for x in ['accuracy', 'micro-precision', 'micro-recall', 'micro-fscore', 'macro-precision', 'macro-recall', 'macro-fscore', 'train_time', 'test_time']])
		else:
			results.append([bm_results[x] for x in ['accuracy', 'precision', 'recall', 'fscore', 'train_time', 'test_time']])
		preds.append(bm_results['pred_lb'])
		if (cfg_param.setdefault('save_pred', False)):
			io.write_npz(dict(pred_lb=bm_results['pred_lb'], true_lb=Y_test), 'pred_%s%s' % (model_name.replace(' ', '_').lower(), lbidstr))
		if (is_mltl and avg == 'all'):
			micro_id, macro_id = '-'.join([model_name,'micro']), '-'.join([model_name,'macro'])
			roc_dict[micro_id] = roc_dict.setdefault(micro_id, 0) + np.interp(mean_fpr, bm_results['micro-roc'][0], bm_results['micro-roc'][1])
			roc_dict[macro_id] = roc_dict.setdefault(macro_id, 0) + np.interp(mean_fpr, bm_results['macro-roc'][0], bm_results['macro-roc'][1])
		else:
			roc_dict[model_name] = roc_dict.setdefault(model_name, 0) + np.interp(mean_fpr, bm_results['roc'][0], bm_results['roc'][1])
		prc_dict[model_name] = prc_dict.setdefault(model_name, 0) + np.interp(mean_recall, bm_results['prc'][0], bm_results['prc'][1])
		for k, v in bm_results['feat_w'].iteritems():
			key = '%s_%s_%s' % (model_name, k[0], k[1])
			featw_data[key] = v
		for k, v in bm_results['sub_feat_w'].iteritems():
			key = '%s_%s_%s' % (model_name, k[0], k[1])
			subfeatw_data[key] = v
		print '\n'
	# Prediction overlap
	if (True if len(Y_train.shape) > 1 and Y_train.shape[1] > 1 else False):
		preds_mt = np.column_stack([x.ravel() for x in preds])
	else:
		preds_mt = np.column_stack(preds)
	preds.append(Y_test)
	tpreds_mt = np.column_stack([x.ravel() for x in preds])
	## Copy from cross_validate function End ##
	povl = pred_ovl(preds_mt, Y_test)
	# Spearman's rank correlation
	spearman = stats.spearmanr(tpreds_mt)
	# Kendall rank correlation
	# kendalltau = stats.kendalltau(preds_mt)
	# Pearson correlation
	# pearson = stats.pearsonr(preds_mt)
	
	## Save performance data
	if (is_mltl and avg == 'all'):
		metric_idx = ['Accuracy', 'Micro Precision', 'Micro Recall', 'Micro F score', 'Macro Precision', 'Macro Recall', 'Macro F score', 'Train time', 'Test time']
	else:
		metric_idx = ['Accuracy', 'Precision', 'Recall', 'F score', 'Train time', 'Test time']
	perf_df = pd.DataFrame(np.array(results).T, index=metric_idx, columns=PL_NAMES)
	povl_idx = [' & '.join(x) for x in imath.subset(PL_NAMES, min_crdnl=1)]
	povl_df = pd.DataFrame(np.array(povl), index=povl_idx, columns=['pred_ovl', 'tpred_ovl'])
	spmnr_val_df = pd.DataFrame(spearman[0], index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	spmnr_pval_df = pd.DataFrame(spearman[1], index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	if (cfg_param.setdefault('save_tpred', True)):
		io.write_npz(tpreds_mt, 'tpred_clf%s' % lbidstr)
	if (cfg_param.setdefault('save_perf', True)):
		perf_df.to_excel('perf_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_perf_npz', False)):
		io.write_df(perf_df, 'perf_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_povl', False)):
		povl_df.to_excel('povl_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_povl_npz', False)):
		io.write_df(povl_df, 'povl_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr', False)):
		spmnr_val_df.to_excel('spmnr_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_npz', False)):
		io.write_df(spmnr_val_df, 'spmnr_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_pval', False)):
		spmnr_pval_df.to_excel('spmnr_pval_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_pval_npz', False)):
		io.write_df(spmnr_pval_df, 'spmnr_pval_clf%s.npz' % lbidstr, with_idx=True)
	# Feature importances
	try:
		save_featw(xdf.columns.values if type(xdf) != HDF5Matrix else np.arange(xdf.shape[1]), featw_data, subfeatw_data, cfg_param=cfg_param, lbid=lbid)
	except Exception as e:
		print e
		
	## Plot figures
	if (is_mltl and avg == 'all'):
		micro_roc_data, micro_roc_labels, micro_roc_aucs, macro_roc_data, macro_roc_labels, macro_roc_aucs = [[] for i in range(6)]
	else:
		roc_data, roc_labels, roc_aucs = [[] for i in range(3)]
	prc_data, prc_labels, prc_aucs = [[] for i in range(3)]
	for pl in PL_NAMES:
		if (is_mltl and avg == 'all'):
			micro_id, macro_id = '-'.join([pl,'micro']), '-'.join([pl,'macro'])
			micro_mean_tpr, macro_mean_tpr = roc_dict[micro_id], roc_dict[macro_id]
			micro_roc_auc = metrics.auc(mean_fpr, micro_mean_tpr)
			macro_roc_auc = metrics.auc(mean_fpr, macro_mean_tpr)
			micro_roc_data.append([mean_fpr, micro_mean_tpr])
			micro_roc_aucs.append(micro_roc_auc)
			micro_roc_labels.append('%s (AUC=%0.2f)' % (pl, micro_roc_auc))
			macro_roc_data.append([mean_fpr, macro_mean_tpr])
			macro_roc_aucs.append(macro_roc_auc)
			macro_roc_labels.append('%s (AUC=%0.2f)' % (pl, macro_roc_auc))
		else:
			mean_tpr = roc_dict[pl]
			mean_roc_auc = metrics.auc(mean_fpr, mean_tpr)
			roc_data.append([mean_fpr, mean_tpr])
			roc_aucs.append(mean_roc_auc)
			roc_labels.append('%s (AUC=%0.2f)' % (pl, mean_roc_auc))
		mean_prcn = prc_dict[pl]
		mean_prc_auc = metrics.auc(mean_recall, mean_prcn)
		prc_data.append([mean_recall, mean_prcn])
		prc_aucs.append(mean_prc_auc)
		prc_labels.append('%s (AUC=%0.2f)' % (pl, mean_prc_auc))
	group_dict = {}
	for i, pl in enumerate(PL_NAMES):
		group_dict.setdefault(tuple(set(difflib.get_close_matches(pl, PL_NAMES))), []).append(i)
	if (not cfg_param.setdefault('group_by_name', False) or len(group_dict) == len(PL_NAMES)):
		groups = None
	else:
		group_array = np.array(group_dict.values())
		group_array.sort()
		groups = group_array.tolist()
	if (is_mltl and avg == 'all'):
		aucs_df = pd.DataFrame([micro_roc_aucs, macro_roc_aucs, prc_aucs], index=['Micro ROC AUC', 'Macro ROC AUC', 'PRC AUC'], columns=PL_NAMES)
		if (cfg_param.setdefault('plot_roc', True)):
			plot.plot_roc(micro_roc_data, micro_roc_labels, groups=groups, fname='micro_roc%s'%lbidstr, plot_cfg=common_cfg)
			plot.plot_roc(macro_roc_data, macro_roc_labels, groups=groups, fname='macro_roc%s'%lbidstr, plot_cfg=common_cfg)
	else:
		aucs_df = pd.DataFrame([roc_aucs, prc_aucs], index=['ROC AUC', 'PRC AUC'], columns=PL_NAMES)
		if (cfg_param.setdefault('plot_roc', True)):
			plot.plot_roc(roc_data, roc_labels, groups=groups, fname='roc%s'%lbidstr, plot_cfg=common_cfg)
	if (cfg_param.setdefault('plot_prc', True)):
		plot.plot_prc(prc_data, prc_labels, groups=groups, fname='prc%s'%lbidstr, plot_cfg=common_cfg)
	if (cfg_param.setdefault('save_auc', False)):
		aucs_df.to_excel('auc%s.xlsx' % lbidstr)
	filt_num, clf_num = len(FILT_NAMES), len(CLF_NAMES)
	if (cfg_param.setdefault('plot_metric', False)):
		for mtrc in metric_idx:
			mtrc_avg_list, mtrc_std_list = [[] for i in range(2)]
			if (global_param['comb']):
				mtrc_avg = perf_avg_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				mtrc_std = perf_std_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=PL_NAMES, labels=None, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clf_ft%s' % (mtrc.replace(' ', '_').lower(), lbidstr), plot_cfg=common_cfg)
			else:
				for i in xrange(filt_num):
					offset = i * clf_num
					mtrc_avg_list.append(perf_avg_df.ix[mtrc,offset:offset+clf_num].as_matrix().reshape((1,-1)))
					mtrc_std_list.append(perf_std_df.ix[mtrc,offset:offset+clf_num].as_matrix().reshape((1,-1)))
				mtrc_avg = np.concatenate(mtrc_avg_list)
				mtrc_std = np.concatenate(mtrc_std_list)
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=CLF_NAMES, labels=FILT_NAMES, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clf_ft%s' % (mtrc.replace(' ', '_').lower(), lbidstr), plot_cfg=common_cfg)

	
# Cross validation
def cross_validate(X, Y, model_iter, model_param={}, avg='micro', kfold=5, cfg_param={}, split_param={}, global_param={}, lbid=''):
	global common_cfg
	FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET = model_param['glb_filtnames'], model_param['glb_clfnames'], global_param['pl_names'], global_param['pl_set']
	lbidstr = ('_' + (str(lbid) if lbid != -1 else 'all')) if lbid is not None and lbid != '' else ''
	to_hdf, hdf5_fpath = cfg_param.setdefault('to_hdf', False), 'crsval_dataset%s.h5' % lbidstr if cfg_param.setdefault('hdf5_fpath', 'crsval_dataset%s.h5' % lbidstr) is None else cfg_param['hdf5_fpath']
	
	# Format the data
	if (type(X) == list):
		assert all([len(x) == len(X[0]) for x in X[1:]])
		X = [pd.DataFrame(x) if (type(x) != pd.io.parsers.TextFileReader and type(x) != pd.DataFrame) else x for x in X]
		X = [pd.concat(x) if (type(x) == pd.io.parsers.TextFileReader and not to_hdf) else x for x in X]
	else:
		if (type(X) != pd.io.parsers.TextFileReader and type(X) != pd.DataFrame):
			X = pd.DataFrame(X) if type(X) != HDF5Matrix else X
		X = pd.concat(X) if (type(X) == pd.io.parsers.TextFileReader and not to_hdf) else X
	if (type(Y) != pd.io.parsers.TextFileReader and type(Y) != pd.DataFrame):
		Y = pd.DataFrame(Y) if (type(Y) == pd.io.parsers.TextFileReader and not to_hdf) else Y
	if (type(Y) != HDF5Matrix):
		Y_mt = Y.as_matrix().reshape((Y.shape[0],)) if (len(Y.shape) == 1 or Y.shape[1] == 1) else Y.as_matrix()
	else:
		Y_mt = Y
	is_mltl = True if len(Y_mt.shape) > 1 and Y_mt.shape[1] > 1 or 2 in Y_mt else False
		
	print 'Benchmark is starting...'
	mean_fpr = np.linspace(0, 1, 100)
	mean_recall = np.linspace(0, 1, 100)
	xdf = X[0] if type(X)==list else X
	if (len(split_param) == 0):
		if (type(xdf) != HDF5Matrix):
			kf = list(KFold(n_splits=kfold, shuffle=True, random_state=0).split(xdf, Y_mt)) if (len(Y_mt.shape) == 1) else list(KFold(n_splits=kfold, shuffle=True, random_state=0).split(xdf, Y_mt[:,0].reshape((Y_mt.shape[0],))))
		else:
			kf = list(KFold(n_splits=kfold, shuffle=False, random_state=0).split(xdf[:], Y_mt[:])) if (len(Y_mt.shape) == 1) else list(KFold(n_splits=kfold, shuffle=False, random_state=0).split(xdf[:], Y_mt[:].reshape((-1,)))) # HDF5Matrix is not working in shuffle indices
	else:
		split_param['shuffle'] = True if type(xdf) != HDF5Matrix else False
		# To-do: implement the split method for multi-label data
		if (split_param.has_key('train_size') and split_param.has_key('test_size')):
			kf = list(StratifiedShuffleSplit(n_splits=kfold, train_size=split_param['train_size'], test_size=split_param['test_size'], random_state=0).split(xdf, Y_mt)) if (len(Y_mt.shape) == 1) else list(StratifiedShuffleSplit(n_splits=kfold, train_size=split_param['train_size'], test_size=split_param['test_size'], random_state=0).split(xdf, Y_mt[:,0].reshape((Y_mt.shape[0],))))
		else:
			kf = list(StratifiedKFold(n_splits=kfold, shuffle=split_param.setdefault('shuffle', True), random_state=0).split(xdf, Y_mt)) if (len(Y_mt.shape) == 1) else list(StratifiedKFold(n_splits=kfold, shuffle=split_param.setdefault('shuffle', True), random_state=0).split(xdf, Y_mt[:,0].reshape((Y_mt.shape[0],))))
	crsval_results, crsval_tpreds, crsval_povl, crsval_spearman, crsval_kendalltau, crsval_pearson = [[] for i in range(6)]
	crsval_roc, crsval_prc, crsval_featw, crsval_subfeatw = [{} for i in range(4)]
	# for i, (train_idx, test_idx) in enumerate(kf):
	for i, X_train, X_test, Y_train, Y_test, train_idx_df, test_idx_df in kf2data(kf, X, Y_mt, to_hdf=to_hdf, hdf5_fpath=hdf5_fpath):
		del PL_NAMES[:]
		PL_SET.clear()
		print '\n' + '-' * 80 + '\n' + '%s time validation' % imath.ordinal(i+1) + '\n' + '-' * 80 + '\n'
		if (cfg_param.setdefault('save_crsval_idx', False)):
			io.write_df(train_idx_df, 'train_idx_crsval_%s%s.npz' % (i, lbidstr), with_idx=True)
			io.write_df(test_idx_df, 'test_idx_crsval_%s%s.npz' % (i, lbidstr), with_idx=True)
		if (cfg_param.setdefault('npg_ratio', None) is not None):
			npg_ratio = cfg_param['npg_ratio']
			Y_train = np.array(Y_train) # HDF5Matrix is not working in matrix slicing and boolean operation
			y = Y_train[:,0] if (len(Y_train.shape) > 1) else Y_train
			if (1.0 * np.abs(y).sum() / Y_train.shape[0] < 1.0 / (npg_ratio + 1)):
				all_true = np.arange(Y_train.shape[0])[y > 0].tolist()
				all_false = np.arange(Y_train.shape[0])[y <= 0].tolist()
				true_id = np.random.choice(len(all_true), size=int(1.0 / npg_ratio * len(all_false)), replace=True)
				true_idx = [all_true[i] for i in true_id]
				all_train_idx = sorted(set(true_idx + all_false))
				X_train = [x.iloc[all_train_idx] if type(x) != HDF5Matrix else x[all_train_idx] for x in X_train] if (type(X_train) is list) else X_train.iloc[all_train_idx] if type(x) != HDF5Matrix else X_train[all_train_idx]
				Y_train = Y_train[all_train_idx,:] if (len(Y_train.shape) > 1) else Y_train[all_train_idx]
		results, preds = [[] for x in range(2)]
		Y_test = np.array(Y_test)
		for vars in model_iter(**model_param):
			if (global_param['comb']):
				mdl_name, mdl = [vars[x] for x in range(2)]
			else:
				filt_name, filter, clf_name, clf= [vars[x] for x in range(4)]
			print '#' * 80
			# Assemble a pipeline
			if ('filter' in locals() and filter != None):
				model_name = '%s [Ft Filt] & %s [CLF]' % (filt_name, clf_name)
				pipeline = Pipeline([('featfilt', clone(filter)), ('clf', clf)])
			elif ('clf' in locals() and clf != None):
				model_name = '%s [CLF]' % clf_name
				pipeline = Pipeline([('clf', clf)])
			else:
				model_name = mdl_name
				pipeline = mdl
			if (model_name in PL_SET): continue
			PL_NAMES.append(model_name)
			PL_SET.add(model_name)
			print model_name
			# Benchmark results
			bm_results = benchmark(pipeline, X_train, Y_train, X_test, Y_test, mltl=is_mltl, signed=global_param.setdefault('signed', True if np.where(Y_mt<0)[0].shape[0]>0 else False), average=avg)
			# Clear the model environment (e.g. GPU resources)
			del pipeline
			# if (type(pipeline) is Pipeline):
				# for cmpn in pipeline.named_steps.values():
					# if (getattr(cmpn, "clear", None)): cmpn.clear()
			# else:
				# if (getattr(pipeline, "clear", None)):
					# pipeline.clear()
			# Obtain the results
			if (is_mltl and avg == 'all'):
				results.append([bm_results[x] for x in ['accuracy', 'micro-precision', 'micro-recall', 'micro-fscore', 'macro-precision', 'macro-recall', 'macro-fscore', 'train_time', 'test_time']])
			else:
				results.append([bm_results[x] for x in ['accuracy', 'precision', 'recall', 'fscore', 'train_time', 'test_time']])
			preds.append(bm_results['pred_lb'])
			if (cfg_param.setdefault('save_crsval_pred', False)):
				io.write_npz(dict(pred_lb=bm_results['pred_lb'], true_lb=Y_test), 'pred_crsval_%s_%s%s' % (i, model_name.replace(' ', '_').lower(), lbidstr))
			if (is_mltl and avg == 'all'):
				micro_id, macro_id = '-'.join([model_name,'micro']), '-'.join([model_name,'macro'])
				crsval_roc[micro_id] = crsval_roc.setdefault(micro_id, 0) + np.interp(mean_fpr, bm_results['micro-roc'][0], bm_results['micro-roc'][1])
				crsval_roc[macro_id] = crsval_roc.setdefault(macro_id, 0) + np.interp(mean_fpr, bm_results['macro-roc'][0], bm_results['macro-roc'][1])
			else:
				crsval_roc[model_name] = crsval_roc.setdefault(model_name, 0) + np.interp(mean_fpr, bm_results['roc'][0], bm_results['roc'][1])
			crsval_prc[model_name] = crsval_prc.setdefault(model_name, 0) + np.interp(mean_recall, bm_results['prc'][0], bm_results['prc'][1])
			for k, v in bm_results['feat_w'].iteritems():
				key = '%s_%s_%s' % (model_name, k[0], k[1])
				crsval_featw.setdefault(key, []).append(v)
			for k, v in bm_results['sub_feat_w'].iteritems():
				key = '%s_%s_%s' % (model_name, k[0], k[1])
				crsval_subfeatw.setdefault(key, []).append(v)
			print '\n'
		# Cross validation results
		crsval_results.append(results)
		# Prediction overlap
		if (True if len(Y_mt.shape) > 1 and Y_mt.shape[1] > 1 else False):
			preds_mt = np.column_stack([x.ravel() for x in preds])
		else:
			preds_mt = np.column_stack(preds)
		preds.append(Y_test)
		tpreds_mt = np.column_stack([x.ravel() for x in preds])
		crsval_tpreds.append(tpreds_mt)
		crsval_povl.append(pred_ovl(preds_mt, Y_test))
		# Spearman's rank correlation
		crsval_spearman.append(stats.spearmanr(tpreds_mt))
		# Kendall rank correlation
		# crsval_kendalltau.append(stats.kendalltau(preds_mt)) 
		# Pearson correlation
		# crsval_pearson.append(stats.pearsonr(preds_mt))
		del X_train, X_test, Y_train, Y_test
		print '\n'
	perf_avg = np.array(crsval_results).mean(axis=0)
	perf_std = np.array(crsval_results).std(axis=0)
	povl_avg = np.array(crsval_povl).mean(axis=0).round()
	spmnr_avg = np.array([crsp[0] for crsp in crsval_spearman]).mean(axis=0)
	spmnr_pval = np.array([crsp[1] for crsp in crsval_spearman]).mean(axis=0)
	# kndtr_avg = np.array([crkdt[0] for crkdt in crsval_kendalltau).mean(axis=0)
	# kndtr_pval = np.array([crkdt[1] for crkdt in crsval_kendalltau]).mean(axis=0)
	# prsnr_avg = np.array([crprs[0] for crprs in crsval_pearson).mean(axis=0)
	# prsnr_pval = np.array([crprs[1] for crprs in crsval_pearson]).mean(axis=0)
	
	## Save performance data
	if (is_mltl and avg == 'all'):
		metric_idx = ['Accuracy', 'Micro Precision', 'Micro Recall', 'Micro F score', 'Macro Precision', 'Macro Recall', 'Macro F score', 'Train time', 'Test time']
	else:
		metric_idx = ['Accuracy', 'Precision', 'Recall', 'F score', 'Train time', 'Test time']
	perf_avg_df = pd.DataFrame(perf_avg.T, index=metric_idx, columns=PL_NAMES)
	perf_std_df = pd.DataFrame(perf_std.T, index=metric_idx, columns=PL_NAMES)
	povl_idx = [' & '.join(x) for x in imath.subset(PL_NAMES, min_crdnl=1)]
	povl_avg_df = pd.DataFrame(povl_avg, index=povl_idx, columns=['pred_ovl', 'tpred_ovl'])
	spmnr_avg_df = pd.DataFrame(spmnr_avg, index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	spmnr_pval_df = pd.DataFrame(spmnr_pval, index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	if (cfg_param.setdefault('save_tpred', True)):
		io.write_npz(crsval_tpreds, 'tpred_clf%s' % lbidstr)
	if (cfg_param.setdefault('save_perf_avg', True)):
		perf_avg_df.to_excel('perf_avg_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_perf_avg_npz', False)):
		io.write_df(perf_avg_df, 'perf_avg_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_perf_std', True)):
		perf_std_df.to_excel('perf_std_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_perf_std_npz', False)):
		io.write_df(perf_std_df, 'perf_std_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_povl', False)):
		povl_avg_df.to_excel('cpovl_avg_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_povl_npz', False)):
		io.write_df(povl_avg_df, 'povl_avg_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_avg', False)):
		spmnr_avg_df.to_excel('spmnr_avg_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_avg_npz', False)):
		io.write_df(spmnr_avg_df, 'spmnr_avg_clf%s.npz' % lbidstr, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_pval', False)):
		spmnr_pval_df.to_excel('spmnr_pval_clf%s.xlsx' % lbidstr)
	if (cfg_param.setdefault('save_spmnr_pval_npz', False)):
		io.write_df(spmnr_pval_df, 'spmnr_pval_clf%s.npz' % lbidstr, with_idx=True)
	# Feature importances
	try:
		save_featw(xdf.columns.values if type(xdf) != HDF5Matrix else np.arange(xdf.shape[1]), crsval_featw, crsval_subfeatw, cfg_param=cfg_param, lbid=lbid)
	except Exception as e:
		print e
	
	## Plot figures
	if (is_mltl and avg == 'all'):
		micro_roc_data, micro_roc_labels, micro_roc_aucs, macro_roc_data, macro_roc_labels, macro_roc_aucs = [[] for i in range(6)]
	else:
		roc_data, roc_labels, roc_aucs = [[] for i in range(3)]
	prc_data, prc_labels, prc_aucs = [[] for i in range(3)]
	for pl in PL_NAMES:
		if (is_mltl and avg == 'all'):
			micro_id, macro_id = '-'.join([pl,'micro']), '-'.join([pl,'macro'])
			micro_mean_tpr, macro_mean_tpr = crsval_roc[micro_id], crsval_roc[macro_id]
			micro_mean_tpr, macro_mean_tpr = micro_mean_tpr / len(kf), macro_mean_tpr / len(kf)
			micro_roc_auc = metrics.auc(mean_fpr, micro_mean_tpr)
			macro_roc_auc = metrics.auc(mean_fpr, macro_mean_tpr)
			micro_roc_data.append([mean_fpr, micro_mean_tpr])
			micro_roc_aucs.append(micro_roc_auc)
			micro_roc_labels.append('%s (AUC=%0.2f)' % (pl, micro_roc_auc))
			macro_roc_data.append([mean_fpr, macro_mean_tpr])
			macro_roc_aucs.append(macro_roc_auc)
			macro_roc_labels.append('%s (AUC=%0.2f)' % (pl, macro_roc_auc))
		else:
			mean_tpr = crsval_roc[pl]
			mean_tpr /= len(kf)
			mean_roc_auc = metrics.auc(mean_fpr, mean_tpr)
			roc_data.append([mean_fpr, mean_tpr])
			roc_aucs.append(mean_roc_auc)
			roc_labels.append('%s (AUC=%0.2f)' % (pl, mean_roc_auc))
		mean_prcn = crsval_prc[pl]
		mean_prcn /= len(kf)
		mean_prc_auc = metrics.auc(mean_recall, mean_prcn)
		prc_data.append([mean_recall, mean_prcn])
		prc_aucs.append(mean_prc_auc)
		prc_labels.append('%s (AUC=%0.2f)' % (pl, mean_prc_auc))
	group_dict = {}
	for i, pl in enumerate(PL_NAMES):
		group_dict.setdefault(tuple(set(difflib.get_close_matches(pl, PL_NAMES))), []).append(i)
	if (not cfg_param.setdefault('group_by_name', False) or len(group_dict) == len(PL_NAMES)):
		groups = None
	else:
		group_array = np.array(group_dict.values())
		group_array.sort()
		groups = group_array.tolist()
	if (is_mltl and avg == 'all'):
		aucs_df = pd.DataFrame([micro_roc_aucs, macro_roc_aucs, prc_aucs], index=['Micro ROC AUC', 'Macro ROC AUC', 'PRC AUC'], columns=PL_NAMES)
		if (cfg_param.setdefault('plot_roc', True)):
			plot.plot_roc(micro_roc_data, micro_roc_labels, groups=groups, fname='micro_roc%s'%lbidstr, plot_cfg=common_cfg)
			plot.plot_roc(macro_roc_data, macro_roc_labels, groups=groups, fname='macro_roc%s'%lbidstr, plot_cfg=common_cfg)
	else:
		aucs_df = pd.DataFrame([roc_aucs, prc_aucs], index=['ROC AUC', 'PRC AUC'], columns=PL_NAMES)
		if (cfg_param.setdefault('plot_roc', True)):
			plot.plot_roc(roc_data, roc_labels, groups=groups, fname='roc%s'%lbidstr, plot_cfg=common_cfg)
	if (cfg_param.setdefault('plot_prc', True)):
		plot.plot_prc(prc_data, prc_labels, groups=groups, fname='prc%s'%lbidstr, plot_cfg=common_cfg)
	if (cfg_param.setdefault('save_auc', False)):
		aucs_df.to_excel('auc%s.xlsx' % lbidstr)
	filt_num, clf_num = len(FILT_NAMES), len(CLF_NAMES)
	if (cfg_param.setdefault('plot_metric', False)):
		for mtrc in metric_idx:
			mtrc_avg_list, mtrc_std_list = [[] for i in range(2)]
			if (global_param['comb']):
				mtrc_avg = perf_avg_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				mtrc_std = perf_std_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=PL_NAMES, labels=None, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clf_ft%s' % (mtrc.replace(' ', '_').lower(), lbidstr), plot_cfg=common_cfg)
			else:
				for i in xrange(filt_num):
					offset = i * clf_num
					mtrc_avg_list.append(perf_avg_df.ix[mtrc,offset:offset+clf_num].as_matrix().reshape((1,-1)))
					mtrc_std_list.append(perf_std_df.ix[mtrc,offset:offset+clf_num].as_matrix().reshape((1,-1)))
				mtrc_avg = np.concatenate(mtrc_avg_list)
				mtrc_std = np.concatenate(mtrc_std_list)
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=CLF_NAMES, labels=FILT_NAMES, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clf_ft%s' % (mtrc.replace(' ', '_').lower(), lbidstr), plot_cfg=common_cfg)
				
				
def tune_param(mdl_name, mdl, X, Y, rdtune, params, mltl=False, avg='micro', n_jobs=-1):
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
			for p_name, p_val in p_option.iteritems():
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
		for k, v in p_option.iteritems():
			idx[dim_names[k]] = dim_vals[k][v]
		score_avg_cube[tuple(idx)] = score_avg_list[i]
		score_std_cube[tuple(idx)] = score_std_list[i]
	return grid.best_params_, grid.best_score_, score_avg_cube, score_std_cube, dim_names, dim_vals

	
def tune_param_optunity(mdl_name, mdl, X, Y, scoring='f1', optfunc='max', solver='particle swarm', params={}, mltl=False, avg='micro', n_jobs=-1):
	import optunity
	struct, param_space, folds, n_iter = [params.setdefault(k, None) for k in ['struct', 'param_space', 'folds', 'n_iter']]
	ext_params = dict.fromkeys(param_space.keys()) if (not struct) else dict.fromkeys(params.setdefault('param_names', []))
	kwargs = dict([('num_iter', n_iter), ('num_folds', folds)]) if (type(folds) is int) else dict([('num_iter', n_iter), ('num_folds', folds.get_n_splits()), ('folds', [list(folds.split(X))] * n_iter)])
	@optunity.cross_validated(x=X, y=Y, **kwargs)
	def perf(x_train, y_train, x_test, y_test, **ext_params):
		mdl.fit(x_train, y_train)
		if (scoring == 'roc'):
			preds = get_score(mdl, x_test, mltl)
			if (mltl):
				import metric as imetric
				return imetric.mltl_roc(y_test, preds, average=avg)
		else:
			preds = mdl.predict(x_test)
		score_func = getattr(optunity, scoring) if (hasattr(optunity, scoring)) else None
		score_func = getattr(metrics, scoring+'_score') if (score_func is None and hasattr(metrics, scoring+'_score')) else score_func
		if (score_func is None):
			print 'Score function %s is not supported!' % scoring
			sys.exit(1)
		return score_func(y_test, preds, average=avg)
	if (optfunc == 'max'):
		config, info, _ = optunity.maximize(perf, num_evals=n_iter, solver_name=solver, pmap=optunity.parallel.create_pmap(n_jobs), **param_space) if (not struct) else optunity.maximize_structured(perf, search_space=param_space, num_evals=n_iter, pmap=optunity.parallel.create_pmap(n_jobs))
	elif (optfunc == 'min'):
		config, info, _ = optunity.minimize(perf, num_evals=n_iter, solver_name=solver, pmap=optunity.parallel.create_pmap(n_jobs), **param_space) if (not struct) else optunity.minimize_structured(perf, search_space=param_space, num_evals=n_iter, pmap=optunity.parallel.create_pmap(n_jobs))
	print("The best parameters of [%s] are %s, with a score of %0.3f" % (mdl_name, config, info.optimum))
	cl_df = optunity.call_log2dataframe(info.call_log)
	cl_df.to_csv('call_log.csv')
	# Store all the parameter candidates into a dictionary of list
	param_grid = dict([(x, sorted(set(cl_df[x]))) for x in cl_df.columns if x != 'value'])
	param_names = param_grid.keys()
	# Index the parameter names and valules
	dim_names = dict([(k, i) for i, k in enumerate(param_names)])
	dim_vals = {}
	for pn in dim_names.keys():
		dim_vals[pn] = dict([(k, i) for i, k in enumerate(param_grid[pn])])
	# Create data cube
	score_avg_cube = np.ndarray(shape=[len(param_grid[k]) for k in param_names], dtype='float') * np.nan
	score_std_cube = np.ndarray(shape=[len(param_grid[k]) for k in param_names], dtype='float') * np.nan
	# Calculate the score list
	score_avg_list = cl_df['value']
	score_std_list = np.zeros_like(cl_df['value'])
	# Fill in the data cube
	for i, p_option in cl_df[param_names].iterrows():
		idx = np.zeros((len(dim_names),), dtype='int')
		for k, v in p_option.iteritems():
			idx[dim_names[k]] = dim_vals[k][v]
		score_avg_cube[tuple(idx)] = score_avg_list[i]
		score_std_cube[tuple(idx)] = score_std_list[i]
	return config, info.optimum, score_avg_cube, score_std_cube, dim_names, dim_vals
	
	
def analyze_param(param_name, score_avg, score_std, dim_names, dim_vals, best_params):
    best_param_idx = dict([(k, (dim_names[k], dim_vals[k][best_params[k]])) for k in dim_names.keys()])
    best_param_idx[param_name] = (best_param_idx[param_name][0], slice(0, score_avg.shape[dim_names[param_name]]))
    _, slicing = zip(*func.sorted_tuples(best_param_idx.values(), key_idx=0))
    param_vals, _ = zip(*func.sorted_dict(dim_vals[param_name], key='value'))
    return np.array(param_vals), score_avg[slicing], score_std[slicing]