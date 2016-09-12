#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: txtclf.py
# Author: Shankai Yan
# E-mail: sk.yan@mY.cityu.edu.hk
# Created Time: 2016-07-05 14:39:18
###########################################################################
#

import difflib
from time import time

import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd

from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics

import util.io as io
import util.math as imath
import util.func as func
import util.plot as plot

common_cfg = {}


def init(plot_cfg={}, plot_common={}):
	if (len(plot_cfg) > 0 and plot_cfg['MON'] is not None):
		plot.MON = plot_cfg['MON']
	global common_cfg
	if (len(plot_common) > 0):
		common_cfg = plot_common


# Benchmark
def benchmark(pipeline, X_train, Y_train, X_test, Y_test, mltl=False, average='micro'):
	print '+' * 80
	print 'Training Model: '
	print pipeline
	t0 = time()
	pipeline.fit(X_train, Y_train)
	train_time = time() - t0
	print 'train time: %0.3fs' % train_time

	t0 = time()
	pred = pipeline.predict(X_test)
	test_time = time() - t0
	print '+' * 80
	print 'Testing: '
	print 'test time: %0.3fs' % test_time

	accuracy = metrics.accuracy_score(Y_test, pred)
	print 'accuracy: %0.3f' % accuracy
	if (average == 'all'):
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
		precision = metrics.precision_score(Y_test, pred, average=average if mltl else 'binary')
		print 'precision: %0.3f' % precision
		recall = metrics.recall_score(Y_test, pred, average=average if mltl else 'binary')
		print 'recall: %0.3f' % recall
		fscore = metrics.fbeta_score(Y_test, pred, beta=1, average=average if mltl else 'binary')
		print 'fscore: %0.3f' % fscore

	print 'classification report:'
	print metrics.classification_report(Y_test, pred)

	print 'confusion matrix:'
	if (mltl):
		pass
	else:
		print metrics.confusion_matrix(Y_test, pred)
	print '+' * 80

	if ((isinstance(pipeline.named_steps['clf'], OneVsRestClassifier) and hasattr(pipeline.named_steps['clf'].estimators_[0], 'predict_proba')) or (not isinstance(pipeline.named_steps['clf'], OneVsRestClassifier) and hasattr(pipeline, 'predict_proba'))):
		if (mltl):
			scores = pipeline.predict_proba(X_test)
		else:
			scores = pipeline.predict_proba(X_test)[:, 1]
	elif (hasattr(pipeline, 'decision_function')):
		scores = pipeline.decision_function(X_test)
	else:
		print 'Neither probability estimate nor decision function is supported in the classification model! ROC and PRC figures will be invalid.'
		scores = [0] * Y_test.shape[0]
	if (mltl):
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

	print 'Training and Testing X shape: (%s) (%s)' % (','.join([str(x) for x in X_train.shape]), ','.join([str(x) for x in X_test.shape]))
	feat_w_dict, sub_feat_w = [{} for i in range(2)]
	filt_feat_idx = feature_idx = np.arange(X_train.shape[1])
	for component in ('featfilt', 'clf'):
		if (pipeline.named_steps.has_key(component)):
			if (hasattr(pipeline.named_steps[component], 'estimators_')):
				for i, estm in enumerate(pipeline.named_steps[component].estimators_):
					filt_subfeat_idx = feature_idx[:]
					if (hasattr(estm, 'get_support')):
						filt_subfeat_idx = feature_idx[estm.get_support()]
					for measure in ('feature_importances_', 'coef_', 'scores_'):
						if (hasattr(estm, measure)):
							filt_subfeat_w = getattr(estm, measure)
							subfeat_w = (filt_subfeat_w.min() - 1) * np.ones_like(feature_idx)
		#					subfeat_w[filt_subfeat_idx] = normalize(estm.feature_importances_.reshape(-1, 1), norm='l1')
							subfeat_w[filt_subfeat_idx] = filt_subfeat_w.reshape(-1, 1)
							# print 'Sub FI shape: (%s)' % ','.join([str(x) for x in filt_subfeat_w.shape])
							# print 'Feature Importance inside %s Ensemble Method: %s' % (component, filt_subfeat_w)
							sub_feat_w[(component, i)] = subfeat_w
			if (hasattr(component, 'get_support')):
				filt_feat_idx = feature_idx[component.get_support()]
			for measure in ('feature_importances_', 'coef_', 'scores_'):
				if (hasattr(pipeline.named_steps[component], measure)):
					filt_feat_w = getattr(pipeline.named_steps[component], measure)
#					print '*' * 80 + '\n%s\n'%filt_feat_w + '*' * 80
					feat_w = (filt_feat_w.min() - 1) * np.ones_like(feature_idx)
#					feat_w[filt_feat_idx] = normalize(filt_feat_w.reshape(-1, 1), norm='l1')
					feat_w[filt_feat_idx] = filt_feat_w.reshape(-1, 1)
#					print '*' * 80 + '\n%s\n'%feat_w + '*' * 80
					feat_w_dict[(component, measure)] = feat_w
					print 'FI shape: (%s)' % ','.join([str(x) for x in feat_w_dict[(component, measure)].shape])
					print 'Sample 10 Feature from %s.%s: %s' % (component, measure, feat_w[feat_w > 0][:10])
#					print 'Feature Importance from %s.%s: %s' % (component, measure, feat_w)
	if (average == 'all'):
		return {'accuracy':accuracy, 'micro-precision':micro_precision, 'micro-recall':micro_recall, 'micro-fscore':micro_fscore, 'macro-precision':macro_precision, 'macro-recall':macro_recall, 'macro-fscore':macro_fscore, 'train_time':train_time, 'test_time':test_time, 'micro-roc':micro_roc, 'macro-roc':macro_roc, 'prc':prc, 'feat_w':feat_w_dict, 'sub_feat_w':sub_feat_w, 'pred_lb':pred}
	else:
		return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'fscore':fscore, 'train_time':train_time, 'test_time':test_time, 'roc':roc, 'prc':prc, 'feat_w':feat_w_dict, 'sub_feat_w':sub_feat_w, 'pred_lb':pred}
	print '\n'


# Calculate the venn digram overlaps
def pred_ovl(preds, pred_true=None, axis=1):
	if (axis == 0):
		preds = preds.T
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
	
	
# Cross validation
def cross_validate(X, Y, lbid, model_iter, model_param={}, avg='micro', kfold=5, cfg_param={}, split_param={}, global_param={}):
	global common_cfg
	FILT_NAMES, CLF_NAMES, PL_NAMES, PL_SET = model_param['glb_filtnames'], model_param['glb_clfnames'], global_param['pl_names'], global_param['pl_set']
	print 'Benchmark is starting...'
	mean_fpr = np.linspace(0, 1, 100)
	mean_recall = np.linspace(0, 1, 100)
	if (len(split_param) == 0):
		kf = KFold(Y.shape[0], n_folds=kfold, shuffle=True, random_state=0)
	else:
		if (split_param.has_key('train_size') and split_param.has_key('test_size')):
			kf = StratifiedShuffleSplit(y, n_iter=kfold, train_size=split_param['train_size'], test_size=split_param['test_size'], random_state=0)
		else:
			kf = StratifiedKFold(y, n_folds=kfold, shuffle=split_param.setdefault('shuffle', True), random_state=0)
	crsval_results, crsval_povl, crsval_spearman, crsval_kendalltau, crsval_pearson = [[] for i in range(5)]
	crsval_roc, crsval_prc, crsval_featw, crsval_subfeatw = [{} for i in range(4)]
	for i, (train_idx, test_idx) in enumerate(kf):
		print '\n' + '-' * 80 + '\n' + '%s time validation' % imath.ordinal(i+1) + '\n' + '-' * 80 + '\n'
		X_train, X_test = X.iloc[train_idx,:].as_matrix(), X.iloc[test_idx,:].as_matrix()
		Y_train, Y_test = Y[train_idx], Y[test_idx]
		train_idx_df, test_idx_df = pd.DataFrame(np.arange(X_train.shape[0]), index=X.index[train_idx]), pd.DataFrame(np.arange(X_test.shape[0]), index=X.index[test_idx])
		if (cfg_param.setdefault('save_crsval_idx', False)):
			io.write_df(train_idx_df, 'train_idx_crsval_%s_%s.npz' % (i, lbid), with_idx=True)
			io.write_df(test_idx_df, 'test_idx_crsval_%s_%s.npz' % (i, lbid), with_idx=True)
		results, preds = [[] for x in range(2)]
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
			if (model_name not in PL_SET):
				PL_NAMES.append(model_name)
				PL_SET.add(model_name)
			print model_name
			# Benchmark results
			bm_results = benchmark(pipeline, X_train, Y_train, X_test, Y_test, mltl=True if len(Y.shape) > 1 and Y.shape[1] > 1 else False, average=avg)
			if (avg == 'all'):
				results.append([bm_results[x] for x in ['accuracy', 'micro-precision', 'micro-recall', 'micro-fscore', 'macro-precision', 'macro-recall', 'macro-fscore', 'train_time', 'test_time']])
			else:
				results.append([bm_results[x] for x in ['accuracy', 'precision', 'recall', 'fscore', 'train_time', 'test_time']])
			preds.append(bm_results['pred_lb'])
			if (cfg_param.setdefault('save_crsval_pred', False)):
				io.write_npz(dict(pred_lb=bm_results['pred_lb'], true_lb=Y_test), 'pred_crsval_%s_%s_%s'%(i,model_name.replace(' ', '_').lower(),lbid))
			if (avg == 'all'):
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
			# if ('RandomForest' in model_name):
				# print pipeline.named_steps['clf'].get_params()
			print '\n'
		# Cross validation results
		crsval_results.append(results)
		# Prediction overlap
		if (True if len(Y.shape) > 1 and Y.shape[1] > 1 else False):
			preds_mt = np.column_stack([x.ravel() for x in preds])
		else:
			preds_mt = np.column_stack(preds)
		preds.append(Y_test)
		tpreds_mt = np.column_stack([x.ravel() for x in preds])
		crsval_povl.append(pred_ovl(preds_mt, Y_test))
		# Spearman's rank correlation
		crsval_spearman.append(stats.spearmanr(tpreds_mt))
		# Kendall rank correlation
#		crsval_kendalltau.append(stats.kendalltau(preds_mt)[0]) 
		# Pearson correlation
#		crsval_pearson.append(stats.pearsonr(preds_mt)[0])
		del X_train, X_test, Y_train, Y_test
		print '\n'
	perf_avg = np.array(crsval_results).mean(axis=0)
	perf_std = np.array(crsval_results).std(axis=0)
	povl_avg = np.array(crsval_povl).mean(axis=0).round()
	spmnr_avg = np.array([crsp[0] for crsp in crsval_spearman]).mean(axis=0)
	spmnr_pval = np.array([crsp[1] for crsp in crsval_spearman]).mean(axis=0)
#	kndtr_avg = np.array(crsval_kendalltau).mean(axis=0)
#	prsnr_avg = np.array(crsval_pearson).mean(axis=0)
	
	## Save performance data
	if (avg == 'all'):
		metric_idx = ['Accuracy', 'Micro Precision', 'Micro Recall', 'Micro F score', 'Macro Precision', 'Macro Recall', 'Macro F score', 'Train time', 'Test time']
	else:
		metric_idx = ['Accuracy', 'Precision', 'Recall', 'F score', 'Train time', 'Test time']
	perf_avg_df = pd.DataFrame(perf_avg.T, index=metric_idx, columns=PL_NAMES)
	perf_std_df = pd.DataFrame(perf_std.T, index=metric_idx, columns=PL_NAMES)
	povl_idx = [' & '.join(x) for x in imath.subset(PL_NAMES, min_crdnl=1)]
	povl_avg_df = pd.DataFrame(povl_avg, index=povl_idx, columns=['pred_ovl', 'tpred_ovl'])
	spmnr_avg_df = pd.DataFrame(spmnr_avg, index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	spmnr_pval_df = pd.DataFrame(spmnr_pval, index=PL_NAMES+['Annotations'], columns=PL_NAMES+['Annotations'])
	if (cfg_param.setdefault('save_perf_avg', False)):
		perf_avg_df.to_excel('perf_avg_%s.xlsx' % lbid)
	if (cfg_param.setdefault('save_perf_avg_npz', False)):
		io.write_df(perf_avg_df, 'perf_avg_%s.npz' % lbid, with_idx=True)
	if (cfg_param.setdefault('save_perf_std', False)):
		perf_std_df.to_excel('perf_std_%s.xlsx' % lbid)
	if (cfg_param.setdefault('save_perf_std_npz', False)):
		io.write_df(perf_std_df, 'perf_std_%s.npz' % lbid, with_idx=True)
	if (cfg_param.setdefault('save_povl', False)):
		povl_avg_df.to_excel('cpovl_avg_%s.xlsx' % lbid)
	if (cfg_param.setdefault('save_povl_npz', False)):
		io.write_df(povl_avg_df, 'povl_avg_%s.npz' % lbid, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_avg', False)):
		spmnr_avg_df.to_excel('spmnr_avg_%s.xlsx' % lbid)
	if (cfg_param.setdefault('save_spmnr_avg_npz', False)):
		io.write_df(spmnr_avg_df, 'spmnr_avg_%s.npz' % lbid, with_idx=True)
	if (cfg_param.setdefault('save_spmnr_pval', False)):
		spmnr_pval_df.to_excel('spmnr_pval_%s.xlsx' % lbid)
	if (cfg_param.setdefault('save_spmnr_pval_npz', False)):
		io.write_df(spmnr_pval_df, 'spmnr_pval_%s.npz' % lbid, with_idx=True)
	# Feature importances
	for k, v in crsval_featw.iteritems():
		feat_w_mt = np.column_stack(v)
		mms = MinMaxScaler()
		feat_w_mt = mms.fit_transform(feat_w_mt)
		feat_w_avg = feat_w_mt.mean(axis=1)
		feat_w_std = feat_w_mt.std(axis=1)
		sorted_idx = np.argsort(feat_w_avg, axis=-1)[::-1]
#		sorted_idx = sorted(range(feat_w_avg.shape[0]), key=lambda k: feat_w_avg[k])[::-1]
		sorted_feat_w = np.column_stack((X.columns.values[sorted_idx], feat_w_avg[sorted_idx], feat_w_std[sorted_idx]))
		feat_w_df = pd.DataFrame(sorted_feat_w, index=sorted_idx, columns=['Feature Name', 'Importance Mean', 'Importance Std'])
		if (cfg_param.setdefault('save_featw', False)):
			feat_w_df.to_excel('featw_%s_%s.xlsx' % (lbid, k.replace(' ', '_')))
		if (cfg_param.setdefault('save_featw_npz', False)):
			io.write_df(feat_w_df, 'featw_%s_%s' % (lbid, k.replace(' ', '_')), with_idx=True)
		if (cfg_param.setdefault('plot_featw', False)):
			plot.plot_bar(feat_w_avg[sorted_idx[:10]].reshape((1,-1)), feat_w_std[sorted_idx[:10]].reshape((1,-1)), X.columns.values[sorted_idx[:10]], labels=None, title='Feature importances', fname='fig_featw_%s_%s' % (lbid, k.replace(' ', '_').lower()), plot_cfg=common_cfg)
	for k, v in crsval_subfeatw.iteritems():
		# continue
		subfeat_w_mt = np.column_stack(v)
		mms = MinMaxScaler()
		subfeat_w_mt = mms.fit_transform(subfeat_w_mt)
		subfeat_w_avg = subfeat_w_mt.mean(axis=1)
		subfeat_w_std = subfeat_w_mt.std(axis=1)
		sorted_idx = np.argsort(subfeat_w_avg, axis=-1)[::-1]
		sorted_subfeat_w = np.column_stack((X.columns.values[sorted_idx], subfeat_w_avg[sorted_idx], subfeat_w_std[sorted_idx]))
		subfeat_w_df = pd.DataFrame(sorted_subfeat_w, index=sorted_idx, columns=['Feature Name', 'Importance Mean', 'Importance Std'])
		if (cfg_param.setdefault('save_subfeatw', False)):
			subfeat_w_df.to_excel('subfeatw_%s_%s.xlsx' % (lbid, k.replace(' ', '_')))
		if (cfg_param.setdefault('save_subfeatw_npz', False)):
			io.write_df(subfeat_w_df, 'subfeatw_%s_%s' % (lbid, k.replace(' ', '_')), with_idx=True)
		if (cfg_param.setdefault('plot_subfeatw', False)):
			plot.plot_bar(subfeat_w_avg[sorted_idx[:10]].reshape((1,-1)), subfeat_w_std[sorted_idx[:10]].reshape((1,-1)), X.columns.values[sorted_idx[:10]], labels=None, title='Feature importances', fname='fig_subfeatw_%s' % k.replace(' ', '_').lower(), plot_cfg=common_cfg)
	
	## Plot figures
	if (avg == 'all'):
		micro_roc_data, micro_roc_labels, micro_roc_aucs, macro_roc_data, macro_roc_labels, macro_roc_aucs = [[] for i in range(6)]
	else:
		roc_data, roc_labels, roc_aucs = [[] for i in range(3)]
	prc_data, prc_labels, prc_aucs = [[] for i in range(3)]
	for pl in PL_NAMES:
		if (avg == 'all'):
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
	if (len(group_dict) == len(PL_NAMES)):
		groups = None
	else:
		group_array = np.array(group_dict.values())
		group_arraY.sort()
		groups = group_arraY.tolist()
	if (avg == 'all'):
		aucs_df = pd.DataFrame([micro_roc_aucs, macro_roc_aucs, prc_aucs], index=['Micro ROC AUC', 'Macro ROC AUC', 'PRC AUC'], columns=PL_NAMES)
		if (cfg_param.setdefault('plot_roc', False)):
			plot.plot_roc(micro_roc_data, micro_roc_labels, groups=groups, fname='micro_roc_%s'%lbid, plot_cfg=common_cfg)
			plot.plot_roc(macro_roc_data, macro_roc_labels, groups=groups, fname='macro_roc_%s'%lbid, plot_cfg=common_cfg)
	else:
		aucs_df = pd.DataFrame([roc_aucs, prc_aucs], index=['ROC AUC', 'PRC AUC'], columns=PL_NAMES)
		if (cfg_param.setdefault('plot_roc', False)):
			plot.plot_roc(roc_data, roc_labels, groups=groups, fname='roc_%s'%lbid, plot_cfg=common_cfg)
	if (cfg_param.setdefault('plot_prc', False)):
		plot.plot_prc(prc_data, prc_labels, groups=groups, fname='prc_%s'%lbid, plot_cfg=common_cfg)
	if (cfg_param.setdefault('save_auc', False)):
		aucs_df.to_excel('auc_%s.xlsx' % lbid)
	filt_num, clf_num = len(FILT_NAMES), len(CLF_NAMES)
	if (cfg_param.setdefault('plot_metric', False)):
		for mtrc in metric_idx:
			mtrc_avg_list, mtrc_std_list = [[] for i in range(2)]
			if (global_param['comb']):
				mtrc_avg = perf_avg_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				mtrc_std = perf_std_df.ix[mtrc,:].as_matrix().reshape((1,-1))
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=PL_NAMES, labels=None, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clf_ft_%s' % (mtrc.replace(' ', '_').lower(), lbid), plot_cfg=common_cfg)
			else:
				for i in xrange(filt_num):
					offset = i * clf_num
					mtrc_avg_list.append(perf_avg_df.ix[mtrc,offset:offset+clf_num].as_matrix().reshape((1,-1)))
					mtrc_std_list.append(perf_std_df.ix[mtrc,offset:offset+clf_num].as_matrix().reshape((1,-1)))
				mtrc_avg = np.concatenate(mtrc_avg_list)
				mtrc_std = np.concatenate(mtrc_std_list)
				plot.plot_bar(mtrc_avg, mtrc_std, xlabels=CLF_NAMES, labels=FILT_NAMES, title='%s by Classifier and Feature Selection' % mtrc, fname='%s_clf_ft_%s' % (mtrc.replace(' ', '_').lower(), lbid), plot_cfg=common_cfg)


def tune_param(mdl_name, mdl, X, Y, rdtune, params, mltl=False, avg='micro', n_jobs=-1):
	if (rdtune):
		param_dist, n_iter = [params[k] for k in ['param_dist', 'n_iter']]
		grid = RandomizedSearchCV(estimator=mdl, param_distributions=param_dist, n_iter=n_iter, scoring='f1_%s' % avg if mltl else 'f1', n_jobs=n_jobs, error_score=0)
	else:
		param_grid, cv = [params[k] for k in ['param_grid', 'cv']]
		grid = GridSearchCV(estimator=mdl, param_grid=param_grid, scoring='f1_micro' if mltl else 'f1', cv=cv, n_jobs=n_jobs, error_score=0)
	grid.fit(X, Y)
	print("The best parameters of [%s] are %s, with a score of %0.3f" % (mdl_name, grid.best_params_, grid.best_score_))
	if (rdtune):
		param_grid = {}
		for p_tuple in grid.grid_scores_:
			for p_name, p_val in p_tuple[0].iteritems():
				param_grid.setdefault(p_name, []).append(p_val)
	else:
		param_grid = grid.param_grid
	dim_names = dict([(k, i) for i, k in enumerate(param_grid.keys())])
	dim_vals = {}
	for pn in dim_names.keys():
		dim_vals[pn] = dict([(k, i) for i, k in enumerate(param_grid[pn])])
	score_avg_cube = np.ndarray(shape=[len(param_grid[k]) for k in param_grid.keys()], dtype='float')
	score_std_cube = np.ndarray(shape=[len(param_grid[k]) for k in param_grid.keys()], dtype='float')
	for gs in grid.grid_scores_:
		idx = np.zeros((len(dim_names),), dtype='int')
		for k, v in gs[0].iteritems():
			idx[dim_names[k]] = dim_vals[k][v]
		score_avg_cube[tuple(idx)] = gs[1]
		score_std_cube[tuple(idx)] = gs[2].std()
	return grid.best_params_, grid.best_score_, score_avg_cube, score_std_cube, dim_names, dim_vals