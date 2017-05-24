#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: metric.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-04-19 16:34:23
###########################################################################
#

import os
import operator
import itertools

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics


def list_roc(y_true, y_pred, average='micro', num_point=10):
	if (type(y_true[0]) is not list):
		if (type(y_pred[0]) is list):
			print 'Formats of the ground truth and the prediction are not consistent!'
			return None
		y_true, y_pred = [y_true], [y_pred]
	max_size, bin_labels = 0, []
	# Transform the list predictions to vectors
	for yt, yp in zip(y_true, y_pred):
		# Determine the top K value
		min_size = min(len(yt), len(yp))
		bin_lbs = []
		for k in range(1, min_size + 1):
			# Obtain the top k element of true list and prediction list
			topk_yt, topk_yp = yt[:k], yp[:k]
			# print topk_yt
			# print topk_yp
			# Transform them into binary labels
			mlb = MultiLabelBinarizer()
			bin_lbs.append(mlb.fit_transform([topk_yt, topk_yp]))
		bin_labels.append(bin_lbs)
		max_size = max(max_size, min_size)
	lb_size = np.array([len(x) for x in bin_labels])
	# Calculate the false positive rate and the true positive rate with different threshold k
	fpr, tpr, thrshd = [[] for x in range(3)]
	for k in range(int(1.0 * max_size / num_point), max_size + 1, num_point):
		idx = np.where(lb_size > k)[0]
		bin_lbs = [bin_labels[x][k] for x in idx]
		if (average == 'macro'):
			# Build the confusion matrix for each class and calculate the mean
			cfs_mts = [metrics.confusion_matrix(bin_lb[0], bin_lb[1]) for bin_lb in bin_lbs]
			tps, fps = zip(*[(cfs_mt[0, 0], cfs_mt[1, 0]) for cfs_mt in cfs_mts])
			tpr.append(tps.mean())
			fpr.append(fps.mean())
		else:
			# Cancatenate the labels of different classes and build the confusion matrix
			bin_lb = np.hstack(bin_lbs)
			cfs_mt = metrics.confusion_matrix(bin_lb[0], bin_lb[1])
			tpr.append(cfs_mt[0, 0])
			fpr.append(cfs_mt[1, 0])
		thrshd.append(1.0 * (k + 1) / max_size)
	sorted_idx = np.argsort(fpr)
	fpr, tpr, thrshd = np.array(fpr)[sorted_idx], np.array(tpr)[sorted_idx], np.array(thrshd)[sorted_idx]
	roc_auc = metrics.auc(fpr, tpr)
	return fpr, tpr, roc_auc, thrshd