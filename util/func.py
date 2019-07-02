#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-04-27 22:15:45
###########################################################################
#

import copy, difflib, operator, itertools, functools
from collections import OrderedDict

from sklearn.multiclass import OneVsRestClassifier


def build_model(mdl_func, mdl_t, mdl_name, tuned=False, pr=None, mltl=False, mltp=True, **kwargs):
	if (tuned and bool(pr)==False):
		print('Have not provided parameter writer!')
		return None
	if (mltl):
		return OneVsRestClassifier(mdl_func(**update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs)), n_jobs=-1) if (mltp) else OneVsRestClassifier(mdl_func(**update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs)))
	else:
		return mdl_func(**update_dict(pr(mdl_t, mdl_name) if tuned else {}, kwargs))


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


def partial_ret(func, ret_idx, other_res):
	def partial_func(**kwargs):
		res_list = func(**kwargs)
		results = []
		for i, res in enumerate(res_list):
			if i in ret_idx:
				results.append(res)
			else:
				other_res.append(res)
		return results if len(ret_idx) > 1 else results[0]
	return partial_func


def find_substr(text):
	def _find_substr(word):
		try:
			start_idx = text.index(word)
		except ValueError as e:
			try:
				print('Cannot find "%s" in "%s"' % (word, text))
			except:
				pass
			return (len(text), len(text))
		end_idx = start_idx + len(word)
		return start_idx, end_idx
	return _find_substr


def strsim(a, b):
	return difflib.SequenceMatcher(None, a, b).ratio()


def alignstrs(str_list, ref_list, ret_all=True, ret_idx=False):
	str_set, ref_set = set(str_list), set(ref_list)
	overlap = str_set & ref_set
	remain = str_set - ref_set
	return [x for x in ref_list if x in overlap] + [x for x in str_list if x in remain]


def capital_first(text):
	new_txt = text[0].upper() + text[1:]
	return new_txt


def padding_list(a, length, dummy=''):
	return a + [dummy] * (length - len(a)) if length > len(a) else a[:length]


def conserved_title(text):
	from titlecase import titlecase
	title = titlecase(text)
	mask = [any([x, y]) for x, y in zip([c.isupper() for c in text], [c.isupper() for c in title])]
	new_txt = ''.join([x.upper() if m else x for x, m in zip(text, mask)])
	return new_txt


def sorted_dict(data, key='value'):
	if (type(data) is not dict):
		print('Please input a Python dictionary!')
		return None
	else:
		if (key == 'value'):
			return sorted(data.items(), key=operator.itemgetter(1))
		else:
			return sorted(data.items(), key=operator.itemgetter(0))


def sorted_tuples(data, key_idx=0):
	if (type(data) is not list):
		print('Please input a Python list of tuple/list!')
		return None
	else:
		return sorted(data, key=operator.itemgetter(key_idx))


def remove_duplicate(collection):
	return list(OrderedDict.fromkeys(collection))


def unique_rowcol(mt, row_idx, col_idx, merge='del'):
	unique_idx, unique_col = {}, {}
	for i, idx in enumerate(row_idx):
		unique_idx.setdefault(idx, []).append(i)
	for i, col in enumerate(col_idx):
		unique_col.setdefault(col, []).append(i)
	duplicate_idx = flatten_list([val[1:] for val in unique_idx.values()])
	duplicate_col = flatten_list([val[1:] for val in unique_col.values()])
	if (merge == 'sum'):
		for idx, ilocs in unique_idx.items():
			if (len(ilocs) == 1): continue
			mt[ilocs[0],:] = mt[ilocs,:].sum(axis=0)
		for col, ilocs in unique_col.items():
			if (len(ilocs) == 1): continue
			mt[:,ilocs[0]] = mt[:,ilocs].sum(axis=1)
	uniq_idx, uniq_col = [i for i in range(len(row_idx)) if i not in duplicate_idx], [i for i in range(len(col_idx)) if i not in duplicate_col]
	return mt[uniq_idx,:][:,uniq_col], (len(uniq_idx), len(uniq_col)), [row_idx[i] for i in uniq_idx], [col_idx[i] for i in uniq_col]


def unique_rowcol_df(df, merge='del'):
	index, columns = df.index, df.columns
	unique_idx, unique_col = {}, {}
	for i, idx in enumerate(index):
		unique_idx.setdefault(idx, []).append(i)
	for i, col in enumerate(columns):
		unique_col.setdefault(col, []).append(i)
	duplicate_idx = flatten_list([val[1:] for val in unique_idx.values()])
	duplicate_col = flatten_list([val[1:] for val in unique_col.values()])
	if (merge == 'sum'):
		for idx, ilocs in unique_idx.items():
			if (len(ilocs) == 1): continue
			df.iloc[ilocs[0],:] = df.iloc[ilocs,:].sum(axis=0)
		for col, ilocs in unique_col.items():
			if (len(ilocs) == 1): continue
			df.iloc[:,ilocs[0]] = df.iloc[:,ilocs].sum(axis=1)
	return df.iloc[[i for i in range(len(index)) if i not in duplicate_idx], [i for i in range(len(columns)) if i not in duplicate_col]]

def update_dict(dict1, dict2):
	dict1.update(dict2)
	return dict1


def flatten_list(nested_list):
	if not hasattr(nested_list, '__iter__') or isinstance(nested_list, basestring): return nested_list
	l = list(itertools.chain.from_iterable(x if hasattr(x, '__iter__') and not isinstance(x, basestring) else [x] for x in nested_list))
	if (len(l) == 0): return []
	if (any([type(j) is list for j in l])):
		return flatten_list(l)
	else:
		return l


def flatten(container):
    for i in container:
        if hasattr(i, '__iter__'):
            for j in flatten(i):
                yield j
        else:
            yield i


def multimatch(re_list, s_list, conn='OR'):
	if (conn == 'OR'):
		for r, s in zip(re_list, s_list):
			if (r.match(s)):
				return True
		else:
			return False
	elif (conn == 'AND'):
		for r, s in zip(re_list, s_list):
			if (not r.match(s)):
				return False
		else:
			return True
