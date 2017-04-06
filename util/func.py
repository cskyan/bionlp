#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: func.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-04-27 22:15:45
###########################################################################
#

import copy
import operator
import itertools
from collections import OrderedDict


def capital_first(text):
	new_txt = text[0].upper() + text[1:]
	return new_txt
	
	
def conserved_title(text):
	from titlecase import titlecase
	title = titlecase(text)
	mask = [any([x, y]) for x, y in zip([c.isupper() for c in text], [c.isupper() for c in title])]
	new_txt = ''.join([x.upper() if m else x for x, m in zip(text, mask)])
	return new_txt
	
	
def sorted_dict(data):
	if (type(data) is not dict):
		print 'Please input a Python dictionary!'
		return None
	else:
		return sorted(data.items(), key=operator.itemgetter(1))
		
		
def sorted_tuples(data, key_idx=0):
	if (type(data) is not list):
		print 'Please input a Python list of tuple/list!'
		return None
	else:
		return sorted(data, key=operator.itemgetter(key_idx))


def remove_duplicate(collection):
	return list(OrderedDict.fromkeys(collection))


def update_dict(dict1, dict2):
	dict1.update(dict2)
	return dict1
	
	
def flatten_list(nested_list):
	l = list(itertools.chain.from_iterable(nested_list))
	if (type(l[0]) is list):
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