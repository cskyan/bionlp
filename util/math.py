#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: math.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-29 21:48:19
###########################################################################
#

from itertools import chain, combinations, izip_longest

import numpy as np


ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

def subset(iterable, min_crdnl=0):
	s = list(iterable) if type(iterable) != list else iterable
	return chain.from_iterable(combinations(s, x) for x in range(min_crdnl, len(s)+1))
	

def grouper(n, iterable, fillvalue=None):
	args = [iter(iterable)] * n
	return izip_longest(*args, fillvalue=fillvalue)
	
	
class SymNDArray(np.ndarray):
	def __setitem__(self, (i, j), value):
		super(SymNDArray, self).__setitem__((i, j), value)                   
		super(SymNDArray, self).__setitem__((j, i), value)


def symmetrize(a):
	return np.array(a + a.T - np.diag(a.diagonal()), dtype=a.dtype)
	

def symarray(input_array, combine_data=True):
	"""
	Returns a symmetrized version of the array-like input_array.
	Further assignments to the array are automatically symmetrized.
	"""
	if (combine_data):
		return symmetrize(np.asarray(input_array)).view(SymNDArray)
	else:
		return np.asarray(input_array).view(SymNDArray)