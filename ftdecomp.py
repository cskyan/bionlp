#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: ftdecomp.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-08-22 15:02:05
###########################################################################
#

import os

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def t_sne(X, n_components=2, **kwargs):
	from ext.tsne import tsne
	return tsne(X, no_dims=n_components, **kwargs)


class DecompTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=2, decomp_func=t_sne, **kwargs):
		self.decomp_func = decomp_func
		self.n_components = n_components
		self.sf_kwargs = kwargs

		
	def _fit(self, X, Y=None):
		if (Y is None):
			self.X_transformed = self.decomp_func(X, self.n_components, **self.sf_kwargs)
		else:
			self.X_transformed = self.decomp_func(X, Y, self.n_components, **self.sf_kwargs)

	
	def fit(self, X, Y=None):
		self._fit(X, Y)
		return self
		
		
	def transform(self, X):
		return self.X_transformed


def main():
	pass


if __name__ == '__main__':
	main()