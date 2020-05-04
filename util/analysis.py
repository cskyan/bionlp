#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2020 by Caspar. All rights reserved.
# File Name: analysis.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2020-05-01 13:23:56
###########################################################################
#


from . import math as imath


## Scikit-Learn Utils ##

def extract_topn_features(vectorizer, inputs, topn=10):
	feature_names = vectorizer.get_feature_names()
	sorted_items = imath.sort_coo_rows(vectorizer.transform(inputs).tocoo())
    sorted_items = [x[:topn] for x in sorted_items]
	return [[(feature_names[idx], round(score, 4)) for idx, score in items] for items in sorted_items]
