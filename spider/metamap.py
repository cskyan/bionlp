#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: metamap.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-07-22 15:29:50
###########################################################################
#

import os

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from ..util import io
from ..util import fs


DATA_PATH = 'D:\data\chmannot\mesh' #os.path.join(os.path.expanduser('~'), 'data', 'chmannot', 'mesh')


def get_mesh(pmids, from_file=None, ft_type='binary', max_df=1.0, min_df=1, fmt='npz', spfmt='csc'):
	mesh_term_list = []
	for pmid in pmids:
		current_phrase = ''
		mesh_terms = []
		try:
			lines = fs.read_file(os.path.join(DATA_PATH, pmid+'.mesh.txt'), 'utf8')
		except Exception as e:
			mesh_term_list.append([])
			continue
		for line in lines:
			if (line.startswith('Phrase')):
				line_snip = line.split('Phrase: ')
				if (len(line_snip) > 1): current_phrase = line_snip[1]
				else: current_phrase = ''
				continue
			if (not line or line.startswith('Processing') or line.startswith('Meta Mapping')):
				continue
			line_snip = line.split('[')
			if (len(line_snip) > 1 and line_snip[1].split(']')[0].strip() == 'Finding'):
				continue
			mesh_term = ' '.join(line_snip[0].split('(')[0].split()[1:]).strip()
			if (mesh_term): mesh_terms.append(mesh_term)
		mesh_term_list.append(set(mesh_terms))
	union_mesh_terms = list(set().union(*mesh_term_list))
	mlb = MultiLabelBinarizer(classes=union_mesh_terms)
	mesh_mt = (mlb.fit_transform(mesh_term_list), mlb.classes_)
	mesh_df = pd.DataFrame(mesh_mt[0], index=pmids, columns=mesh_mt[1])
	if (fmt == 'npz'):
		io.write_df(mesh_df, os.path.join(DATA_PATH, 'mesh.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
	else:
		mesh_df.to_csv(os.path.join(DATA_PATH, 'mesh.csv'), encoding='utf8')
	return mesh_df