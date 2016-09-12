#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: hoc.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-04 15:17:47
###########################################################################
#

import os
import re
import fnmatch
import codecs
import operator
import logging
from optparse import OptionParser
from itertools import chain
from collections import Counter

import numpy as np
import scipy as sp
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from ..util import io
from ..util import fs


DATA_PATH = 'D:\data\chmannot' #os.path.join(os.path.expanduser('~'), 'data', 'chmannot')
ANNOT_PATH = os.path.join(DATA_PATH, 'annot')
ABS_PATH = os.path.join(DATA_PATH, 'txt')
META_PATH = os.path.join(DATA_PATH, 'absMeta')
FEATS_PATH = os.path.join(DATA_PATH, 'features', 'featuresets')

SC=';;'

HM_MAP={'activating invasion and metastasis':'IM', 'sustaining proliferative signaling':'PS', 'inducing angiogenesis':'A', 'cellular energetics':'CE', 'resisting cell death':'CD', 'tumor promoting inflammation':'TPI', 'enabling replicative immortality':'RI', 'evading growth suppressors':'GS', 'avoiding immune destruction':'ID', 'genomic instability and mutation':'GI'}
IHM_MAP={'IM':'activating invasion and metastasis', 'PS':'sustaining proliferative signaling', 'A':'inducing angiogenesis', 'CE':'cellular energetics', 'CD':'resisting cell death', 'TPI':'tumor promoting inflammation', 'RI':'enabling replicative immortality', 'GS':'evading growth suppressors', 'ID':'avoiding immune destruction', 'GI':'genomic instability and mutation'}
NORM_HM={'GS':'self-sufficient cell division', 'PS':'insensitivity to signals to stop cell division', 'CD':'resisting cell death', 'RI':'limitless reproductive potential', 'A':'creating their own blood supply', 'IM':'ability to invade other organs', 'CE':'ability to survive with little oxygen', 'ID':'evading the immune system', 'GI':'genomic instability', 'TPI':'inflammation'}
FT_NUM={'PS':{'lem':1471, 'nn':355, 'ner':302, 'parse':435, 'vc':121, 'mesh':280, 'chem':129}, 'GS':{'lem':863, 'nn':176, 'ner':108, 'parse':189, 'vc':119, 'mesh':156, 'chem':55}, 'CD':{'lem':1403, 'nn':289, 'ner':215, 'parse':380, 'vc':119, 'mesh':262, 'chem':114}, 'RI':{'lem':506, 'nn':59, 'ner':54, 'parse':53, 'vc':105, 'mesh':80, 'chem':26}, 'A':{'lem':590, 'nn':93, 'ner':82, 'parse':96, 'vc':105, 'mesh':97, 'chem':28}, 'IM':{'lem':1052, 'nn':220, 'ner':161, 'parse':253, 'vc':114, 'mesh':172, 'chem':43}, 'GI':{'lem':1215, 'nn':188, 'ner':100, 'parse':235, 'vc':126, 'mesh':216, 'chem':77}, 'TPI':{'lem':843, 'nn':122, 'ner':99, 'parse':152, 'vc':111, 'mesh':121, 'chem':34}, 'CE':{'lem':410, 'nn':54, 'ner':31, 'parse':53, 'vc':103, 'mesh':68, 'chem':20}, 'ID':{'lem':498, 'nn':48, 'ner':55, 'parse':59, 'vc':99, 'mesh':68, 'chem':14}}

def get_pmids():
	return [os.path.splitext(os.path.basename(file))[0] for file in fs.traverse(ABS_PATH)]

def fetch_artcls(pmid_list):
	articles = []
	annot_patn = re.compile('\[\'.*?\'\]')
	for pmid in pmid_list:
		annot_text = ' '.join(fs.read_file(os.path.join(ANNOT_PATH, pmid+'.txt'), 'utf8')).lower()
		## Extract the hallmark label
		annots = [annot.strip('[]\'\" ') for annot_list in annot_patn.findall(annot_text) if annot_list != '[]' for annot in annot_list.split(',')]
		## Extract the abstracts
		abs_text = ' '.join(fs.read_file(os.path.join(ABS_PATH, pmid+'.txt'), 'utf8'))
		## Extract the Mesh heading and the chemicals
		meta_text = SC.join(fs.read_file(os.path.join(META_PATH, pmid+'.absmeta'), 'utf8')).split('~~~chem:')
		mesh_headings = meta_text[0].split('~~~mesh:')[1].strip().split(SC)
		chemicals = meta_text[1].strip().split(SC)
		articles.append({'id':pmid,'abs':abs_text, 'annots':annots, 'mesh':mesh_headings, 'chem':chemicals})
	return articles


def get_fsnames():
	return [f.split('.')[0] for f in fs.listf(FEATS_PATH)]
	
def get_featsets(feat_sets, label_num, labels=[]):
	feat_files = fs.listf(os.path.join(FEATS_PATH, 'byLabel'))
	featset_patn = re.compile('\s\d')
	if (len(labels) != 0):
		feature_sets = [[] for i in range(label_num)]
	else:
		feature_sets = []
	feat_stat = {}
	# Every feature set
	for fset in feat_sets:
		fs_list = []
		fs_stat = []
		# Every matched file
		for f in fnmatch.filter(feat_files, fset+'*'):
			ft_per_lb = []
			for line in fs.read_file(os.path.join(FEATS_PATH, 'byLabel', f), code='utf8'):
				feat_match = featset_patn.search(line)
				if (not feat_match):
					continue
				# Deal with different types of features
				if (fset == 'parse'):
					feature = line[:feat_match.start()].strip(' []').replace('\'', '').replace(', ', ',')
				else:
					feature = line[:feat_match.start()].strip(' ')
				ft_per_lb.append(feature)
			fs_per_lb = set(ft_per_lb)
			fs_list.append(fs_per_lb)
			fs_stat.append(len(fs_per_lb))
		
		# If the number of feature-set files is not equal to that of labels, combine the redundance into the last label
		if (len(fs_list) > label_num):
			fs_list[label_num-1].update(set.union(*fs_list[label_num:]))
			fs_stat[label_num-1] = sum(fs_stat[label_num-1:])
			del fs_list[label_num:]
			del fs_stat[label_num:]
			
		if (len(labels) != 0):
			for i in range(len(feature_sets)):
				feature_sets[i].append(fs_list[i])
		else:
			feature_sets.append(set.union(*fs_list))
		feat_stat[fset] = fs_stat
	return feature_sets, feat_stat
	
def get_data(articles, from_file=None, ft_type='binary', max_df=1.0, min_df=1, fmt='npz', spfmt='csc'):
	# Read from local files
	if (from_file):
		if (type(from_file) == bool):
			file_name = 'X.npz' if (fmt == 'npz') else 'X.csv'
		else:
			file_name = from_file
		print 'Reading file: %s and Y.%s' % (file_name, fmt)
		if (fmt == 'npz'):
			return io.read_df(os.path.join(DATA_PATH, file_name), with_idx=True, sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, 'Y.npz'), with_idx=True, sparse_fmt=spfmt)
		else:
			return pd.read_csv(os.path.join(DATA_PATH, file_name), index_col=0, encoding='utf8'), pd.read_csv(os.path.join(DATA_PATH, 'Y.csv'), index_col=0, encoding='utf8')
	## Feature columns
	ft_pmid, ft_abs, ft_lem, ft_nnv, ft_ner, ft_parse, ft_vc, ft_mesh, ft_chem, label = [[] for i in range(10)]
	ft_order = ['lem', 'nn', 'ner', 'parse', 'vc', 'mesh', 'chem']
	ft_name = {'lem':'LBoW', 'nn':'N-Bigrams', 'ner':'NE', 'parse':'GR', 'vc':'VC', 'mesh':'MeSH', 'chem':'Chem'}
	ft_dic = {'lem':ft_lem, 'ner':ft_ner, 'parse':ft_parse, 'vc':ft_vc, 'mesh':ft_mesh, 'chem':ft_chem}
	hm_lb = ['PS', 'GS', 'CD', 'RI', 'A', 'IM', 'GI', 'TPI', 'CE', 'ID']
	bft_dic, hm_stat = [{} for i in range(2)]
	label_set = set()
#	sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
	for artcl in articles:
		ft_pmid.append(artcl['id'])
		ft_abs.append(artcl['abs'])
		ft_mesh.append(artcl['mesh'])
		ft_chem.append(artcl['chem'])
		label.append(artcl['annots'])
		label_set.update(artcl['annots'])
		c = Counter(artcl['annots'])
		for hm, num in c.iteritems():
			hs = hm_stat.setdefault(hm, [0,0])
			hs[0], hs[1] = hs[0] + 1, hs[1] + num
#			hs[0], hs[1] = hs[0] + 1, hs[1] + len(sent_splitter.tokenize(artcl['abs'].strip()))
#	uniq_lb = list(label_set)
	uniq_lb = [IHM_MAP[lb] for lb in hm_lb]
	
	## Get the feature sets of the specific hallmark
#	feat_sets = get_fsnames()
	feat_sets = ft_order
	feature_sets, feat_stat = get_featsets(feat_sets, len(uniq_lb))
	
	ft_stat_mt = np.array([feat_stat[ft] for ft in ft_order]).T
	ft_stat_pd = pd.DataFrame(ft_stat_mt, index=hm_lb, columns=[ft_name[fset] for fset in feat_sets])
	hm_stat_pd = pd.DataFrame([hm_stat[lb] for lb in uniq_lb], index=hm_lb, columns=['No. abstracts', 'No. sentences'])
	if (fmt == 'npz'):
		io.write_df(ft_stat_pd, os.path.join(DATA_PATH, 'ft_stat.npz'))
		io.write_df(hm_stat_pd, os.path.join(DATA_PATH, 'hm_stat.npz'), with_idx=True)
	else:
		ft_stat_pd.to_csv(os.path.join(DATA_PATH, 'ft_stat.csv'), encoding='utf8')
		hm_stat_pd.to_csv(os.path.join(DATA_PATH, 'hm_stat.csv'), encoding='utf8')
		
	## Extract the features from the preprocessed data
	for i in range(len(feat_sets)):
		fset = feat_sets[i]
		feature_set = feature_sets[i]
		if (fset == 'chem' or fset == 'mesh'):
			continue
		if (fset == 'nn'):
			continue
		for pmid in ft_pmid:
			feature, extra_feat = [[], []]
			prev_term = ''
			for line in fs.read_file(os.path.join(DATA_PATH, fset, '.'.join([pmid, fset, 'txt'])), 'utf8'):
				if (line == '~~~'):
					continue
				if (fset == 'lem'):
					if (line == '.	.	.' or line == '~~~	~~~' or line == ',	,	,'):
						continue
					items = line.split()
					if (len(items) < 3): # Skip the unrecognized words
						continue
					feature.append(items[2].lower())
					# Extract NN feature
					if (items[1] == 'NN'):
						if (prev_term != ''):
							extra_feat.append(prev_term + ' ' + items[0].lower())
						prev_term = items[0].lower()
					else:
						prev_term = ''
				if (fset == 'ner'):
					feature.append(line)
				if (fset == 'parse'):
					record = line.strip('()').replace(' _ ', ' ').split()
					feature.append(','.join([w.split('_')[0] for w in record]).lower())
				if (fset == 'vc'):
					feature.extend(line.split())
			ft_dic[fset].append(feature)
			if (fset == 'lem'):
				ft_nnv.extend(extra_feat)

	## Convert the raw features into binary features
	ft_type = ft_type.lower()
	for i in range(len(feat_sets)):
		fset = feat_sets[i]
		feature_set = feature_sets[i]
		if (fset == 'nn'):
			bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', max_df=max_df, min_df=min_df, vocabulary=set(ft_nnv), binary=True if ft_type=='binary' else False)
			ft_nn = bigram_vectorizer.fit_transform(ft_abs).tocsc()
			nn_classes = [cls[0] for cls in sorted(bigram_vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
			bft_dic[fset] = (ft_nn, nn_classes)
			continue
#		overall_ft = list(set([ft for samp in ft_dic[fset] for ft in samp if ft]))
#		mlb = MultiLabelBinarizer(classes=overall_ft)
#		bft_dic[fset] = (mlb.fit_transform(ft_dic[fset]), mlb.classes_)
		count_vectorizer = CountVectorizer(tokenizer=lambda text: [t for t in text.split('*#@') if t], lowercase=False, stop_words='english', token_pattern=r'\b\w+\b', max_df=max_df, min_df=min_df, binary=True if ft_type=='binary' else False)
		ft_all = count_vectorizer.fit_transform(['*#@'.join(samp) for samp in ft_dic[fset]])
		all_classes = [cls[0] for cls in sorted(count_vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
		bft_dic[fset] = (ft_all, all_classes)
	
	## Convert the annotations of each document to binary labels
	mlb = MultiLabelBinarizer(classes=uniq_lb)
	bin_label = (mlb.fit_transform(label), mlb.classes_)
	
	## Generate the features as well as the labels to form a completed dataset
	feat_mt = sp.sparse.hstack([bft_dic[fset][0] for fset in ft_order])
	if (ft_type == 'tfidf'):
		transformer = TfidfTransformer(norm='l2', sublinear_tf=False)
		feat_mt = transformer.fit_transform(feat_mt)
	feat_cols = ['%s_%s' % (fset, w) for fset in ft_order for w in bft_dic[fset][1]]
	feat_df = pd.DataFrame(feat_mt.todense(), index=ft_pmid, columns=feat_cols)
	label_df = pd.DataFrame(bin_label[0], index=ft_pmid, columns=bin_label[1])
	
	obj_samp_idx = np.random.random_integers(0, feat_df.shape[0] - 1, size=200).tolist()
	ft_samp_idx = np.random.random_integers(0, feat_df.shape[1] - 1, size=1000).tolist()
	samp_feat_df = feat_df.iloc[obj_samp_idx, ft_samp_idx]
	samp_lb_df = label_df.iloc[obj_samp_idx,:]

	if (fmt == 'npz'):
		io.write_df(feat_df, os.path.join(DATA_PATH, 'X.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(label_df, os.path.join(DATA_PATH, 'Y.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(samp_feat_df, os.path.join(DATA_PATH, 'sample_X.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
		io.write_df(samp_lb_df, os.path.join(DATA_PATH, 'sample_Y.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
	else:
		feat_df.to_csv(os.path.join(DATA_PATH, 'X.csv'), encoding='utf8')
		label_df.to_csv(os.path.join(DATA_PATH, 'Y.csv'), encoding='utf8')
		samp_feat_df.to_csv(os.path.join(DATA_PATH, 'sample_X.csv'), encoding='utf8')
		samp_lb_df.to_csv(os.path.join(DATA_PATH, 'sample_Y.csv'), encoding='utf8')
	return feat_df, label_df

	
def get_mltl_csv(lbs=[], mltlx=True):
	if (len(lbs) == 0):
		return None, None
	Xs = []
	Ys = []
	for lb in lbs:
		if (mltlx):
			X_bylb = pd.read_csv(os.path.join(DATA_PATH, 'X_%i.csv' % lb), index_col=0, encoding='utf8')
			Xs.append(X_bylb)
		labels = pd.read_csv(os.path.join(DATA_PATH, 'y_%i.csv' % lb), index_col=0, header=None, squeeze=True, encoding='utf8')
		Ys.append(labels)
	if (not mltlx):
		Xs.append(pd.read_csv(os.path.join(DATA_PATH, 'X.csv'), index_col=0, encoding='utf8'))
	return Xs, Ys
	
	
def get_mltl_npz(lbs=[], mltlx=True, spfmt='csc'):
	if (len(lbs) == 0):
		return None, None
	Xs = []
	Ys = []
	for lb in lbs:
		if (mltlx):
			X_bylb = io.read_df(os.path.join(DATA_PATH, 'X_%i.npz' % lb), with_idx=True, sparse_fmt=spfmt)
			Xs.append(X_bylb)
		labels = io.read_df(os.path.join(DATA_PATH, 'y_%i.npz' % lb), with_col=False, with_idx=True)
		Ys.append(labels)
	if (not mltlx):
		Xs.append(io.read_df(os.path.join(DATA_PATH, 'X.npz'), with_idx=True, sparse_fmt=spfmt))
	return Xs, Ys

	
def get_mltl_iter(lb, mltlx=True, chunksize=20):
	labels = pd.read_csv(os.path.join(DATA_PATH, 'y_%i.csv' % lb), index_col=0, header=None, squeeze=True, encoding='utf8')
	if (mltlx):
		reader = pd.read_csv(os.path.join(DATA_PATH, 'X_%i.csv' % lb), index_col=0, encoding='utf8', iterator=True, chunksize=chunksize)
	else:
		reader = pd.read_csv(os.path.join(DATA_PATH, 'X.csv'), index_col=0, encoding='utf8', iterator=True, chunksize=chunksize)

	def data_iter():
		for df in reader:
			yield df
	return data_iter, labels
	
	
def ft_filter(X, Y):
	ft_order = ['lem', 'nn', 'ner', 'parse', 'vc', 'mesh', 'chem']
	hm_lb = ['PS', 'GS', 'CD', 'RI', 'A', 'IM', 'GI', 'TPI', 'CE', 'ID']
	init_min_freq, init_max_freq, min_interval, epsilon = 5, 500, 10, 2
	ft_idx = {}
	for i, col in enumerate(X.columns):
		for ft in ft_order:
			if (col.startswith(ft+'_')):
				ft_idx.setdefault(ft, []).append(i)
				break
	Xs = []
	for i in range(Y.shape[1]):
		sub_X = X.iloc[np.arange(Y.shape[0])[Y.iloc[:,i].values == 1],:]
		ft_freqs = sub_X.sum(axis=0)
		agg_ft_freqs = np.array([ft_freqs.iloc[ft_idx[ft]].sum() for ft in ft_order])
		filt_fts = []
		for idx, ft in enumerate(ft_order):
			tgt_num = FT_NUM[hm_lb[i]][ft]
			ft_freq = ft_freqs.iloc[ft_idx[ft]]
			ordered_ft_indice = np.argsort(ft_freq.values)[::-1]
			ordered_ft_freq = ft_freq.iloc[ordered_ft_indice]
			foff, begin, end, step = ordered_ft_freq, 0, ordered_ft_freq.shape[0], max(1, int(0.01 * ordered_ft_freq[0]))
			min_freq, max_freq = init_min_freq, min(init_max_freq, foff.shape[0])
			while (foff.shape[0] > tgt_num):
				filt_condition = np.all([ordered_ft_freq > min_freq, ordered_ft_freq < max_freq], axis=0)
				foff, idx_range = ordered_ft_freq[filt_condition], np.arange(ordered_ft_freq.shape[0])[filt_condition]
				begin, end, min_freq, max_freq = idx_range[0], idx_range[-1], min_freq + step, max_freq - step
			offset = (tgt_num - foff.shape[0]) / 2
			begin = max(0, begin - offset)
			end = min(ordered_ft_freq.shape[0], begin + tgt_num)
			filt_ft = ordered_ft_freq.index[begin:end]
			filt_fts.extend(filt_ft.tolist())
		filt_X = X.loc[:,filt_fts]
		Xs.append(filt_X)
	return Xs


def main():
	pass


if __name__ == '__main__':
	# Logging setting
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Parse commandline arguments
	op = OptionParser()
	op.add_option('-f', '--fmt', default='npz', help='data stored format: csv or npz [default: %default]')
	op.add_option('-s', '--spfmt', default='csc', help='sparse data stored format: csc or csr [default: %default]')
	op.add_option('-t', '--type', default='binary', help='feature type: binary, numeric, tfidf or mixed [default: %default]')

	(opts, args) = op.parse_args()
	if len(args) > 0:
		op.print_help()
		op.error('Please input options instead of arguments.')
		exit(1)

	main()