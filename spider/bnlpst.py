#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2015 by Caspar. All rights reserved.
# File Name: bnlpst.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2015-11-23 21:44:57
###########################################################################
#

import os
import sys
import re
import math
import string
from sets import Set
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as prfs
import nltk
from nltk.stem.porter import PorterStemmer

from ..util import io
from ..util import fs


DATA_PATH = 'D:\\data\\bioevent\\bnlpst2011\\bgi' #os.path.join(os.path.expanduser('~'), 'data', 'bioevent', 'bnlpst2011', 'bgi')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
DEV_PATH = os.path.join(DATA_PATH, 'dev')
TEST_PATH = os.path.join(DATA_PATH, 'test')

TRGWD_PATH = os.path.join(DATA_PATH, 'trigger')

[STEM, POS, SPOS, ANNOT_TYPE, ANNOT_HASH] = [Set([]) for x in range(5)]
ft_offset = {'train':[], 'dev':[], 'test':[]}
dist_mts = {}
prdcss = {}


def get_docid(dataset='train'):
	if (dataset == 'train'):
		files = [os.path.splitext(fpath)[0] for fpath in os.listdir(TRAIN_PATH) if re.match(r'.*\.txt', fpath)]
	elif (dataset == 'dev'):
		files = [os.path.splitext(fpath)[0] for fpath in os.listdir(DEV_PATH) if re.match(r'.*\.txt', fpath)]
	elif (dataset == 'test'):
		files = [os.path.splitext(fpath)[0] for fpath in os.listdir(TEST_PATH) if re.match(r'.*\.txt', fpath)]
	return files


def get_a1(fpath):
	try:
		annots = {'id':[], 'type':[], 'loc':[], 'str':[]}	# Biological term annotation, a.k.a. named entity
		words = {'id':[], 'loc':[], 'str':[], 'stem':[], 'pos':[], 'stem_pos':[], 'annotp':[]} # Word tokenization
		depends = {'id':[], 'type':[], 'oprnd':[]} # Syntactic dependencies
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				if (line[0] == 'T'):
					annots['id'].append(record[0])
					annots['type'].append(record[1])
					annots['loc'].append((int(record[2]), int(record[3])))
					annots['str'].append(' '.join(record[4:]))
				if (line[0] == 'W'):
					words['id'].append(record[0])
					words['loc'].append((int(record[2]), int(record[3])))
					words['str'].append(' '.join(record[4:]))
					words['pos'].append('')
				if (line[0] == 'R'):
					depends['id'].append(record[0])
					depends['type'].append(record[1])
					depends['oprnd'].append((record[2], record[3]))
	except:
		print 'Can not open the file \'%s\'!' % fpath
		exit(-1)

	# Extract the part-of-speech from the annotations for every word
	oprnds = []
	pos_list = []
	for i in xrange(len(depends['id'])):
		operands, dpnd_tp = depends['oprnd'][i], depends['type'][i]
		pos = []
		#%%% Bad RE design, to be modified %%%#
		match_rs = re.compile('.+:(.+)-(.+)\((.*)\)').match(dpnd_tp)
		if (match_rs == None):
			match_rs = re.compile('.+:(.+)-(.+)').match(dpnd_tp)
		if (match_rs == None):
			# Appositive, expressed as 'AS_$synonym'
			match_rs = re.compile('(appos)').match(dpnd_tp)
			words['pos'][words['id'].index(operands[0])] = 'AS_%s' % operands[1]
			words['pos'][words['id'].index(operands[1])] = 'AS_%s' % operands[0]
			continue
		else:
			pos.extend(match_rs.groups()[:2])
		#%%%%%%#
		if (len(pos) != len(operands)):
			print "Annotation Error!"
			continue
		oprnds.extend(operands)
		pos_list.extend(pos)
	for x in xrange(len(oprnds)):
		words['pos'][words['id'].index(oprnds[x])] = pos_list[x]

	# Deal with appositive, link the synonym together into a list, assign the pos to each synonym according to the one has non-appos pos 
	for x in xrange(len(words['pos'])):
		match_rs = re.compile('^AS_(.*)').match(words['pos'][x])
		identcls = [x]
		while (match_rs):
			wid = words['id'].index(match_rs.groups()[0])
			# found the beginning word again
			if (len(identcls) > 1 and identcls[0] == wid):
				break
			identcls.append(wid)
			match_rs = re.compile('^AS_(.*)').match(words['pos'][identcls[-1]])
		if (not match_rs): # The last identical word has non-appos pos
			for y in identcls:
				words['pos'][y] = words['pos'][identcls[-1]]
		else:
			for y in identcls:	# The last identical word does not have non-appos pos, namely found a cycle link
				words['pos'][y] = ''
			continue
	return annots, words, depends


# Get the annotations, segmentations, and the dependcies from a1 file
def get_preprcs(docids, dataset='train'):
	preprcs = []
	if (dataset == 'train'):
		dir_path = TRAIN_PATH
	elif (dataset == 'dev'):
		dir_path = DEV_PATH
	elif (dataset == 'test'):
		dir_path = TEST_PATH
	for did in docids:
		preprcs.append(get_a1(os.path.join(dir_path, did + '.a1')))
	return preprcs


def get_a2(fpath):
	try:
		events = {'id':[], 'type':[], 'oprnd_tps':[], 'oprnds':[]} # Interaction events
		with open(fpath, 'r') as fd:
			for line in fd.readlines():
				record = line.split()
				if (line[0] == 'E'):
					events['id'].append(record[0])
					events['type'].append(record[1])
					loprnd, roprnd = record[2].split(':'), record[3].split(':')
					events['oprnd_tps'].append((loprnd[0], roprnd[0]))
					events['oprnds'].append((loprnd[1], roprnd[1]))
	except:
		print 'Can not open the file \'%s\'!' % fpath
		exit(-1)
	return events
	

# Get the events from a2 file	
def get_evnts(docids, dataset='train'):
	event_list = []
	if (dataset == 'train'):
		dir_path = TRAIN_PATH
	elif (dataset == 'dev'):
		dir_path = DEV_PATH
	elif (dataset == 'test'):
		dir_path = TEST_PATH
	for did in docids:
		event_list.append(get_a2(os.path.join(dir_path, did + '.a2')))
	return event_list
	
	
def get_corpus(docids, dataset='train', ext_fmt='txt'):
	corpus = []
	if (dataset == 'train'):
		dir_path = TRAIN_PATH
	elif (dataset == 'dev'):
		dir_path = DEV_PATH
	elif (dataset == 'test'):
		dir_path = TEST_PATH
	for did in docids:
		corpus.append(' '.join(fs.read_file(os.path.join(dir_path, did+'.%s'%ext_fmt), 'utf8')))
	return corpus


def get_trgwd(fpath):
	trg_wds = []
	try:
		with open(fpath, 'r') as fd:
			trg_wds.extend(fd.readline().split())
	except:
		print 'Can not open the file \'%s\'!' % fpath
	return trg_wds


def multimatch(re_list, s_list, conn='OR'):
	if (conn == 'OR'):
		for r, s in zip(re_list, s_list):
			if (r.match(s)):
				return True
		else:
			return False
	else:
		for r, s in zip(re_list, s_list):
			if (not r.match(s)):
				return False
		else:
			return True


def _extr_verb(str):
	return ''
	r_pos = re.compile('.*V.*')
	text = nltk.word_tokenize(str)
	pos = nltk.pos_tag(text)
	for w, p in pos:
		if (r_pos.match(p)):
			return w
	return ''


def _find_trg(words, wid1, wid2, dist_mt, prdcs):
	trg_wds = get_trgwd(os.path.join(TRGWD_PATH, 'TRIGGERWORDS.txt'))
	r_patn = re.compile(r'\b^'+r'$\b|\b^'.join(trg_wds)+r'$\b', flags=re.I | re.X)
	trigger, prdc = None, wid2
	r_pos = re.compile('.*V.*')
#	if (r_pos.match(words['pos'][wid1]) or r_pos.match(words['stem_pos'][wid1]) or r_patn.match(words['str'][wid1]) or r_patn.match(words['stem'][wid2])):
	if (r_pos.match(words['pos'][wid1]) or r_patn.match(words['str'][wid1]) or r_patn.match(words['annotp'][wid1]) or r_patn.match(words['stem'][wid2])):
#	if (r_pos.match(words['pos'][wid1]) or r_patn.match(words['str'][wid1]) or r_patn.match(words['stem'][wid2])):
		return wid1
#	elif (r_pos.match(words['pos'][wid2]) or r_pos.match(words['stem_pos'][wid1]) or r_patn.match(words['str'][wid2]) or r_patn.match(words['stem'][wid2])):
	elif (r_pos.match(words['pos'][wid2]) or r_patn.match(words['str'][wid2]) or r_patn.match(words['annotp'][wid1]) or r_patn.match(words['stem'][wid2])):
#	elif (r_pos.match(words['pos'][wid2]) or r_patn.match(words['str'][wid2]) or r_patn.match(words['stem'][wid2])):
		return wid2
	while (prdcs[wid1, prdc] != -9999):
		if (words['pos'][prdc] == 'PHRASE'):
			verb = _extr_verb(words['str'][prdc])
			if (verb != ''):
				prdc = words['str'].index(verb)
#		if (r_pos.match(words['pos'][prdc]) or r_pos.match(words['stem_pos'][prdc]) or r_patn.match(words['str'][prdc]) or r_patn.match(words['stem'][prdc])):
		if (r_pos.match(words['pos'][prdc]) or r_patn.match(words['str'][prdc]) or r_patn.match(words['annotp'][prdc]) or r_patn.match(words['stem'][prdc])):
		#if (r_pos.match(words['pos'][prdc]) or r_patn.match(words['str'][prdc]) or r_patn.match(words['stem'][prdc])):
			trigger = prdc
			break
		if (prdcs[wid1, prdc] == wid1):
			break
		prdc = prdcs[wid1,prdc]
	#print words['str'][wid1], '|', words['str'][wid2], '|', words['str'][trigger] if trigger else ''
	return trigger


def gen_edges(trigger, annots, dist_mt, prdcs):
	edges = []
	# Construct theme argument edge
	for w in xrange(len(dist_mt.shape[0])):
		idx_pair = (w, trigger)
		if (dist_mt[idx_pair] != np.inf and annots[w] != ''):
			edges.append(idx_pair)
	# Construct cause argument edge
	for w in xrange(len(dist_mt.shape[1])):
		idx_pair = (trigger, w)
		if (dist_mt[idx_pair] != np.inf and annots[w] != ''):
			edges.append((w, trigger))
	return edges


def dpnd_trnsfm(dict_data, shape):
	data_mt = np.array(dict_data.items())
	idx_list, v_list = data_mt[:,0], data_mt[:,1]
	# Convert the value vector to a singular
	int_func = np.frompyfunc(int, 1, 1)
	hash_func = np.frompyfunc(lambda x: 1, 1, 1)
	idx_list = int_func(idx_list)
	data = hash_func(v_list).mean(axis=1)
	return coo_matrix((data, (idx_list[:,0], idx_list[:,1])), shape=shape, dtype='float16')
		
		
def get_data(raw_data, dataset='train', ft_type='binary', max_df=1.0, min_df=1, fmt='npz', spfmt='csc'):
	global ft_offset
	idx_range = [0, 0]
	# Read from local files
	if (raw_data is None):
		if (fmt == 'npz'):
			word_X_name, word_y_name, edge_X_name, edge_y_name = '%swX.npz' % dataset, '%swy.npz' % dataset, '%seX.npz' % dataset, '%seY.npz' % dataset
			if (dataset == 'test'):
				print 'Reading file: %s' % (word_X_name)
				return io.read_df(os.path.join(DATA_PATH, word_X_name), sparse_fmt=spfmt)
			print 'Reading file: %s, %s, %s, %s' % (word_X_name, word_y_name, edge_X_name, edge_y_name)
			return io.read_df(os.path.join(DATA_PATH, word_X_name), sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, word_y_name), with_col=False, sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, edge_X_name), sparse_fmt=spfmt), io.read_df(os.path.join(DATA_PATH, edge_y_name), sparse_fmt=spfmt)
		else:
			word_X_name, word_y_name, edge_X_name, edge_y_name = '%sw_X.csv' % dataset, '%sw_y.csv' % dataset, '%se_X.csv' % dataset, '%se_y.csv' % dataset
			print 'Reading file: %s, %s, %s, %s' % (word_X_name, word_y_name, edge_X_name, edge_y_name)
			return pd.read_csv(os.path.join(DATA_PATH, word_X_name), sparse_fmt=spfmt), pd.read_csv(os.path.join(DATA_PATH, word_y_name), sparse_fmt=spfmt), pd.read_csv(os.path.join(DATA_PATH, edge_X_name), sparse_fmt=spfmt), pd.read_csv(os.path.join(DATA_PATH, edge_y_name), sparse_fmt=spfmt)

	## Token features
	[ft_str, ft_cap, ft_pun, ft_digit, ft_stem, ft_pos] = [[] for x in range(6)]
	## Annotation features
	[ft_annotp, ft_evntp, ft_evntoprt, ft_evntoprd, ft_trigger, edges, ft_edgelb] = [[] for x in range(7)]
	## Statistical features
	[stem_frqs, pos_frqs, spos_frqs, annotp_frqs, annoth_frqs, evntp_frqs, evntoprt_frqs, evntoprd_frqs, dpnd_mt] = [{} for x in range(9)]
	# Extract information from raw data
	for docid, corpus, preprcs, events in zip(raw_data['docids'], raw_data['corpus'], raw_data['preprcs'], raw_data['evnts']):
		annots, words, depends = preprcs
		word_num = len(words['id'])
		# Reset and record the word ID range for each document
		idx_range[0], idx_range[1] = (idx_range[1], idx_range[1] + word_num)
		ft_offset[dataset].append(idx_range[0])
		## Construct the token feature columns
		ft_str.extend(words['str'])
		ft_cap.extend([1 if any(str.isupper(c) for c in w) else 0 for w in words['str']])
		ft_pun.extend([1 if any(c in string.punctuation for c in w) else 0 for w in words['str']])
		ft_digit.extend([1 if any(c.isdigit() for c in w) else 0 for w in words['str']])
		# Construct the stem feature column
		stemmer = PorterStemmer()
		words['stem'] = [stemmer.stem(w) for w in words['str']]
		ft_stem.extend(words['stem'])
		# Construct the part-of-speech feature column
		ft_pos.extend(words['pos'])

		## Construct the annotation feature column
		for i in xrange(len(words['loc'])):
			loc = words['loc'][i]
			try:
				idx = annots['loc'].index(loc)
				words['annotp'].append(annots['type'][idx])
			except ValueError:
				words['annotp'].append('')
		ft_annotp.extend(words['annotp'])

		## Feature statistics

		## Construct the dependcy matrix
		sub_dpnd_mt = {}
		for i in xrange(len(depends['id'])):
			idx_pair = (words['id'].index(depends['oprnd'][i][0]), words['id'].index(depends['oprnd'][i][1]))
			match_rs = re.compile('(.+):.+\((.*)\)').match(depends['type'][i])
			dpnd_tp = ['', '']
			if (match_rs == None):
				match_rs = re.compile('(.+):.+').match(depends['type'][i])
			else:
				dpnd_tp[1] = match_rs.groups()[1]
			if (match_rs == None):
				match_rs = re.compile('(.+)').match(depends['type'][i])
			dpnd_tp[0] = match_rs.groups()[0]
			dpnd_mt[(idx_range[0]+idx_pair[0], idx_range[0]+idx_pair[1])] = dpnd_tp
			sub_dpnd_mt[(idx_pair[0], idx_pair[1])] = dpnd_tp
		
		if (dataset == 'test'):
			continue
		## Construct the trigger feature column
		ft_trigger.extend([0 for x in range(word_num)])
		## Construct the event type feature column
		ft_evntp.extend(events['type'])
		event_num = len(events['id'])
		# Connect the event operands to the corresponding words and annotate the trigger
		sdmt = dpnd_trnsfm(sub_dpnd_mt, (word_num, word_num))
		dist_mt, prdcs = shortest_path(sdmt, directed=False, return_predecessors=True)
		dist_mts[tuple(idx_range)] = dist_mt
		prdcss[tuple(idx_range)] = prdcs
		for i in xrange(event_num):
			loc_pairs = (annots['loc'][annots['id'].index(events['oprnds'][i][0])], annots['loc'][annots['id'].index(events['oprnds'][i][1])])
			try:
				idx_pair = (words['loc'].index(loc_pairs[0]), words['loc'].index(loc_pairs[1]))
			except Exception as e:
				pass
				#idx_pair = (words['str'].index(annots['str'][annots['id'].index(events['oprnds'][i][0])].split(' ')[0]), words['str'].index(annots['str'][annots['id'].index(events['oprnds'][i][1])].split(' ')[-1]))
			events['oprnds'][i] = [idx_range[0]+idx_pair[0], idx_range[0]+idx_pair[1]]
			# Find the trigger
			trigger = _find_trg(words, idx_pair[0], idx_pair[1], dist_mt, prdcs)
			if (trigger):
				ft_trigger[idx_range[0] + trigger] = 1
			# Generate the argument training samples, format: <src_wid, tgt_wid, distance>
			if (trigger):
				if (trigger == idx_pair[0]):
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[1], dist_mt[trigger, idx_pair[1]]))
					ft_edgelb.append(events['type'][i])
				elif (trigger == idx_pair[1]):
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[0], dist_mt[trigger, idx_pair[0]]))
					ft_edgelb.append(events['type'][i])
				else:
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[0], dist_mt[trigger, idx_pair[0]]))
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[1], dist_mt[trigger, idx_pair[1]]))
					ft_edgelb.extend([events['type'][i]]*2)
		# Construct the event operands type column
		ft_evntoprt.extend(events['oprnd_tps'])
		# Event statistics
		for eventp in events['type']:
			evntp_frqs[eventp] = evntp_frqs.setdefault(eventp, 0) + 1
		for evntoprts in events['oprnd_tps']:
			evntoprt_frqs[evntoprts[0]] = evntoprt_frqs.setdefault(evntoprts[0], 0) + 1
			evntoprt_frqs[evntoprts[1]] = evntoprt_frqs.setdefault(evntoprts[1], 0) + 1

	## Construct word matrix
	ft_str_mt = np.array(ft_str, dtype='str').reshape(len(ft_str),1)
	ft_cap_mt = np.array(ft_cap).reshape(len(ft_cap),1)
	ft_pun_mt = np.array(ft_pun).reshape(len(ft_pun),1)
	ft_digit_mt = np.array(ft_digit).reshape(len(ft_digit),1)
	
	ft_stem_mt = label_binarize(ft_stem, classes=list(set(ft_stem)))
	ft_pos_mt = label_binarize(ft_pos, classes=list(set(ft_pos)))
	
	# Construct the character bi-grams and trigrams
	bgv = CountVectorizer(ngram_range=(2, 2), analyzer='char', max_df=max_df, min_df=min_df, binary=True if ft_type=='binary' else False)
	tgv = CountVectorizer(ngram_range=(3, 3), analyzer='char', max_df=max_df, min_df=min_df, binary=True if ft_type=='binary' else False)
	ft_chbigram = bgv.fit_transform(ft_str).todense()
	ft_chtrigram = tgv.fit_transform(ft_str).todense()
	
	ft_annotp_mt = label_binarize(ft_annotp, classes=list(set(ft_annotp)))
	word_mt = np.hstack((ft_cap_mt, ft_pun_mt, ft_digit_mt, ft_stem_mt, ft_pos_mt, ft_chbigram, ft_chtrigram))
#	wm_cols = ['has_cap', 'has_pun', 'has_num', 'loc_s', 'loc_e']
	wm_cols = ['has_cap', 'has_pun', 'has_num'] + list(set(ft_stem)) + list(set(ft_pos)) + bgv.vocabulary_.keys() + tgv.vocabulary_.keys()
	if (dataset == 'test'):
		word_df = pd.DataFrame(word_mt, columns=wm_cols)
		io.write_df(word_df, os.path.join(DATA_PATH, '%swX.npz'%dataset), sparse_fmt=spfmt, compress=True)
		return word_df
	else:
		# Construct trigger label
		trg_label = np.array(ft_trigger)
		# Construct trigger-argument pair sample matrix
		edge_data = np.array(edges)
		edge_mt = np.hstack((word_mt[edge_data[:,0].astype(int),:], word_mt[edge_data[:,1].astype(int),:]))
		em_cols = ['lf_%s' % col for col in wm_cols] + ['rt_%s' % col for col in wm_cols]
		edge_label = np.array(ft_edgelb)
		# Combine all the data into Pandas DataFrame
		word_df = pd.DataFrame(word_mt, columns=wm_cols)
		trg_lb = pd.DataFrame(trg_label.reshape((-1,1)))
		edge_df = pd.DataFrame(edge_mt, columns=em_cols)
		edge_lb_mt = label_binarize(edge_label, classes=list(set(edge_label)))
		edge_lb = pd.DataFrame(edge_lb_mt, columns=list(set(edge_label)))
		
		if (fmt == 'npz'):
			io.write_df(word_df, os.path.join(DATA_PATH, '%swX.npz'%dataset), sparse_fmt=spfmt, compress=True)
			io.write_df(trg_lb, os.path.join(DATA_PATH, '%swy.npz'%dataset), with_col=False, sparse_fmt=spfmt, compress=True)
			io.write_df(edge_df, os.path.join(DATA_PATH, '%seX.npz'%dataset), sparse_fmt=spfmt, compress=True)
			io.write_df(edge_lb, os.path.join(DATA_PATH, '%seY.npz'%dataset), sparse_fmt=spfmt, compress=True)
		else:
			word_df.to_csv(os.path.join(DATA_PATH, '%swX.csv'%dataset), encoding='utf8')
			trg_lb.to_csv(os.path.join(DATA_PATH, '%swy.csv'%dataset), encoding='utf8')
			edge_df.to_csv(os.path.join(DATA_PATH, '%seX.csv'%dataset), encoding='utf8')
			edge_lb.to_csv(os.path.join(DATA_PATH, '%seY.csv'%dataset), encoding='utf8')

		return word_df, trg_lb, edge_df, edge_lb
		
		
def post_process(data, type):
	if (type == 'trigger'):
		pass
	elif (type == 'edge'):
		pass


def get_data_bak(dir_path, trained=False, with_annot=True):
	idx_range = [0, 0]
	## Token features
	[ft_str, ft_cap, ft_pun, ft_digit, ft_loc, ft_stem, ft_pos, ft_spos] = [[] for x in range(8)]
	## Annotation features
	[ft_annotp, ft_evntp, ft_evntoprt, ft_evntoprd, ft_trigger, edges, ft_edgelb] = [[] for x in range(7)]
	## Statistical features
	[stem_frqs, pos_frqs, spos_frqs, annotp_frqs, annoth_frqs, evntp_frqs, evntoprt_frqs, evntoprd_frqs, dpnd_mt] = [{} for x in range(9)]

	# Extract information from a1 and a2 files
	a1_files = [f for f in os.listdir(dir_path) if re.match(r'.*\.a1', f)]
	if (with_annot):
		a2_files = [f for f in os.listdir(dir_path) if re.match(r'.*\.a2', f)]
	else:
		a2_files = [''] * len(a1_files)
	for a1_file, a2_file in zip(a1_files, a2_files):
		#print a1_file, a2_file
		# Get the annotations, segmentations, and the dependcies from a1 file
		annots, words, depends = get_a1(os.path.join(dir_path, a1_file))
		#print ' '.join(words['str'])
		# Reset the ID range
		orig_word_num = len(words['id'])
		idx_range[0], idx_range[1] = (idx_range[1], idx_range[1] + len(words['id']))
		if (trained):
			ft_offset.append(idx_range[0])
		# Recognize multiword token
		for i in xrange(len(annots['id'])):
			annot_pos = annots['loc'][i]
			try:
				idx = words['loc'].index(annot_pos)
			except ValueError:
				words['id'].append(annots['id'][i])
				words['str'].append(annots['str'][i])
				words['loc'].append(annots['loc'][i])
				words['pos'].append('PHRASE')
				idx_range[1] += 1
		word_num = len(words['id'])
		## Construct the token feature columns
		ft_str.extend(words['str'])
		#ft_cap.extend(map(is_cap, words['str']))
		ft_cap.extend([1 if any(str.isupper(c) for c in w) else 0 for w in words['str']])
		ft_pun.extend([1 if any(c in string.punctuation for c in w) else 0 for w in words['str']])
		ft_digit.extend([1 if any(c.isdigit() for c in w) else 0 for w in words['str']])
		# Construct the position feature column
		ft_loc.extend(words['loc'])
		# Construct the stem feature colums
		stemmer = PorterStemmer()
		words['stem'] = ['NULL'] * len(words['id'])
		words['stem'] = [stemmer.stem(w) for w in words['str'][:orig_word_num]] + [''] * (word_num - orig_word_num) # Some of the phrases' stems are still phrase -.-!
		print ' '.join(words['stem'])
		ft_stem.extend(words['stem'])
		# The part-of-speech feature column for the stem
		#words['stem_pos'] = map(lambda x: nltk.pos_tag(nltk.word_tokenize(x))[0][1], words['str'][:orig_word_num]) + ['PHRASE'] * (word_num - orig_word_num)
		#print ' '.join(words['stem_pos'])
		
		# Construct the part-of-speech feature column
		ft_pos.extend(words['pos'])
		#ft_spos.extend(words['stem_pos'])

		## Construct the annotation feature column
		for i in xrange(len(words['loc'])):
			loc = words['loc'][i]
			try:
				idx = annots['loc'].index(loc)
				words['annotp'].append(annots['type'][idx])
			except ValueError:
				words['annotp'].append('')
		ft_annotp.extend(words['annotp'])

		## Feature statistics
		stems = filter(re.compile('\w+').match, ft_stem)
		for stem in stems:
			stem_frqs[stem] = stem_frqs.setdefault(stem, 0) + 1
		if not (trained):
			STEM.update(stem_frqs.keys())
		poses = filter(re.compile('.+').match, words['pos'])
		for pos in poses:
			pos_frqs[pos] = pos_frqs.setdefault(pos, 0) + 1
		sposes = filter(re.compile('.+').match, words['stem_pos'])
		for spos in sposes:
			spos_frqs[spos] = spos_frqs.setdefault(spos, 0) + 1
		if not (trained):
			POS.update(pos_frqs.keys())
			SPOS.update(spos_frqs.keys())
		annotps = filter(re.compile('.+').match, ft_annotp[len(ft_annotp)-len(words['id']):])
		for annotp in annotps:
			annotp_frqs[annotp] = annotp_frqs.setdefault(annotp, 0) + 1
		if not (trained):
			ANNOT_TYPE.update(annotp_frqs.keys())
		## Construct the dependcy matrix
		sub_dpnd_mt = {}
		for i in xrange(len(depends['id'])):
			idx_pair = (words['id'].index(depends['oprnd'][i][0]), words['id'].index(depends['oprnd'][i][1]))
			match_rs = re.compile('(.+):.+\((.*)\)').match(depends['type'][i])
			dpnd_tp = ['', '']
			if (match_rs == None):
				match_rs = re.compile('(.+):.+').match(depends['type'][i])
			else:
				dpnd_tp[1] = match_rs.groups()[1]
			if (match_rs == None):
				match_rs = re.compile('(.+)').match(depends['type'][i])
			dpnd_tp[0] = match_rs.groups()[0]
			dpnd_mt[(idx_range[0]+idx_pair[0], idx_range[0]+idx_pair[1])] = dpnd_tp
			sub_dpnd_mt[(idx_pair[0], idx_pair[1])] = dpnd_tp
		# Construct the trigger feature column
		ft_trigger.extend([0 for x in range(len(words['id']))])

		if not (with_annot):
			continue

		# Get the events from a2 file
		events = get_a2(os.path.join(dir_path, a2_file))
		# Construct the event type feature column
		ft_evntp.extend(events['type'])
		# Connect the event operands to the corresponding words and annotate the trigger
		sdmt = dpnd_trnsfm(sub_dpnd_mt, (len(words['id']), len(words['id'])))
		dist_mt, prdcs = shortest_path(sdmt, directed=False, return_predecessors=True)
		if (trained):
			dist_mts[tuple(idx_range)] = dist_mt
			prdcss[tuple(idx_range)] = prdcs
		for i in xrange(len(events['id'])):
			loc_pairs = (annots['loc'][annots['id'].index(events['oprnds'][i][0])], annots['loc'][annots['id'].index(events['oprnds'][i][1])])
			idx_pair = (words['loc'].index(loc_pairs[0]), words['loc'].index(loc_pairs[1]))
			events['oprnds'][i] = [idx_range[0]+idx_pair[0], idx_range[0]+idx_pair[1]]
			# Find the trigger
			trigger = _find_trg(words, idx_pair[0], idx_pair[1], dist_mt, prdcs)
			if (trigger):
				ft_trigger[idx_range[0] + trigger] = 1
			# Generate the argument training samples
			if (trigger):
				if (trigger == idx_pair[0]):
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[1], dist_mt[trigger, idx_pair[1]]))
					ft_edgelb.append(events['type'][i])
				elif (trigger == idx_pair[1]):
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[0], dist_mt[trigger, idx_pair[0]]))
					ft_edgelb.append(events['type'][i])
				else:
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[0], dist_mt[trigger, idx_pair[0]]))
					edges.append((idx_range[0] + trigger, idx_range[0] + idx_pair[1], dist_mt[trigger, idx_pair[1]]))
					ft_edgelb.extend([events['type'][i]]*2)
		# Construct the event operands type column
		ft_evntoprt.extend(events['oprnd_tps'])
		# Construct the event operands hash column
		ft_evntoprd.extend(events['oprnds'])
		# Event statistics
		for eventp in events['type']:
			evntp_frqs[eventp] = evntp_frqs.setdefault(eventp, 0) + 1
		for evntoprts in events['oprnd_tps']:
			evntoprt_frqs[evntoprts[0]] = evntoprt_frqs.setdefault(evntoprts[0], 0) + 1
			evntoprt_frqs[evntoprts[1]] = evntoprt_frqs.setdefault(evntoprts[1], 0) + 1
		for evntoprds in events['oprnds']:
			evntoprd_frqs[ft_loc[evntoprds[0]]] = evntoprd_frqs.setdefault(ft_loc[evntoprds[0]], 0) + 1
			evntoprd_frqs[ft_loc[evntoprds[1]]] = evntoprd_frqs.setdefault(ft_loc[evntoprds[1]], 0) + 1
	if (trained):
		ft_offset.append(idx_range[1])

	## Construct word matrix
	ft_str_mt = np.array(ft_str, dtype='str').reshape(len(ft_str),1)
	ft_cap_mt = np.array(ft_cap).reshape(len(ft_cap),1)
	ft_pun_mt = np.array(ft_pun).reshape(len(ft_pun),1)
	ft_digit_mt = np.array(ft_digit).reshape(len(ft_digit),1)
	ft_loc_mt = np.array(ft_loc)
	ft_stem_mt = label_binarize(ft_stem, classes=list(STEM))
	ft_pos_mt = label_binarize(ft_pos, classes=list(POS))
#	ft_spos_mt = label_binarize(ft_spos, classes=list(POS))
	ft_annotp_mt = label_binarize(ft_annotp, classes=list(ANNOT_TYPE))
	word_mt = np.hstack((ft_cap_mt, ft_pun_mt, ft_digit_mt, ft_loc_mt, ft_pos_mt, ft_annotp_mt))
	wm_cols = ['has_cap', 'has_pun', 'has_num', 'loc_s', 'loc_e'] + ['pos_%s' % pos for pos in list(POS)] + ['annotp_%s' % atp for atp in list(ANNOT_TYPE)]
#	wm_cols = ['has_cap', 'has_pun', 'has_num', 'loc_s', 'loc_e'] + ['pos_%s' % pos for pos in list(POS)] + ['spos_%s' % pos for pos in list(POS)] + ['annotp_%s' % atp for atp in list(ANNOT_TYPE)]
	if (with_annot):
		# Construct trigger label
		trg_label = np.array(ft_trigger)
		# Construct trigger-argument pair sample matrix
		edge_data = np.array(edges)
		edge_mt = np.hstack((word_mt[edge_data[:,0].astype(int),:], word_mt[edge_data[:,1].astype(int),:]))
		em_cols = ['lf_%s' % col for col in wm_cols] + ['rt_%s' % col for col in wm_cols]
		edge_label = np.array(ft_edgelb)
	else:
		trg_label, edge_mt, em_cols, edge_label = [None] * 3
		
	# Combine all the data into Pandas DataFrame
	word_df = pd.DataFrame(word_mt, columns=wm_cols)
	edge_df = pd.DataFrame(edge_mt, columns=em_cols)
	trg_lb = pd.Series(trg_label, name='trigger_label')
	edge_lb = pd.Series(edge_label, name='edge_label')
	
	return word_df, trg_lb, edge_df, edge_lb


def test_normal():
	pca = PCA()
	kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
	
	## Training
	word_df, trg_label, edge_df, edge_label = get_data(TRAIN_PATH)
	pd.concat([word_df, trg_label], axis=1).to_csv(os.path.join(DATA_PATH, 'word.csv'))
	pd.concat([edge_df, edge_label], axis=1).to_csv(os.path.join(DATA_PATH, 'edge.csv'))
	#word_df = pca.fit_transform(word_df)
	trg_clf = svm.SVC()
	trg_clf.fit(word_df, trg_label)
	#trg_clf.fit(word_mt[trg_label==1 or word_mt[:,wft_head['ANNOT']]=='V'], trg_label)
	edge_clf = svm.SVC()
	edge_clf.fit(edge_df, edge_label)
	
	## Developing
	word_df_dev, trg_label_dev, edge_df_dev, edge_label_dev = get_data(DEV_PATH, trained=True)
	#word_df_dev = pca.fit_transform(word_df_dev)
	ptrg_label_dev = trg_clf.predict(word_df_dev)
	# Print out the precision, recall, and F-score
	print prfs(trg_label_dev, ptrg_label_dev, average='weighted')
	# Generate candidate edges
	cndt_edges = []
	ft_annotp = word_df_dev.filter(regex=('^annotp_.+'))
	for idx1, idx2 in zip(ft_offset[:-1], ft_offset[1:]):
		annots = ft_annotp[idx1:idx2]
		dist_mt = dist_mts[(idx1, idx2)]
		prdcs = prdcss[(idx1, idx2)]
		for wid in xrange(len(ptrg_label_dev[idx1:idx2])):
			if (ptrg_label_dev[idx1 + wid] == 1):
				cndt_edges.extend((ep1 + idx1, ep2 + idx1) for ep1, ep2 in gen_edges(wid, annots, dist_mt, prdcs))
	if (len(cndt_edges) > 0):
		edge_data = np.array(cndt_edges)
		cndt_edge_mt = np.hstack((word_mt_dev[edge_data[:,0].astype(int),:], word_mt[edge_data[:,1].astype(int),:]))
		pedge_label_dev = edge_clf.predict(cndt_edge_mt)
		# Print out the precision, recall, and F-score
		print prfs(edge_label_dev, pedge_label_dev)
		# Extract the edge to text

	## Testing
	#word_df_test, _, _, _ = get_data(TEST_PATH, trained=True, with_annot=False)
	#pred_trg_label = trg_clf.predict(word_df_test)


def test_complex():
	pass


def main():
	if len(sys.argv) < 2:
		print 'usage: %s [%s]' % (sys.argv[0], '|'.join(['normal', 'complex']))
	elif len(sys.argv) == 2:
		if (sys.argv[1] == 'normal'):
			test_normal()
		elif (sys.argv[1] == 'complex'):
			test_complex()
		else:
			print 'usage: %s [%s]' % (sys.argv[0], '|'.join(['normal', 'complex']))
	else:
		pass


if __name__ == '__main__':
	main()