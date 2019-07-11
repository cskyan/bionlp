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

import os, sys, subprocess
from collections import OrderedDict

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

import ftfy

from .. import nlp
from ..util import fs, io, oo, func
# from bionlp import nlp
# from bionlp.util import fs, io, oo, func

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\ontolib\\store'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'ontolib', 'store')
ANT_PATH = os.path.join(DATA_PATH, 'metamap')
SC=';;'

ONTO_MAPS = {'HP':'HPO'}


def annotext(text, ontos=[]):
	try:
		client = Wrapper()
	except Exception as e:
		print(e)
		Wrapper.start_service()
		client = Wrapper()
	res = client.raw_parse(text, src=[ONTO_MAPS[x] for x in ontos])
	return [dict(id=concept.cui, score=concept.score) for concept in func.flatten_list(list(res[0].values())) if hasattr(concept, 'cui')]


class Wrapper():
	@staticmethod
	def start_service(srv=(True, True)):
		if (srv[0]):
			subprocess.call('skrmedpostctl start', shell=True)
		if (srv[1]):
			subprocess.call('wsdserverctl start', shell=True)

	@staticmethod
	def restart_service(srv=(True, True)):
		if (srv[0]):
			subprocess.call('skrmedpostctl restart', shell=True)
		if (srv[1]):
			subprocess.call('wsdserverctl restart', shell=True)

	@staticmethod
	def stop_service(srv=(True, True)):
		if (srv[0]):
			subprocess.call('skrmedpostctl stop', shell=True)
		if (srv[1]):
			subprocess.call('wsdserverctl stop', shell=True)

	@staticmethod
	def status():
		tag_srv, wsd_srv = int(subprocess.check_output('ps -ef | grep taggerServer | wc -l', shell=True)) - 2, int(subprocess.check_output('ps -ef | grep DisambiguatorServer | wc -l', shell=True)) - 2
		return tag_srv, wsd_srv

	def __init__(self):
		from pymetamap import MetaMap
		self.mm = MetaMap.get_instance(os.path.join(os.environ['MM_HOME'], 'bin', 'metamap'))

	def __del__(self):
		del self.mm

	def __enter__(self):
		Wrapper.start_service([1 - x for x in Wrapper.status()])
		return self

	def __exit__(self, type, value, traceback):
		Wrapper.stop_service()

	def _post_process(self, concepts, error):
		result = OrderedDict()
		for concept in concepts:
			result.setdefault(int(concept.index) - 1, []).append(concept)
		return result, error

	def raw_parse(self, text, src=[]):
		import spacy
		spacy_nlp = spacy.load('en_core_web_sm')
		doc = spacy_nlp(ftfy.fix_text(text).encode('ascii', 'replace').decode('ascii'))
		sents = [str(sent) for sent in doc.sents]
		return self._post_process(*self.mm.extract_concepts(sents, range(1, len(sents) + 1), restrict_to_sources=src))

	def parse(self, tokens):
		return self._post_process(*self.mm.extract_concepts(tokens, range(1, len(tokens) + 1)))


def get_mesh_from_file(pmids):
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
	return mesh_term_list


def mesh_countvec(pmids, from_file=None, ft_type='binary', max_df=1.0, min_df=1, fmt='npz', spfmt='csr'):
	mesh_term_list = get_mesh_from_file(pmids)
	union_mesh_terms = list(set().union(*mesh_term_list))
	mlb = MultiLabelBinarizer(classes=union_mesh_terms)
	mesh_mt = (mlb.fit_transform(mesh_term_list), mlb.classes_)
	mesh_df = pd.DataFrame(mesh_mt[0], index=pmids, columns=mesh_mt[1])
	if (fmt == 'npz'):
		io.write_df(mesh_df, os.path.join(DATA_PATH, 'mesh.npz'), with_idx=True, sparse_fmt=spfmt, compress=True)
	else:
		mesh_df.to_csv(os.path.join(DATA_PATH, 'mesh.csv'), encoding='utf8')
	return mesh_df


if __name__ == '__main__':
	text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
	print([a['id'] for a in annotext(text, ontos=['HP'])])
