#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: phen2gene.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-07-01 01:05:35
###########################################################################
#

import os, sys, json, copy, time, hashlib, requests

from bionlp.util import fs, io


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
P2G_PATH = os.path.join(DATA_PATH, 'phen2gene')
SC=';;'


BASE_URL = 'https://phen2gene.wglab.org/api'
FUNC_URL = {}
WEIGHT_MODEL = {'sk': 'skewness', 'w': 'ontology-based information content', 'ic':'information content by Sanchez', 'u':'unweighted'}


def _get_ranked_genes(query_hpos, weight=None, interval=0, cache_path=P2G_PATH, skip_cache=False):
	query_hpos = sorted(set(query_hpos))
	query_url = '%s?HPO_list=%s%s' % (BASE_URL, ';'.join(query_hpos).replace('_', ':'), ('&weight_model=%s' % weight) if weight in WEIGHT_MODEL else '')
	hpo_ids_str = '_'.join(map(lambda x: x.lstrip('HP:_'), query_hpos))
	cachef = os.path.join(cache_path, '%s%s.json' % (weight+'_' if weight in WEIGHT_MODEL else '', hashlib.md5(hpo_ids_str.encode('utf-8')).hexdigest()))
	if (os.path.exists(cachef) and not skip_cache):
		res = io.read_json(cachef)
	else:
		res = requests.get(query_url).json()
		io.write_json(res, cachef)
		time.sleep(interval)
	return res

def get_ranked_genes(query_hpos, weight=None, ret_score=False, interval=0, cache_path=P2G_PATH, skip_cache=False):
	res = _get_ranked_genes(query_hpos, weight=weight, interval=interval, cache_path=cache_path, skip_cache=skip_cache)
	rgs = sorted([(int(r['Rank']), r['Gene'], r['Score']) for r in res['results']], key=lambda x: x[0])
	return [r[1:] for r in rgs] if ret_score else [r[1] for r in rgs]

def get_topk_genes(query_hpos, k, weight=None, ret_score=False, interval=0, cache_path=P2G_PATH, skip_cache=False):
	res = _get_ranked_genes(query_hpos, weight=weight, interval=interval, cache_path=cache_path, skip_cache=skip_cache)
	rgs = sorted([(int(r['Rank']), r['Gene'], r['Score']) for r in res['results'] if int(r['Rank']) <= k], key=lambda x: x[0])
	return [r[1:] for r in rgs] if ret_score else [r[1] for r in rgs]


if __name__ == '__main__':
    query_hpos = ['HP_0002459']
    print(get_ranked_genes(query_hpos, ret_score=True))
    print(get_topk_genes(query_hpos, 5, ret_score=True))
