#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: geo.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-05-25 18:28:46
###########################################################################
#

import os, sys

import GEOparse

from .. import nlp
from ..util import fs


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
GEO_PATH = os.path.join(DATA_PATH, 'geo')
SC=';;'


GSE_KEYMAP = {'platform_id':'platform', 'submission_date':'submit_date', 'last_update_date':'last_update', 'sample_id':'samples', 'geo_accession':'id', 'pubmed_id':'pmid', 'title':'title', 'summary':'summary'}
GSM_KEYMAP = {'platform_id':'platform', 'submission_date':'submit_date', 'last_update_date':'last_update', 'geo_accession':'id', 'title':'title', 'description':'description', 'data_processing':'data_processing', 'source_name_ch1':'source', 'organism_ch1':'organism', '':'tissue', '':'tissue_type', 'treatment_protocol_ch1':'treat_protocol', 'growth_protocol_ch1':'growth_protocol', 'extract_protocol_ch1':'extract_protocol', 'label_protocol_ch1':'label_protocol', 'label_ch1':'label', 'hyb_protocol':'hybrid_protocol', 'scan_protocol':'scan_protocol', 'characteristics_ch1':'trait', 'table':'data'}


def fetch_geo(accessions, saved_path=GEO_PATH, skip_cached=False):
	fs.mkdir(saved_path)
	geo_set = set(accessions)
	count = 0
	for acc in geo_set:
		cachef = os.path.join(saved_path, '%s_family.soft.gz' % acc)
		if (os.path.exists(cachef)):
			if (skip_cached): continue
			res = GEOparse.get_GEO(filepath=cachef)
		else:
			try:
				res = GEOparse.get_GEO(geo=acc, destdir=saved_path)
			except:
				print('Failed to fetch the GEO file: %s' % acc)
				continue
		count += 1
		yield res
	print("Number of obtained GEO documents: %i\n" % len(geo_set))


def parse_geo(gse, view='brief', with_samp=True):
	samples = []
	if (with_samp):
		for gsm_id, gsm in gse.gsms.items():
			samples.append((gsm_id, update_keys(gsm.metadata, type='GSM')))
	return (gse.get_accession(), update_keys(gse.metadata, type='GSE'), samples)

def parse_geos(gses, view='brief', with_samp=True):
	results = []
	for gse in gses:
		results.append(parse_geo(gse, view=view, with_samp=with_samp))
	return results

def update_keys(data, type='GSE'):
	data['ftype'] = 'SERIES'if type.upper() == 'GSE' else 'SAMPLE'
	key_map = GSE_KEYMAP if type.upper() == 'GSE' else GSM_KEYMAP
	for k, v in key_map.items():
		if (k in data):
			tmp = data[k]
			del data[k]
			data[v] = tmp
		else:
			data[v] = ''
			continue
		# Unroll some common attributes
		for key in ['platform', 'submit_date', 'last_update', 'id', 'title']:
			if (v == key):
				data[key] = data[key][0]
		if (type.upper() == 'GSE'):
			if (v == 'pmid'):
				data['pmid'] = data['pmid'][0]
			if (v == 'summary'):
				keywords = []
				paragraphs = data['summary']
				for i in range(len(paragraphs) - 1, -1, -1):
					if (paragraphs[i].startswith('Keywords')):
						phrases = paragraphs[i].split(':')
						if (len(phrases) == 1):
							phrases = paragraphs[i].split('=')
						keywords.append(phrases[1].strip())
					else:
						data['summary'] = '\n'.join(paragraphs[:i+1])
						break
				keywords.reverse()
				data['keywords'] = SC.join(keywords)
		elif (type.upper() == 'GSM'):
			# Unroll some GSM attributes
			for key in ['description', 'type', 'source', 'organism', 'treat_protocol', 'growth_protocol', 'extract_protocol', 'label_protocol', 'label', 'hybrid_protocol', 'scan_protocol']:
				if (v == key):
					data[key] = data[key][0]
			if (v == 'trait'):
				data[v] = SC.join(data[v])
	return data
