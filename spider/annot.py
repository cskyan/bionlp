#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: annot.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-02-14 21:44:35
###########################################################################
#

import os, sys, json, bisect

import becas

from ..util import fs, func, ontology
from .. import nlp


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'annot')
SC=';;'


def init():
	becas.email = 'you@example.com'


# Annotate the text as entirety
def annotext(text, retype='dict', with_mdf=False):
	if (type(text) is dict):
		results = text
	else:
		results = becas.annotate_text(text)
	if (retype == 'dict'):
		return results
	elif (retype == 'group'):
		if (results.setdefault('text', '') == ''):
			with_mdf = False
		if (with_mdf):
			init_tokens, locs = nlp.tokenize(results['text'], model='word', ret_loc=True)
			tokens, locs = nlp.del_punct(init_tokens, location=locs)
			pos_tags = nlp.pos(tokens)
		groups = {}
		for entity in results['entities']:
			word, uid_str, offset = entity.split('|')
			mdf = ''
			if (with_mdf):
				start_loc, end_loc = zip(*locs)
				tkn_id = bisect.bisect_left(list(start_loc), int(offset))
				if (tkn_id > 0 and (pos_tags[tkn_id - 1][1] == 'JJ' or pos_tags[tkn_id - 1][1] == 'NN' or pos_tags[tkn_id - 1][1] == 'NNP')):
					mdf = tokens[tkn_id - 1]
			for uid in uid_str.split(';'):
				uid_list = uid.split(':')
				src, ids, tp = uid_list[0], uid_list[1:-1], uid_list[-1]
				groups.setdefault(tp, []).append(dict(src=src, ids=ids, word=word, offset=offset, modifier=mdf))
		return groups


# Annotate each sentences separately
def exportext(text, fpath='annot', fmt='json'):
	content = becas.export_text(text, fmt)
	fs.write_file(content, os.path.splitext(fpath)[0] + '.' + fmt, code='utf8')
	return content


def annotabs(pmid, retype='dict'):
	results = becas.annotate_publication(pmid)
	if (retype == 'dict'):
		return results
	elif (retype == 'group'):
		groups = {}
		for entity in results['entities_title'] + results['entities_abstract']:
			word, uid, offset = entity.split('|')
			uid_list = uid.split(':')
			src, ids, tp = uid_list[0], uid_list[1:-1], uid_list[-1]
			groups.setdefault(tp, []).append(dict(src=src, ids=ids, word=word, offset=offset))
		return groups


def exportabs(pmid, fpath='annot'):
	content = becas.export_publication(pmid)
	fs.write_file(content, os.path.splitext(fpath)[0] + '.xml', code='utf8')
	return content


def annotonto(text, ontog, lang='en', idns='', prdns=[], idprds={}, dominant=False, lbprds={}):
	annotations = []
	init_tokens, locs = nlp.tokenize(text, model='word', ret_loc=True)
	if (len(init_tokens) == 0): return annotations
	try:
		tokens, locs = nlp.del_punct(init_tokens, location=locs)
	except:
		tokens, locs = init_tokens, locs
	for token, loc in zip(tokens, locs):
		idlabels = ontology.get_id(ontog, token, lang=lang, idns=idns, prdns=prdns, idprds=idprds)
		if (dominant):
			annotations.extend(func.flatten_list([[(id, idlb, token, loc) for idlb in ontology.get_label(ontog, id, lang=lang, idns=idns, prdns=prdns, lbprds=lbprds)] for id, label in idlabels]))
		else:
			annotations.extend([(id, label, token, loc) for id, label in idlabels])
	return annotations
