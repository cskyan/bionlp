#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: mer.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2020-04-30 02:06:13
###########################################################################
#

import os, sys, json, copy, requests

import ftfy
import merpy
merpy.download_lexicons()

# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'mer')
SC=';;'


INITED_ONTOS = set([])


def annotext(text, ontos=[], tool='strbase', negex=True, **kwargs):
	lexicons = merpy.get_lexicons()
	entities = []
	for onto in ontos:
		onto_lwr, onto_upr = onto.lower(), onto.upper()
		if not all([onto_lwr in lxcn for lxcn in lexicons]): continue
		if onto_lwr not in INITED_ONTOS:
			merpy.process_lexicon(onto_lwr)
			INITED_ONTOS.add(onto_lwr)
		ents = [dict(id=r[-1].split('/')[-1], text=r[2], loc=(r[0], r[1])) for r in merpy.get_entities(text, onto_lwr) if len(r) >= 4]
		ents = [ent for ent in ents if ent['id'].startswith(onto_upr)]
		entities.extend(ents)
	return entities


if __name__ == '__main__':
    text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
    print([a['id'] for a in annotext(text, ontos=['HP'])])
