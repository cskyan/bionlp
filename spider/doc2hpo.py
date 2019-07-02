#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: doc2hpo.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-07-01 01:05:35
###########################################################################
#

import os, sys, json, copy, requests

import ftfy

# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'doc2hpo')
SC=';;'


BASE_URL = 'https://impact2.dbmi.columbia.edu/doc2hpo/parse'
FUNC_URL = {'strbase':'acdat', 'mmlite':'metamaplite', 'ncbo':'ncbo'}


def annotext(text, ontos=[], tool='strbase', negex=True, **kwargs):
    url = '%s/%s' % (BASE_URL, FUNC_URL[tool])
    params = dict(note=text, negex=negex, **kwargs)
    res = requests.post(url, json=params).json()['hmName2Id']
    return [dict(id=r['hpoId'], loc=(r['start'], r['start'] + r['length']), text=r['hpoName'], negated=r['negated']) for r in res]


if __name__ == '__main__':
    text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
    print([a['id'] for a in annotext(text, ontos=['HP'], tool='mmlite')])
