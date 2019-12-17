#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: mti.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-12-14 17:08:16
###########################################################################
#

import os, sys, json, copy
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd

import ftfy
from py4j.java_gateway import JavaGateway
try:
	GATEWAY = JavaGateway()
	GATEWAY.jvm.py4j.GatewayServer.turnLoggingOn()
except Exception as e:
	print('Gateway Server is not available!')

from ..util import fs
# from bionlp.util import fs
# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'clinphen')
SC=';;'


def annotext(text, ontos=[], cache_path='.cache', encoding='ascii'):
	timestamp = int(datetime.timestamp(datetime.now()))
	fs.mkdir(cache_path)
	inpath = os.path.join(cache_path, '%s.in' % timestamp)
	fs.write_file(text, inpath, encoding)
	df = pd.read_csv(StringIO(GATEWAY.entry_point.get_result(os.path.abspath(inpath))), sep='|', header=None, error_bad_lines=False)
	try:
		res = [dict(id=r[2], loc=tuple(np.cumsum(list(map(int, r[8].split('^')[:2])))), name=r[5].lstrip('RtM via: ').split(';')[0] if type(r[5]) is str else '', text=r[1].lstrip('*') if type(r[1]) is str else '') for i, r in df.iterrows()]
	except Exception as e:
		print(e)
		print(df)
		res = []
	return res


def annotexts(texts, ontos=[], cache_path='.cache', encoding='ascii'):
	return [annotext(text, ontos, cache_path, encoding) for text in texts]


def batch_annotexts(texts, ontos=[], cache_path='.cache', encoding='ascii'):
	timestamp = int(datetime.timestamp(datetime.now()))
	inpath, outpath = os.path.join(cache_path, 'mtiin%i' % timestamp), os.path.join(cache_path, 'mtiout%i' % timestamp)
	_ = [fs.mkdir(x) for x in [inpath, outpath]]
	_ = [fs.write_file(texts[i], os.path.join(inpath, '%i' % i), encoding) for i in range(len(texts))]
	GATEWAY.entry_point.batch(os.path.abspath(inpath), os.path.abspath(outpath), 10)
	dfs = [pd.read_csv(os.path.join(outpath, '%i' % i), sep='|', header=None, error_bad_lines=False) for i in range(len(texts))]
	return [[dict(id=r[2], loc=tuple(np.cumsum(list(map(int, r[8].split('^')[:2])))), name=r[5].lstrip('RtM via: ').split(';')[0] if type(r[5]) is str else '', text=r[1].lstrip('*') if type(r[1]) is str else '') for i, r in df.iterrows()] for df in dfs]


if __name__ == '__main__':
    # text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
    # print([a['id'] for a in annotext(text)])
	texts = ['Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.', 'Normal chest x-XXXX.The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax.']
	print([[a['id'] for a in annots] for annots in annotexts(texts)])
