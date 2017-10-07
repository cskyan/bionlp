#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: dnorm.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-04-18 15:26:10
###########################################################################
#

import os
import sys
import time
import subprocess
# from collections import OrderedDict

import pandas as pd

from .. import nlp
from ..util import fs


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp\\dnorm'
	TMP_PATH = os.path.join(os.environ['TMPDIR'], 'dnorm') if (os.environ.has_key('TMPDIR') and os.path.exists(os.environ['TMPDIR'])) else 'D:\\data\\bionlp\\dnormtmp\\'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp', 'dnorm')
	TMP_PATH = os.path.join(os.environ['TMPDIR'], 'dnorm') if (os.environ.has_key('TMPDIR') and os.path.exists(os.environ['TMPDIR'])) else '/tmp/dnorm'

DNORM_HOME = os.environ.setdefault('DNORM_HOME', 'DNorm')
CONFIG = os.path.join(DNORM_HOME, 'config', 'banner_NCBIDisease_TEST.xml')
LEXICON= os.path.join(DNORM_HOME, 'data', 'CTD_diseases.tsv')
MATRIX= os.path.join(DNORM_HOME, 'output', 'simmatrix_NCBIDisease_e4.bin')
CLASS = ['dnorm.jar', 'colt.jar', 'lucene-analyzers-3.6.0.jar', 'lucene-core-3.6.0.jar', 'libs.jar', 'commons-configuration-1.6.jar', 'commons-collections-3.2.1.jar', 'commons-lang-2.4.jar', 'commons-logging-1.1.1.jar', 'banner.jar', 'dragontool.jar', 'heptag.jar', 'mallet.jar', 'mallet-deps.jar', 'trove-3.0.3.jar']
CLASSPATH = ':'.join([os.path.join(DNORM_HOME, 'libs', jar) for jar in CLASS])
	

def _gen_txt_list(text):
	index = 0
	for doc in text:
		for txt in doc.split('\n'):
			yield index, txt
			index += 1
def _gen_txt_flat(text):
	for i, txt in enumerate(text.split('\n')):
		yield i, txt
	
	
def annot_dss(text, keep_tmp=False):
	fs.mkdir(TMP_PATH)
	OUTPUT_PATH = os.path.join(TMP_PATH, 'output')
	fs.mkdir(TMP_PATH)
	fs.mkdir(OUTPUT_PATH)
	fname = str(int(time.time()))
	tmp_file = os.path.join(TMP_PATH, fname + '.tmp')
	output_f = os.path.join(OUTPUT_PATH, fname + '.out.tmp')
	if (type(text) is list):
		gen_txt = _gen_txt_list
	else:
		gen_txt = _gen_txt_flat
	with open(tmp_file, 'a') as f:
		for i, txt in gen_txt(text):
			f.write('\t'.join([str(i), txt]) + '\n')
	curdir = os.getcwd()
	os.chdir(DNORM_HOME)
	try:
		ret_code = subprocess.check_call('java -Xmx10G -Xms10G -cp %s dnorm.RunDNorm %s %s %s %s %s' % (CLASSPATH, CONFIG, LEXICON, MATRIX, tmp_file, output_f), shell=True)
	except Exception as e:
		if ('ret_code' in locals()):
			print 'Return code: %s' % ret_code
		print e
	os.chdir(curdir)
	df = pd.read_table(output_f, header=None, names=['id', 'start', 'end', 'concept', 'cid'], index_col=0)
	if (not keep_tmp):
		os.remove(tmp_file)
		os.remove(output_f)
	return df