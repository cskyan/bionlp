#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: obo.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2020-06-15 13:04:25
###########################################################################
#

import os, sys, copy, time, subprocess
from datetime import datetime

import ftfy

# from ..util import fs
from bionlp.util import fs
# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp\\obo'
	CACHE_PATH = os.path.join(os.environ['TMPDIR'], 'obo') if ('TMPDIR' in os.environ and os.path.exists(os.environ['TMPDIR'])) else 'D:\\data\\bionlp\\obotmp\\'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp', 'obo')
	CACHE_PATH = os.path.join(os.environ['TMPDIR'], 'obo') if ('TMPDIR' in os.environ and os.path.exists(os.environ['TMPDIR'])) else '/tmp/obo'

OBO_HOME = os.environ.setdefault('OBO_HOME', 'OBO')
JAR_FPATH = os.path.join(OBO_HOME, "OBOAnnotatorNoGUI.jar")
CLASS = ['']
CLASSPATH = ':'.join([os.path.join(OBO_HOME, 'libs', jar) for jar in CLASS])


def annotext(text, ontos=[], cache_path=CACHE_PATH, clean_cache=True, encoding='utf-8'):
	text = '\n'.join(['0'] + ['\t%s' % txt.strip(' \t\n') for txt in text.split('\n')])
	ret_data = annotate(text, cache_path=cache_path, clean_cache=clean_cache)
	try:
		res = [dict(id=annot[0], loc=annot[2], text=annot[1]) for annot in ret_data['0']] if len(ret_data) > 0 else []
	except Exception as e:
		print(e)
		res = []
	return res


def annotexts(texts, ontos=[], cache_path=CACHE_PATH, clean_cache=True, encoding='utf-8'):
	merged_text = '\n'.join(['\n'.join([str(i)] + ['\t%s' % txt.strip(' \t\n') for txt in text.split('\n')] + ['\n']) for i, text in enumerate(texts)])
	ret_data = annotate(merged_text, cache_path=cache_path, clean_cache=clean_cache)
	try:
		res = [[dict(id=annot[0], loc=annot[2], text=annot[1]) for annot in ret_data[str(i)]] for i, text in enumerate(texts)]
	except Exception as e:
		print(e)
		res = []
	return res


def annotate(text, cache_path='.cache', clean_cache=False, encoding='utf-8'):
	fs.mkdir(cache_path)
	fname = str(os.getpid()) + str(datetime.timestamp(datetime.now())).replace('.', '')
	tmp_file = os.path.join(cache_path, fname + '.tmp')
	output_dir = os.path.join(cache_path, fname + '.out')
	fs.mkdir(output_dir)
	fs.write_file(text, tmp_file, encoding)
	curdir = os.getcwd()
	os.chdir(OBO_HOME)
	try:
		ret_code = subprocess.check_call('java -Xmx1024M -Xms128M -cp %s -jar %s %s %s' % (CLASSPATH, JAR_FPATH, tmp_file, output_dir), shell=True)
	except Exception as e:
		print(e)
	os.chdir(curdir)
	res = output_reader(os.path.join(output_dir, 'annotations.txt'))
	if (clean_cache):
		os.remove(tmp_file)
		os.remove(os.path.join(output_dir, 'annotations.txt'))
		os.remove(os.path.join(output_dir, 'annotationsCONCEPTS.txt'))
		os.rmdir(output_dir)
	return res


def output_reader(fpath):
	onto_prefix = 'HP_'
	res, record_id = {}, None
	with open(fpath, 'r') as fd:
		for line in fd.readlines():
			if line=='\n':
			    record_id = None
			elif record_id is None:
			    record_id = line.strip('\n')
			    res[record_id] = []
			else:
			    record = line.strip('\n').split('\t')
			    res[record_id].append((onto_prefix+record[0], record[1], (int(record[2]), int(record[3]))))
	return res


if __name__ == '__main__':
	text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
	print([a['id'] for a in annotext(text, ontos=['HP'])])
	texts = ['Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.', 'Normal chest x-XXXX.The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax.']
	print([[a['id'] for a in annots] for annots in annotexts(texts)])
