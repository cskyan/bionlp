#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: hgnc.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-04-12 22:05:08
###########################################################################
#

import os
import sys
import time
import copy
import requests
import StringIO

import pandas as pd

from .. import nlp
from ..util import fs


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
HGNC_PATH = os.path.join(DATA_PATH, 'hgnc')


SYMBOL_URL = 'http://www.genenames.org/cgi-bin/symbol_checker'
MAX_TRIAL = None


def symbol_checker(text, match_case=False, approved_symbols=True, previous_symbols=False, synonyms=False, entry_withdrawn=False, show_unmatched=False):
	data = dict(data=text, format='text', submit='submit')
	data['case'] = 'sensitive' if match_case else 'insensitive'
	keymap = {'Approved symbol':approved_symbols, 'Previous symbol':previous_symbols, 'Synonyms':synonyms, 'Entry withdrawn':entry_withdrawn, 'Unmatched':show_unmatched}
	data['show_types'] = [k for k, v in keymap.iteritems() if v]
	trial = 0 if MAX_TRIAL is None else MAX_TRIAL
	while (MAX_TRIAL is None or trial > 0):
		res = requests.post(url=SYMBOL_URL, data=data)
		if (res.ok	and res.status_code == 200):
			break
		trial -= 1
	else:
		raise RuntimeError('Cannot connect to the service!')
	txt = StringIO.StringIO(res.text)
	try:
		return pd.read_table(txt)
	except Exception as e:
		print e
		fname = str(int(time.time()))
		fs.write_file(text, 'symbol_checker_%s.in' % fname, code='utf-8')
		fs.write_file(res.text, 'symbol_checker_%s.out' % fname, code='utf-8')
		return pd.DataFrame([])	