#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: biogrid.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-04-07 11:31:46
###########################################################################
#

import os, sys, copy, time, json

from apiclient import APIClient
# from abc import ABCMeta

from .. import nlp


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
BIOGRID_PATH = os.path.join(DATA_PATH, 'biogrid')


class BioGRIDAPI(APIClient, object):
	BASE_URL = 'http://webservice.thebiogrid.org/'
	_function_url = {'interaction':'interaction'}
	_default_param = {'interaction':dict(format='json', searchNames='true', includeInteractors='true', includeInteractorInteractions='false', taxId='9606|10090|10116', geneList='')}
	def __init__(self, function='interaction', api_key=''):
		if (function not in self._default_param):
			raise ValueError('The function %s is not supported!' % function)
		if (not api_key or api_key.isspace()):
			raise ValueError('The accesskey cannot be empty!')
		# super(APIClient, self).__init__()
		# APIClient is old-stype class
		APIClient.__init__(self)
		self.function = function
		self.func_url = self._function_url[function]
		self.api_key = api_key
		for v in self._default_param.values():
			v['accesskey'] = api_key
	def _handle_response(self, response):
		return json.loads(nlp.clean_text(response.data, encoding='utf-8', replacement=None).replace('\\', ''))

	def call(self, max_trail=-1, interval=3, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.items() if k in args)
		trail = 0
		while max_trail <= 0 or trail < max_trail:
			try:
				res = APIClient.call(self, '/%s'% self.func_url, **args)
				break
			except Exception as e:
				print(e)
				time.sleep(interval)
				trail += 1
		return res

	def get_orgnsm(self):
		args = copy.deepcopy(self._default_param[self.function])
		return APIClient.call(self, '/organisms', **args)
